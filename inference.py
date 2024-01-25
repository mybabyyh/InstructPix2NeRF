# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from download import find_model
from ditmodels import DiT_models
import argparse
from utils.model_utils import setup_model
from utils.common import tensor2im
import math
import os
import numpy as np
import imageio
import scipy.interpolate
from tqdm import tqdm
from PIL import Image,ImageDraw,ImageFont
from camera_utils import LookAtPoseSampler
# from gfpgan import GFPGANer
# from basicsr.archs.rrdbnet_arch import RRDBNet
# from realesrgan import RealESRGANer
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils import data_utils
# from utils.constant import ATTRIBUTES, ATTRIBUTES_RANGE
from criteria import id_loss
from models.classification import Classification
import json
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import torch.nn.functional as F
from utils import common, train_utils
import random


class ClassConfig:
    model_name: str = "vit_small_patch16_224"
    pretrained: bool = True
    n_classes: int = 40
    lr: int = 0.00001

class InferenceDataset(Dataset):

    def __init__(self, data_root, camera_param, transform=None):
        self.data_paths = sorted(data_utils.make_dataset(data_root))        
        self.transform = transform
        # load camera param
        self.camera_dict={}
        with open(camera_param, 'r', encoding='utf8') as fp:
            camera_param = json.load(fp)
            for img in camera_param['labels']:
                self.camera_dict[img[0]] = torch.tensor(img[1])


    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        img_name = os.path.basename(self.data_paths[index])
        img = Image.open(self.data_paths[index])
        img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)
        # return img, img_name, self.camera_dict[f'{img_name.split("_")[0]}.png']
        return img, img_name, self.camera_dict[img_name]


def parse_and_log_images(id_logs, x, y, y_hat, prompt, log_dir, title, step, subscript=None, display_count=None):
    im_data = []
    if display_count is None:
        display_count = x.shape[0]
    for i in range(display_count):
        cur_im_data = {
            'input_face': common.tensor2im(x[i]),
            'target_face': common.tensor2im(y[i]),
            'output_face': common.tensor2im(y_hat[i]),
            'prompt': prompt[i],
        }
        if id_logs is not None:
            for key in id_logs[i]:
                cur_im_data[key] = id_logs[i][key]
        im_data.append(cur_im_data)
    log_images(log_dir, title, step, im_data=im_data, subscript=subscript)

def log_images(log_dir, name, step, im_data, subscript=None, log_latest=False):
    fig = common.vis_faces(im_data)
    if log_latest:
        step = 0
    if subscript:
        path = os.path.join(log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
    else:
        path = os.path.join(log_dir, name, '{:04d}.jpg'.format(step))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)




def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    # latent_size = args.image_size // 8
    latent_size = 4
    in_channels = 512
    model = DiT_models[args.model](
        input_size=latent_size,
    ).to(device)
    
    ckpt_path = args.ckpt
    print(f'ckpt_path: {ckpt_path}')
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()  # important!
    diffusion = create_diffusion(f'ddim{str(args.num_sampling_steps)}')

    net, _ = setup_model(args.preim3d_ckpt, args.eg3d_weight, device)
    generator = net.decoder
    generator.eval()
    
    # ID loss 
    id_loss_ = id_loss.IDLoss().to(device).eval()

    transform = transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


    t_start = int(args.strength * args.num_sampling_steps)

   
    prompt_dict=[]
    with open(args.prompt, 'r', encoding='utf8') as fp:
        prompt_param = json.load(fp)
        for img in prompt_param['prompt']:
            prompt={}
            prompt['img_name'] = img[0]
            prompt['text_scale'] = img[1]
            prompt['img_scale'] = img[2]
            prompt['text'] = img[3]
            prompt_dict.append(prompt)   

    sample_count = 0
    for item in prompt_dict:
        img_name = item['img_name']
        if sample_count >= args.sample_num:
            break
        img_path = os.path.join(args.image_dir, img_name)

        sample_count += 1
        
        img_in = Image.open(img_path)

        img_in.save(f'./data/test/{img_name[:-4]}.jpg')
        img_in = img_in.convert('RGB')
        img_in = transform(img_in)

        img_in = img_in.unsqueeze(0)
        input_prompt = item['text']
        text_scale = item['text_scale']
        img_scale = item['img_scale']

        x = img_in.to(device)
        
        if os.path.exists(img_path):
            for cnt in range(1):
                text_scale = item['text_scale']
                img_scale = item['img_scale']
                save_path = os.path.join(args.save_dir, input_prompt, img_name[:-4])
                os.makedirs(save_path, exist_ok=True)


                batch_size = x.shape[0]

                class_labels = [1 for i in range(batch_size)]

                codes = net.encoder(x)
                w = codes.unsqueeze(1).permute(0,3,1,2) 
                padding = torch.zeros(w.shape[:-1]).unsqueeze(3).repeat(1,1,1,2).to(w.device)
                w = torch.cat((w, padding), dim=3)
                shape = (w.shape[0], w.shape[1], 4, 4)
                w = w.reshape(shape)
                
                # Create sampling noise:
                z = torch.randn(batch_size, in_channels, latent_size, latent_size, device=device)
                y = torch.tensor(class_labels, device=device)

                t = torch.tensor([t_start] * w.shape[0], device=device)
                noise = torch.randn_like(w)
                w_t = diffusion.q_sample(w, t, noise=noise)
                
                uncond = torch.zeros_like(z)
                cond = torch.cat([w, w, uncond], 0)
                # Setup classifier-free guidance:
                z = torch.cat([w_t, w_t, w_t], 0)
                y_null = torch.tensor([1] * batch_size, device=device)
                y = torch.cat([y, y, y_null], 0)

                prompt = [input_prompt for i in range(batch_size)]
                prompt_null = ['' for i in range(2*batch_size)]
                prompt += prompt_null

                idfeatures = None
                if args.id_cond:
                    idfeatures = id_loss_.extract_feats(x)
                    idfeatures_null = torch.zeros_like(idfeatures, device=device)
                    idfeatures = torch.cat([idfeatures, idfeatures, idfeatures_null], 0)
                
                model_kwargs = dict(y=y, prompt=prompt, idfeatures=idfeatures, cond=cond, cfg_text_scale=text_scale, cfg_img_scale=img_scale)

                # Sample images:
                samples = diffusion.ddim_decode(
                    model.forward_with_cfg_text_and_cfg_img, z.shape, z, t_start, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                )
                samples, _, _ = samples.chunk(3, dim=0)  # Remove null class samples

                shape = (samples.shape[0], samples.shape[1], 1, 16)
                latents = samples.reshape(shape)[:,:,:,:14].permute(0,2,3,1).squeeze(1)

                latents = latents + net.latent_avg.repeat(latents.shape[0], 1, 1)

                yaw = random.uniform(math.pi / 9, math.pi / 9)
                pitch = random.uniform(-math.pi / 18, math.pi / 18)
                camera_lookat_point = torch.tensor(generator.rendering_kwargs['avg_camera_pivot'], device=device)
                cam2world_pose = LookAtPoseSampler.sample(math.pi / 2 + yaw, math.pi / 2 + pitch, camera_lookat_point,
                                                            radius=generator.rendering_kwargs['avg_camera_radius'],
                                                            device=device)
                focal_length = 4.2647
                intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]],
                                            device=device)
                c_rand = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                decoder_batch_size = batch_size
                steps = batch_size // decoder_batch_size

                for idx in range(steps):
                    latents_batch = latents[idx * decoder_batch_size:(idx+1)*decoder_batch_size,:,:]

                    imgs = generator.synthesis(latents_batch, c_rand.repeat(decoder_batch_size, 1), noise_mode='const')['image']
                    
                    for b_idx, img in enumerate(imgs):
                        result = tensor2im(img)
                        img_save_path = os.path.join(save_path, f"{0}_{0}_{text_scale:.1f}_{img_scale:.1f}_{img_name[:-4]}_{cnt}_1.jpg")
                        result.save(img_save_path)
                        torch.save(latents_batch[b_idx:b_idx+1], os.path.join(save_path, f"{0}_{0}_{text_scale:.1f}_{img_scale:.1f}_{img_name[:-4]}_{cnt}_latent.pt"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-B_G/1_pair")
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None, help="path to a InstructPix2NeRF checkpoint.")
    parser.add_argument("--preim3d-ckpt", type=str, default=None)
    parser.add_argument("--eg3d-weight", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--decoder-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--sample_num", type=int, default=10)
    parser.add_argument("--id-cond", action="store_true", help="id condition")
    args = parser.parse_args()
    main(args)
