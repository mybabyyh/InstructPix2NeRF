# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import json
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import torch.nn.functional as F

from utils import common, train_utils


import math
from utils.model_utils import setup_model

from ditmodels import DiT_models
from diffusion import create_diffusion
from torch.utils.data import Dataset
from camera_utils import LookAtPoseSampler
from criteria import id_loss
from criteria.lpips.lpips import LPIPS
import random


class TextPairImagesDataset(Dataset):

    def __init__(self, data_root, camera_param, transform=None):
        # self.data_paths = sorted(data_utils.make_dataset(data_root))
        
        self.transform = transform

        # load camera param
        self.camera_dict={}
        with open(camera_param, 'r', encoding='utf8') as fp:
            camera_param = json.load(fp)
            for img in camera_param['labels']:
                # self.camera_dict[img[0]] = img[1]
                # print('torch.tensor(img[1]):', torch.tensor(img[1]).shape)
                self.camera_dict[img[0].split('.')[0]] = torch.tensor(img[1])
                # print(f'{img[0].split(".")[0][-5:]}:', torch.tensor(img[1]).unsqueeze(0).shape)

        self.text_img_pair = []
        for data_path in os.listdir(data_root):
            # data_path = os.
            # print('img_path: ', data_path)
            if os.path.isdir(os.path.join(data_root, data_path)):
                img_0_name = os.path.join(data_root, data_path, f'{data_path}_0.png')
                img_1_name = os.path.join(data_root, data_path, f'{data_path}_1.png')
                text_path = os.path.join(data_root, data_path, f'{data_path}.txt')
                # print('text_path: ', text_path)
                with open (text_path,  "r") as file:
                    for line in file:
                        self.text_img_pair.append((img_0_name, img_1_name, line.strip()))
    
                
        print('dataset len:', len(self.text_img_pair))

    def __len__(self):
        return len(self.text_img_pair)

    def __getitem__(self, index):

        img_0_path = self.text_img_pair[index][0]
        img_1_path = self.text_img_pair[index][1]
        text = self.text_img_pair[index][2]
        img_0 = Image.open(img_0_path)
        img_1 = Image.open(img_1_path)
        img_0 = img_0.convert('RGB')
        img_1 = img_1.convert('RGB')

        if self.transform:
            img_0 = self.transform(img_0)
            img_1 = self.transform(img_1)

        y = 0
        img_name = os.path.basename(img_0_path).split('_')[0]

        c = self.camera_dict[img_name]
        return img_0, img_1, text, c, y, img_name


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def identity_loss(generator, x0, x1):
    
    return None


def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img


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


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        log_images_dir = f"{experiment_dir}/images"  # Stores saved log images
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_images_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    # latent_size = args.image_size // 8
    latent_size = 4
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )

    # load checkpoint weight
    if args.ckpt_path is not None:
        checkpoint = torch.load(args.ckpt_path, map_location=lambda storage, loc: storage)
        if "ema" in checkpoint:  # supports checkpoints from train.py
            checkpoint = checkpoint["ema"]
        model.load_state_dict(checkpoint, strict=False)
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")


    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule

    net, _ = setup_model(args.preim3d_ckpt, args.eg3d_weight, device)
    encoder = net.encoder
    encoder.eval()
    generator = net.decoder
    generator.eval()
    for p in generator.parameters():
        p.requires_grad = True
    
    # ID loss 
    id_loss_ = id_loss.IDLoss().to(device).eval()

    # LPIPS 
    lpips_loss = LPIPS(net_type='alex').to(device).eval()


    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)


    transform = transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])

    dataset = TextPairImagesDataset(args.data_path, args.camera_param, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x0, x1, text, c, y, img_name in loader:
            # print(f'x: {x}   y: {y}')
            x0 = x0.to(device)
            x1 = x1.to(device)
            c = c.to(device)

            x0_init = x0.clone()
            x1_init = x1.clone()
            y = y.to(device)
            prompt = text
            # print('prompt: ', prompt)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                codes0 = encoder(x0)
                x0 = codes0.unsqueeze(1).permute(0,3,1,2) 
                padding = torch.zeros(x0.shape[:-1]).unsqueeze(3).repeat(1,1,1,2).to(x0.device)
                x0 = torch.cat((x0, padding), dim=3)
                shape = (x0.shape[0], x0.shape[1], 4, 4)
                x0 = x0.reshape(shape)

                
                codes1 = encoder(x1)
                x1 = codes1.unsqueeze(1).permute(0,3,1,2)
                padding = torch.zeros(x1.shape[:-1]).unsqueeze(3).repeat(1,1,1,2).to(x1.device)
                x1 = torch.cat((x1, padding), dim=3)
                shape = (x1.shape[0], x1.shape[1], 4, 4)
                x1 = x1.reshape(shape)
                # print('x: ', x.shape)
                # print('vae.encode(x): ', vae.encode(x).shape)
                # x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                # print('x: ', x.shape)
            t = torch.randint(0, diffusion.num_timesteps, (x1.shape[0],), device=device)
            # attr = attribute_classifier(torch.nn.functional.interpolate(torch.clamp(imgs_gen, -1., 1.), size=(224, 224), mode='bilinear'))
            
            idfeatures = id_loss_.extract_feats(x0_init)
            # print('idfeatures: ', idfeatures.shape)

            model_kwargs = dict(y=y, prompt=prompt, idfeatures=idfeatures)
            # model_kwargs = dict(prompt=prompt)
            loss_dict = diffusion.training_losses(model=model, x_start=x1, t=t, cond=x0, model_kwargs=model_kwargs)

            # add ID regularization
            target = loss_dict['target']
            model_output = loss_dict['model_output']
            pred_xstart = loss_dict['pred_xstart']
            sample = loss_dict['sample']



            shape = (pred_xstart.shape[0], pred_xstart.shape[1], 1, 16)
            latents = pred_xstart.reshape(shape)[:,:,:,:14].permute(0,2,3,1).squeeze(1)
          
            latents = latents + net.latent_avg.repeat(latents.shape[0], 1, 1)
          
            codes1_start = codes1 + net.latent_avg.repeat(codes1.shape[0], 1, 1)
            
            yaw = 0
            pitch = 0
            camera_lookat_point = torch.tensor(generator.rendering_kwargs['avg_camera_pivot'], device=device)
            cam2world_pose = LookAtPoseSampler.sample(math.pi / 2 + yaw, math.pi / 2 + pitch, camera_lookat_point,
                                                        radius=generator.rendering_kwargs['avg_camera_radius'],
                                                        device=device)
            focal_length = 4.2647
            intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]],
                                        device=device)
            c0 = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            gen_batch = args.gen_batch_size

            denoise_step_th = 600

            latents_select = latents[t < denoise_step_th, :, :]
            x0_init_select = x0_init[t < denoise_step_th, :, :]
            x1_init_select = x1_init[t < denoise_step_th, :, :]
            prompt_select = [prompt[i] for i, ti in enumerate(t) if ti < denoise_step_th]
            c_select = c.repeat(latents_select.shape[0], 1)

            batch_rate = latents_select.shape[0] // gen_batch
            batch_rate = batch_rate + 1 if latents_select.shape[0] % gen_batch != 0 else batch_rate
            

            id_loss_x0 = 0
            l2_loss_x0 = 0
            lpips_loss_x0 = 0
            cur_batch = 0

            if args.lambda_id > 0:
                for i in range(batch_rate): 
                    # with torch.no_grad():
                    gen_batch_final = gen_batch
                    if i == batch_rate - 1:
                        gen_batch_final = latents_select.shape[0] - cur_batch
                    
                    c_final = c0.repeat(gen_batch_final, 1)
                    latents_final = latents_select[cur_batch:cur_batch + gen_batch_final,:,:]
                    x1_batch = x1_init_select[cur_batch:cur_batch + gen_batch_final,:]
                    x0_batch = x0_init_select[cur_batch:cur_batch + gen_batch_final,:]

                    imgs = generator.synthesis(latents_final, c_final, noise_mode='const')['image']
                                    
                    cur_batch += gen_batch_final
                    imgs = torch.nn.functional.interpolate(imgs, size=(256, 256), mode='bilinear') 

                    if args.lambda_id > 0:
                        id_losses, _, id_logs = id_loss_(imgs, x1_batch, x0_batch)
                        id_loss_x0 += id_losses
            if args.lambda_id > 0 or args.lambda_lpips > 0:
                cur_batch = 0
                for i in range(batch_rate): 
                    # with torch.no_grad():
                    gen_batch_final = gen_batch
                    if i == batch_rate - 1:
                        gen_batch_final = latents_select.shape[0] - cur_batch
                    
                    c_final = c_select[cur_batch : cur_batch + gen_batch_final,:]
                    latents_final = latents_select[cur_batch:cur_batch + gen_batch_final,:,:]
                    x1_batch = x1_init_select[cur_batch:cur_batch + gen_batch_final,:]
                    x0_batch = x0_init_select[cur_batch:cur_batch + gen_batch_final,:]

                    imgs = generator.synthesis(latents_final, c_final, noise_mode='const')['image']
                                    
                    cur_batch += gen_batch_final
                    imgs = torch.nn.functional.interpolate(imgs, size=(256, 256), mode='bilinear') 

                    if args.lambda_l2 > 0:
                        loss_l2 = F.mse_loss(imgs, x1_batch)
                        l2_loss_x0 += loss_l2
                    if args.lambda_lpips > 0:
                        loss_lpips = lpips_loss(imgs, x1_batch)
                        lpips_loss_x0 += loss_lpips

            id_loss_x0 = id_loss_x0 / batch_rate if batch_rate != 0 else 0.0
            l2_loss_x0 = l2_loss_x0 / batch_rate if batch_rate != 0 else 0.0
            lpips_loss_x0 = lpips_loss_x0 / batch_rate if batch_rate != 0 else 0.0
            
            loss = loss_dict["loss"].mean() + id_loss_x0 * args.lambda_id + l2_loss_x0 * args.lambda_l2 + lpips_loss_x0 * args.lambda_lpips
            # pred_xstart.retain_grad()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)


            
            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, ID loss: {id_loss_x0}, L2 loss: {l2_loss_x0}, LPIPS loss: {lpips_loss_x0} Train Steps/Sec: {steps_per_sec:.2f}")
                
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--ckpt-path", type=str, default=None)  
    parser.add_argument("--preim3d-ckpt", type=str, default=None)
    parser.add_argument("--eg3d-weight", type=str, default=None)
    parser.add_argument("--camera-param", type=str, default=None)
    parser.add_argument("--camera_param", type=str, default=None)
    parser.add_argument("--lambda-id", type=float, default=0.05)
    parser.add_argument("--lambda-l2", type=float, default=0.05)
    parser.add_argument("--lambda-lpips", type=float, default=0.05)
    parser.add_argument("--gen-batch-size", type=int, default=8)
    args = parser.parse_args()
    main(args)
