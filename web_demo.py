import gradio as gr
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
from PIL import Image
from camera_utils import LookAtPoseSampler
from torchvision import transforms

from moviepy.editor import *

def add_text(history, text):
    history = history + [(text, None)]
    return history, ""


def add_file(history, file):
    history = history + [((file.name,), None)]
    return history




def get_model():

    torch.manual_seed(10)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    # latent_size = args.image_size // 8
    latent_size = 4
    in_channels = 512
    model = DiT_models['DiT-B_G/1_pair'](
        input_size=latent_size,
        num_classes=1
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = './pretrained/instructpix2nerf.pt'
    print(f'ckpt_path: {ckpt_path}')
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()  # important!
    diffusion = create_diffusion(f'ddim{str(50)}')
    
    net, _ = setup_model('./pretrained/preim3d.pt', './pretrained/G_ema.pkl', device)
    generator = net.decoder
    generator.eval()

    return net, generator, diffusion, model

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

# print(check_name())
global_states = list(get_model())


def gen_interp_video(G, mp4: str, ws, w_frames=60*4, kind='linear', grid_dims=(1,1), num_keyframes=None, wraps=2, psi=1, truncation_cutoff=14, cfg='FFHQ', image_mode='image', gen_shapes=False, device=torch.device('cuda'), **video_kwargs):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]

    if num_keyframes is None:
        if ws.shape[0] % (grid_w*grid_h) != 0:
            raise ValueError('Number of input seeds must be divisible by grid W*H')
        num_keyframes = ws.shape[0] // (grid_w*grid_h)

    # all_ws = np.zeros(num_keyframes*grid_h*grid_w, dtype=np.int64)
    # for idx in range(num_keyframes*grid_h*grid_w):
    #     all_ws[idx] = ws[idx % ws.shape[0]]

    camera_lookat_point = torch.tensor(G.rendering_kwargs['avg_camera_pivot'], device=device)
    # zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])).to(device)
    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, camera_lookat_point, radius=G.rendering_kwargs['avg_camera_radius'], device=device)
    focal_length = 4.2647 if cfg != 'Shapenet' else 1.7074 # shapenet has higher FOV
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c = c.repeat(ws.shape[0], 1)
    # ws = G.mapping(z=zs, c=c, truncation_psi=psi, truncation_cutoff=truncation_cutoff)
    _ = G.synthesis(ws[:1], c[:1]) # warm up
    ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])
    # Interpolation.
    grid = []
    for yi in range(grid_h):
        row = []
        for xi in range(grid_w):
            x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
            y = np.tile(ws[yi][xi].detach().cpu().numpy(), [wraps * 2 + 1, 1, 1])
            interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
            row.append(interp)
        grid.append(row)

    # Render video.
    max_batch = 10000000
    voxel_resolution = 256
    video_out = imageio.get_writer(mp4, mode='I', fps=6, codec='libx264', **video_kwargs)

    # if gen_shapes:
    #     outdir = 'interpolation_{}_{}/'.format(all_seeds[0], all_seeds[1])
    #     os.makedirs(outdir, exist_ok=True)
    all_poses = []
    for frame_idx in tqdm(range(num_keyframes * w_frames)):
        imgs = []
        for yi in range(grid_h):
            for xi in range(grid_w):
                pitch_range = 0.25
                yaw_range = 0.35
                cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                        3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                        camera_lookat_point, radius=G.rendering_kwargs['avg_camera_radius'], device=device)
                all_poses.append(cam2world_pose.squeeze().cpu().numpy())
                focal_length = 4.2647 if cfg != 'Shapenet' else 1.7074 # shapenet has higher FOV
                intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
                c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                interp = grid[yi][xi]
                w = torch.from_numpy(interp(frame_idx / w_frames)).to(device)

                entangle = 'camera'

                img = G.synthesis(ws=w.unsqueeze(0), c=c, noise_mode='const')[image_mode][0]

                if image_mode == 'image_depth':
                    img = -img
                    img = (img - img.min()) / (img.max() - img.min()) * 2 - 1

                imgs.append(img)

        video_out.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))
    video_out.close()
    all_poses = np.stack(all_poses)

    if gen_shapes:
        print(all_poses.shape)
        with open(mp4.replace('.mp4', '_trajectory.npy'), 'wb') as f:
            np.save(f, all_poses)





def bot(history):
    t_start = int(50*0.3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, generator, diffusion, model = global_states
    transform = transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])



    
    latest_text = None
    latest_text_no = 0
    latest_img = None
    latest_img_no = 0
    for no, msg in enumerate(reversed(history)):
        if isinstance(msg[0], str):
            latest_text = msg[0]
            latest_text_no = no
            break
    for no, msg in enumerate(reversed(history)):
        if isinstance(msg[0], tuple):
            if msg[0][0].endswith('.png') or msg[0][0].endswith('.jpg'):
                latest_img = msg[0][0]
                latest_img_no = no
                break


    if latest_text is not None and latest_img is not None and latest_img_no > latest_text_no:
        image_path = latest_img
        print(image_path)

        input_prompt = latest_text
        
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = transform(img)

        class_labels = [1 for i in range(1)]
        x = img.unsqueeze(0)
        x = x.to(device)
        codes = net.encoder(x)
        x = codes.unsqueeze(1).permute(0,3,1,2) 
        padding = torch.zeros(x.shape[:-1]).unsqueeze(3).repeat(1,1,1,2).to(x.device)
        x = torch.cat((x, padding), dim=3)
        shape = (x.shape[0], x.shape[1], 4, 4)
        x = x.reshape(shape)
        latent_size = 4
        in_channels = 512
        # Create sampling noise:
        n = len(class_labels)
        z = torch.randn(n, in_channels, latent_size, latent_size, device=device)
        y = torch.tensor(class_labels, device=device)

        t = torch.tensor([t_start] * x.shape[0], device=device)

        noise = torch.randn_like(x)
        x_t = diffusion.q_sample(x, t, noise=noise)

        
        uncond = torch.zeros_like(z)
        cond = torch.cat([x, x, uncond], 0)
        # Setup classifier-free guidance:
        z = torch.cat([x_t, x_t, x_t], 0)
        y_null = torch.tensor([1] * n, device=device)
        y = torch.cat([y, y, y_null], 0)

        prompt = [input_prompt for i in range(n)]
        prompt_null = ['' for i in range(2*n)]
        prompt += prompt_null
        model_kwargs = dict(y=y, prompt=prompt, cond=cond, cfg_text_scale=2.0, cfg_img_scale=1.0)

        samples = diffusion.ddim_decode(
            model.forward_with_cfg_text_and_cfg_img, z.shape, z, t_start, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        print('samples: ', samples.shape)
        samples, _, _ = samples.chunk(3, dim=0)  # Remove null class samples



        shape = (samples.shape[0], samples.shape[1], 1, 16)
        latents = samples.reshape(shape)[:,:,:,:14].permute(0,2,3,1).squeeze(1)
        
        print('latents: ', latents.shape)
        print('latents mean: ', latents.abs().mean())

        latents = latents + net.latent_avg.repeat(latents.shape[0], 1, 1)
        print('final latents mean: ', latents.abs().mean())


        yaw_d = math.pi / 9
        pitch_d = math.pi / 36
        yaws = [-yaw_d, yaw_d]
        pitchs = [-pitch_d, pitch_d]
       
        for idx in range(n):
          
            video_save_path = f'{image_path.split(".")[0]}_0001.mp4'
            gen_interp_video(G=generator, mp4=video_save_path, ws=latents[idx:idx+1,:,:], bitrate='10M', grid_dims=(1, 1),
                                num_keyframes=None,
                                w_frames=12, psi=0.7, truncation_cutoff=14,
                                cfg='FFHQ', image_mode='image', gen_shapes=False)

            
        gif_save_path = f'{image_path.split(".")[0]}_0001.gif'
        clip = (VideoFileClip(video_save_path)
        #   .subclip(1,3)
          .resize(1.0))
        clip.write_gif(gif_save_path)


        response = (gif_save_path,)
        # response = (video_save_path,)
        history[-1][1] = response


    # history = history + [((file.name,), None)]
    # response = "**That's cool!**"
    # history[-1][1] = response
    return history

css = """
#warning {background-color: #FFCCCB} 
#chatbot {background-color: #2add9c; text-align: right; width: 60%;} 
.feedback textarea {font-size: 24px !important}

.bot {text-align: left}
.human {text-align: right, width: 60%;}

"""
with gr.Blocks(css=css) as demo:
    chatbot = gr.Chatbot([], label="InstructPixel2NeRF", elem_id="chatbot").style(height=700)

    with gr.Row():
        with gr.Column(scale=0.475):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter, or upload an image"
            ).style(container=False)
        with gr.Column(scale=0.125, min_width=0):
            btn = gr.UploadButton("📁", file_types=["image"])

    txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(bot, chatbot, chatbot)
    btn.upload(add_file, [chatbot, btn], [chatbot]).then(bot, chatbot, chatbot)

demo.launch()