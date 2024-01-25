
CUDA_VISIBLE_DEVICES=4 python inference.py --model DiT-B_G/1_pair \
--ckpt ./pretrained/instructpix2nerf.pt  \
--preim3d-ckpt ./pretrained/preim3d.pt  \
--eg3d-weight ./pretrained/G_ema.pkl  \
--image-dir ./data/test  \
--prompt ./data/prompt/prompt.json  \
--save-dir ./outputs/instructpix2nerf/test  \
--num-sampling-steps 50  --strength 0.3  --sample_num 300 
