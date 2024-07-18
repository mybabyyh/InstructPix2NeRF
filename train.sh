CUDA_VISIBLE_DEVICES=0,1,2  torchrun --nnodes=1 --nproc_per_node=3  --master_port=25641 train.py --model DiT-B_G/1_pair --num-classes 1 \
--data-path /pathto/pair_edit_e4e_ip2p_no_inversion  \
--preim3d-ckpt ./checkpoints/preim3d_style_ip2p_105000.pt  \
--eg3d-weight ./checkpoints/G_ema.pkl  \
--camera_param /pathto/pair_edit_e4e_big_style_ip2p/dataset.json   \
--ckpt-every 30000 --global-batch-size  6  --epochs 100000  --gen-batch-size 2  \
--lambda-id 0.1  --lambda-l2 0.0  --lambda-lpips 0.00
