
CUDA_VISIBLE_DEVICES=0,1 nohup python -u -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port=47587 fashion_catereg.py >show_catereg.out 2>&1 &
