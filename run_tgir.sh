
CUDA_VISIBLE_DEVICES=0,1 nohup python -u -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port=45999 fashion_tgir.py >show_tgir.out 2>&1 &
