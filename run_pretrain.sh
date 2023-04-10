CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port=47589 fashion_pretrain.py >show_pretrain.out 2>&1 &

