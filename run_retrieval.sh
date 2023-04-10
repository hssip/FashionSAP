
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port=45577 fashion_retrieval.py >show_retrieval.out 2>&1 &
