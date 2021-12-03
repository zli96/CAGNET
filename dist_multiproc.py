# srun -p gpu --pty -N 1 -n X --mem=32GB --gres=gpu:X --time=h:mm:ss /bin/bash
# source /deac/csc/ballardGrp/software/CAGNET/bin/activate
# module load compilers/gcc/6.5.0
# python -m torch.distributed.launch --nproc_per_node=X --nnodes=1 dist_multiproc.py
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import os.path as osp

dist.init_process_group(backend='nccl')
rank = dist.get_rank()
size = dist.get_world_size()
device = torch.device('cuda:{}'.format(rank))
print(device)
print("rank: "+ str(rank) + " world_size: " + str(size))
torch.cuda.set_device(device)
