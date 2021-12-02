# sbatch test.slurm
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import os.path as osp
import numpy as np

# mp.set_start_method('spawn', force=True)
if "SLURM_PROCID" in os.environ.keys():
    os.environ["RANK"] = os.environ["SLURM_PROCID"]
    rank = os.environ["SLURM_PROCID"]

if "SLURM_NTASKS" in os.environ.keys():
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    world_size = os.environ["SLURM_NTASKS"]

if "MASTER_ADDR" not in os.environ.keys():
    os.environ["MASTER_ADDR"] = "127.0.0.1"

os.environ["MASTER_PORT"] = "12394"
dist.init_process_group(backend='nccl')

rank = dist.get_rank()
size = dist.get_world_size()
device = torch.device('cuda:{}'.format(rank))
t = torch.cuda.get_device_properties(device).total_memory
print(t)
print("rank: "+ str(rank) + " world_size: " + str(size))

g_cuda = torch.Generator()
g_cuda.manual_seed(rank)
m = torch.randint(10, (2,2), generator=g_cuda)
m = m.to(device, dtype=torch.float32)
print(m)

group = dist.new_group(list(range(size)))
sum = torch.zeros(2,2, dtype=torch.float32).to(device)
for i in list(range(size)):
    
    if rank == i:
        dist.broadcast(m, src=i, group=group)
        n = m
    else:
        n = torch.zeros(2,2, device=device, dtype=torch.float32)
        dist.broadcast(n, src=i, group=group)
    sum = torch.add(torch.mm(m,n), sum) 
print(sum)




# dataset = 'Cora'
# dataset = Planetoid('../data', dataset, transform=T.NormalizeFeatures())
# data = dataset[0]
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# data.to(device)

# group = dist.new_group(list(range(size)))

# a = data.edge_index[0,:] < 10
# data.edge_index[:,a.nonzero().squeeze()].size()