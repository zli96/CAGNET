# sbatch test.slurm
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn import Parameter
from torch_sparse import spmm
import torch.distributed as dist
import os
import os.path as osp
import numpy as np

class COOMatrix():
    # indices should be a 2x? pytorch tensor containing the indices
    def __init__(self, indices, nrows, ncols):
        self.indices = indices
        self.nrows = nrows 
        self.ncols = ncols
        self.values = torch.ones(indices.size()[1])

# Assumes that the COO matrix only has 1s for nonzero elements.
# indices: 2x? torch tensor, should be the indices of a COO format matrix.
# row_ranges: 2x? numpy array specifying how to split the input matrix.
# axis: R or C, specifying whether to split along rows or columns.
# unspec_dim: if axis is R, this is the # of columns in the COO matrix. Otherwise it is the # of rows.
# returns an array of COOMatrix
def splitCOO(indices, ranges, unspec_dim, axis:str):
    COOs = []
    for i in range(0,ranges.shape[1]):
        size = ranges[1,i] - ranges[0,i]
        if(axis.lower() == 'r'):
            offset = ranges[0,i]
            indices_a = indices[0,:] >= ranges[0,i]
            indices_b = indices[0,:] < ranges[1,i]
            temp_indices = indices[:, (indices_a & indices_b)]
            temp_indices[0,:] = temp_indices[0,:] - offset
            temp_coo = COOMatrix(temp_indices, size, unspec_dim) 
        else:
            offset = ranges[0,i]
            indices_a = indices[1,:] >= ranges[0,i]
            indices_b = indices[1,:] < ranges[1,i]
            temp_indices = indices[:, (indices_a & indices_b)]
            temp_indices[1,:] = temp_indices[1,:] - offset
            temp_coo = COOMatrix(temp_indices, unspec_dim, size)
        COOs.append(temp_coo)
    return COOs

# This function computes A*B with 2 assumptions: 1. A is sparse in COO format. 2. both A and B is distributed along the rows and each processor broadcast their local B in turns
# M1_ind: indices of the local A in device
# M2: local B in device
def dist_spmm(M1_ind, M2, proc_row_range, rank, group_size, group, device):
    M2_ncol = M2.size()[1]
    # this is the same as b_M1 nrows and M2 nrows
    M1_nrow = proc_row_range[1,rank] - proc_row_range[0,rank] 
    sum = torch.zeros((M1_nrow,M2_ncol), dtype=torch.float).to(device)
    M1_blocks = splitCOO(M1_ind, proc_row_range, M1_nrow, 'c')
    for i in range(0, group_size):  
        N_nrows = proc_row_range[1,i] - proc_row_range[0,i] 
        if rank == i:
            dist.broadcast(M2, src=i, group=group)
            N = M2
        else:
            N = torch.zeros(N_nrows, M2_ncol, device=device, dtype=torch.float32)
            dist.broadcast(N, src=i, group=group)
        M1_blocks[i].values = M1_blocks[i].values.to(device)
        temp = spmm(M1_blocks[i].indices, M1_blocks[i].values, M1_nrow, N_nrows, N)
        sum = torch.add(temp, sum) 
    return sum

class GCNFunc(torch.autograd.Function):
    @staticmethod
    # def forward(ctx, inputs, weight, adj_matrix, func):
    def forward(ctx, inputs, weight, adj_inx, proc_row_range, func, rank, group_size, group, device):
        global run
        # inputs: H
        # adj_matrix: A
        # weight: W
        # func: sigma

        ctx.save_for_backward(inputs, weight, adj_inx)
        ctx.func = func
        ctx.device = device
        ctx.group_size = group_size
        ctx.group = group
        ctx.rank = rank
        ctx.proc_row_range = proc_row_range

        # z = torch.sparse.mm(adj_matrix, inputs)
        # z = spmm(adj_inx, adj_value, adj_nrows, adj_nrows, inputs)
        z = dist_spmm(adj_inx, inputs, proc_row_range, rank, group_size, group, device)
        z = torch.mm(z, weight)

        z.requires_grad = True
        ctx.z = z

        # if activations:
        if func is F.log_softmax:
            h = func(z, dim=1)
        elif func is F.relu:
            h = func(z)
        else:
            h = z

        return h

    @staticmethod
    def backward(ctx, grad_output):
        global run

        inputs, weight, adj_idx = ctx.saved_tensors
        func = ctx.func
        z = ctx.z
        device = ctx.device         
        group_size = ctx.group_size     
        group = ctx.group          
        rank = ctx.rank           
        proc_row_range = ctx.proc_row_range 

        with torch.set_grad_enabled(True):
            if func is F.log_softmax:
                func_eval = func(z, dim=1)
            elif func is F.relu:
                func_eval = func(z)
            else:
                func_eval = z

            sigmap = torch.autograd.grad(outputs=func_eval, inputs=z, grad_outputs=grad_output)[0]
            grad_output = sigmap

        # First backprop 
        # ag = torch.sparse.mm(adj_matrix, grad_output)
        # ag = spmm(adj_idx, adj_value, adj_nrows, adj_nrows, grad_output)
        ag = dist_spmm(adj_idx, grad_output, proc_row_range, rank, group_size, group, device)

        grad_input = torch.mm(ag, weight.t())

        # Second backprop equation (reuses the A * G^l computation)
        grad_weight = torch.mm(inputs.t(), ag)
        dist.all_reduce(grad_weight, op=dist.reduce_op.SUM, group=group)

        return grad_input, grad_weight, None, None, None, None, None, None, None # not sure why I had to add one more gradient here

def train(inputs, weight1, weight2, adj_indx, proc_row_range, optimizer, data, rank, group_size, group, device):
    outputs = GCNFunc.apply(inputs, weight1, adj_indx, proc_row_range, F.relu, rank, group_size, group, device)
    outputs = GCNFunc.apply(outputs, weight2, adj_indx, proc_row_range, F.log_softmax, rank, group_size, group, device)

    optimizer.zero_grad()
    rank_train_mask = torch.split(data.train_mask, outputs.size(0), dim=0)[rank]
    datay_rank = torch.split(data.y, outputs.size(0), dim=0)[rank]

    # Note: bool type removes warnings, unsure of perf penalty
    # loss = F.nll_loss(outputs[data.train_mask.bool()], data.y[data.train_mask.bool()])
    if list(datay_rank[rank_train_mask].size())[0] > 0:
        loss = F.nll_loss(outputs[rank_train_mask], datay_rank[rank_train_mask])
        print('loss: {:.4f}'.format(loss))
        loss.backward()
    else:
        fake_loss = (outputs * torch.cuda.FloatTensor(outputs.size(), device=device).fill_(0)).sum()
        # fake_loss = (outputs * torch.zeros(outputs.size())).sum()
        fake_loss.backward()

    optimizer.step()

    return outputs

def test(outputs, data):
    logits, accs = outputs, []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)

    if len(accs) == 1:
        accs.append(0)
        accs.append(0)
    return accs

########################
## set up environment ##
########################
if __name__ == '__main__':
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

    ###########################
    ## inside each processor ##
    ###########################
    rank = dist.get_rank()
    size = dist.get_world_size()
    group = dist.new_group(list(range(size)))
    device = torch.device('cuda:{}'.format(rank))

    dataset = 'Cora'
    dataset = Planetoid('../data', dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
    inputs = data

    num_classes=dataset.num_classes
    num_features = dataset.num_features # f
    num_nodes = data.x.size()[0] # n

    #Set up to get my block row of the adjacency/feature matrix
    nrows_per_proc = np.ones(size)
    proc_row_range = np.ones((2,size),dtype = int)
    rnrows = np.floor(num_nodes/size)
    for i in range(0, size):
        if num_nodes % size == 0:
            nrows_per_proc[i] = rnrows
        else:
            if i < num_nodes % size:
                nrows_per_proc[i] = rnrows + 1
            else:
                nrows_per_proc[i] = rnrows
        # row range lower bound (inclusive)
        proc_row_range[0,i] = int(np.sum(nrows_per_proc[0:i]))
        # row range upper bound (exclusive)
        proc_row_range[1,i] = int(np.sum(nrows_per_proc[0:i+1]))

    # Get the local block row of the adjacency matrix
    local_adj = splitCOO(data.edge_index, proc_row_range, num_nodes, 'r')[rank]
    local_adj.indices = local_adj.indices.to(device)

    # Now get the local block row of the feature matrix
    local_feature = data.x[proc_row_range[0,rank]:proc_row_range[1,rank], :]
    local_feature = local_feature.to(device)
    # dist_gemm(local_adj.indices, local_feature, device, proc_row_range, rank, size, group)

    # Generate random weight matrices
    mid_layer = 16
    torch.manual_seed(0)
    weight1_nonleaf = torch.rand(num_features, mid_layer, requires_grad=True)
    weight1_nonleaf = weight1_nonleaf.to(device)
    weight1_nonleaf.retain_grad()
    weight2_nonleaf = torch.rand(mid_layer, num_classes, requires_grad=True)
    weight2_nonleaf = weight2_nonleaf.to(device)
    weight2_nonleaf.retain_grad()
    weight1 = Parameter(weight1_nonleaf)
    weight2 = Parameter(weight2_nonleaf)

    epochs=100
    optimizer = torch.optim.Adam([weight1, weight2], lr=0.01)
    data = data.to(device)
    for epoch in range(1, epochs):
        output = train(local_feature, weight1, weight2, local_adj.indices, proc_row_range, optimizer, data, rank, size, group, device)
        outputs = []
        for i in range(0, size):
            outputs.append(torch.zeros(int(nrows_per_proc[i]), int(num_classes)).to(device))
        dist.all_gather_multigpu(outputs, output)
        outputs = torch.cat(outputs, dim=0)
        train_acc, val_acc, test_acc = test(outputs, data)
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, val_acc, test_acc))
    
