import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch.nn import Parameter
from torch_sparse import spmm
import numpy as np

class GCNFunc(torch.autograd.Function):
    @staticmethod
    # def forward(ctx, inputs, weight, adj_matrix, func):
    def forward(ctx, inputs, weight, adj_inx, adj_value, adj_nrows, func):
        global run
        # inputs: H
        # adj_matrix: A
        # weight: W
        # func: sigma

        ctx.save_for_backward(inputs, weight, adj_inx, adj_value)
        ctx.func = func

        # z = torch.sparse.mm(adj_matrix, inputs)
        z = spmm(adj_inx, adj_value, adj_nrows, adj_nrows, inputs)
        z = torch.mm(z, weight)

        z.requires_grad = True
        ctx.z = z
        ctx.adj_nrows = adj_nrows

        # if activations:
        if func is F.log_softmax:
            h = func(z, dim=1)
        elif func is F.relu:
            h = func(z)
        else:
            h = z

        return h
        # else:
        #     return z

    @staticmethod
    def backward(ctx, grad_output):
        global run

        inputs, weight, adj_idx, adj_value = ctx.saved_tensors
        func = ctx.func
        z = ctx.z
        adj_nrows = ctx.adj_nrows

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
        ag = spmm(adj_idx, adj_value, adj_nrows, adj_nrows, grad_output)

        grad_input = torch.mm(ag, weight.t())

        # Second backprop equation (reuses the A * G^l computation)
        grad_weight = torch.mm(inputs.t(), ag)

        return grad_input, grad_weight, None, None, None, None, None, None

def train(inputs, weight1, weight2, adj_indx, adj_value, adj_nrows, optimizer, data):
    outputs = GCNFunc.apply(inputs, weight1, adj_indx, adj_value, adj_nrows, F.relu)
    outputs = GCNFunc.apply(outputs, weight2, adj_indx, adj_value, adj_nrows, F.log_softmax)

    optimizer.zero_grad()

    # Note: bool type removes warnings, unsure of perf penalty
    loss = F.nll_loss(outputs[data.train_mask.bool()], data.y[data.train_mask.bool()])
    print('loss: {:.4f}'.format(loss))
    loss.backward()

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

dataset = 'Cora'
dataset = Planetoid('../data', dataset, transform=T.NormalizeFeatures())
data = dataset[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inputs = data
num_features = dataset.num_features
mid_layer = 16
epochs=100
num_classes=dataset.num_classes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)
weight1_nonleaf = torch.rand(num_features, mid_layer, requires_grad=True)
weight1_nonleaf = weight1_nonleaf.to(device)
weight1_nonleaf.retain_grad()

weight2_nonleaf = torch.rand(mid_layer, num_classes, requires_grad=True)
weight2_nonleaf = weight2_nonleaf.to(device)
weight2_nonleaf.retain_grad()

weight1 = Parameter(weight1_nonleaf)
weight2 = Parameter(weight2_nonleaf)

optimizer = torch.optim.Adam([weight1, weight2], lr=0.01)
temp_value = np.ones((data.edge_index.shape[1],),dtype='float32')
# idx_array=data.edge_index.data.numpy()
num_nodes = data.x.size()[0]
# adj_matrix = torch.sparse_coo_tensor((idx_array[0,:],idx_array[1,:]), temp_value, ( num_nodes, num_nodes))
inputs.x  = inputs.x.to(device)
# adj_matrix = adj_matrix.to(device)
data = data.to(device)
for epoch in range(1, epochs):
  outputs = train(inputs.x, weight1, weight2, data.edge_index, torch.tensor(temp_value).to(device), num_nodes, optimizer, data)
  train_acc, val_acc, test_acc = test(outputs, data)
  log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
  print(log.format(epoch, train_acc, val_acc, test_acc))