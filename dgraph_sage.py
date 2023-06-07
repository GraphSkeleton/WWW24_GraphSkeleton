# With the preprocessed edge_index

import os.path as osp
from torch_geometric.utils import to_undirected
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch_cluster import random_walk
from tqdm import tqdm
import time
from torch_geometric.data import Data

from torch_geometric.typing import List, Optional, Tuple, Union
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.loader.neighbor_sampler import EdgeIndex, Adj
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv
import numpy as np
from sklearn import metrics
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
import logging
from models import SAGE_NeighSampler

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    # -----------------------------------------------general settings--------------------------------------------------
    parser.add_argument('--sizes', default=[-1,-1,-1], nargs='+', type=int,
                        help='sampled node number for each layer')
    parser.add_argument('--device', default='cuda',
                        help='device for computing')
    parser.add_argument('--cut', default='no', choices = ['no', 'zip'],
                        help='cut target or random')
    parser.add_argument('--sel_ratio', type=float, default=0.5)
    parser.add_argument('--batch-size', default=10, type=int,
                        help='batch size')
    parser.add_argument('--batchnorm', action='store_true',
                        help='if use batch normalization')  
    parser.add_argument('--rd_ratio', default=0.1, type=float,
                        help='random cut ratio')
    parser.add_argument('--k', default=2, type=int,
                        help='hop number for skeleton')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='learning rate')
    parser.add_argument('--epoch', default=200, type=int,
                        help='epoch for train')
    parser.add_argument('--num-layers', default=3, type=int,
                        help='layer number of sage')   
    parser.add_argument('--hidden-channel', default=256, type=int,
                        help='hiden channel of gcnconv layer')   
    parser.add_argument('--mlp', action='store_true',
                        help='if use mlp for classification')   
    parser.add_argument('--sample-mask', action='store_true',
                        help='sample using mask?') 
    parser.add_argument('--iter', default=10, type=int, 
                        help='iteration for running')  
    args = parser.parse_args()
    return args



def train(epoch):
    model.train()

    # pbar = tqdm(total=train_idx.size(0))
    # pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        n_id = n_id.to(device)

        optimizer.zero_grad()
        out = model(x[n_id], adjs)

        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        # pbar.update(batch_size)

    # pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / train_idx.size(0)

    return loss, approx_acc



@torch.no_grad()
def test(masks):
    model.eval()

    out = model.inference(x)

    aucs = []
    for mask in masks:
        test_auc = metrics.roc_auc_score(data.y[mask].cpu().numpy(), out[mask,1].detach().cpu().numpy())
        aucs.append(test_auc)
    return aucs



EPS = 1e-15
args = parse_args()
print(args)

if args.device == 'cpu':
    print('cpu')
    device = torch.device('cpu')
else:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    print('Using GPU:' + str(np.argmax(memory_gpu)))
    device = torch.device('cuda:'+str(np.argmax(memory_gpu)))


if args.cut =='no':
    print('-------origin-------')
    file_path = f'./DGraphFin/dgraphfin.npz'
    dataset = np.load(file_path)
    train_mask = torch.from_numpy(dataset['train_mask'])
    valid_mask = torch.from_numpy(dataset['valid_mask'])
    test_mask = torch.from_numpy(dataset['test_mask'])
    train_idx = train_mask
    x = torch.from_numpy(dataset['x'])
    x = (x-x.mean(0))/x.std(0)
    y = torch.from_numpy(dataset['y']).to(device)
    # edge_index = torch.from_numpy(dataset['edge_index'].T)
    edge_index = torch.from_numpy(dataset['edge_index'].T)

elif args.cut == 'zip':
    print('-------skeleton-------')
    file_path = 'xinye_gamma.npy'
    zip = np.load(file_path, allow_pickle=True).item()
    train_mask = torch.from_numpy(zip['train_mask'])
    valid_mask = torch.from_numpy(zip['valid_mask'])
    test_mask = torch.from_numpy(zip['test_mask'])
    train_idx = train_mask

    x = torch.from_numpy(zip['x'])
    x = (x-x.mean(0))/x.std(0)
    y = torch.from_numpy(zip['y']).to(device)
    edge_index = torch.from_numpy(zip['edge_index']).type(torch.int64)

edge_index = to_undirected(edge_index)
print('#N{}, #E: {}'.format(x.shape, edge_index.shape))

edge_attr = None
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


train_loader = NeighborSampler(edge_index, node_idx=train_mask,
                               sizes=args.sizes, batch_size=args.batch_size,
                               shuffle=True, num_workers=12)
subgraph_loader = NeighborSampler(edge_index, node_idx=None, sizes=[-1],
                                  batch_size=args.batch_size, shuffle=False,
                                  num_workers=12)


# model = SAGE(args, device, subgraph_loader, x.shape[1], args.hidden_channel, out_channels=2, num_layers=args.num_layers, batchnorm=args.batchnorm)
model = SAGE_NeighSampler(device, subgraph_loader, x.shape[1], args.hidden_channel, out_channels=2, num_layers=args.num_layers, dropout=0.0, batchnorm=args.batchnorm)

print(model)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-7)
# x, edge_index = data.x.to(device), data.edge_index.to(device)
x, edge_index = x.to(torch.float32).to(device), edge_index.to(torch.float32).to(device)


best_val_acc = best_test_acc = 0
test_accs = []
Time = []
for run in range(args.iter):
    print('')
    print(f'Run {run:02d}:')
    print('')

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start = time.perf_counter()
    best_val_acc = final_test_acc = 0
    for epoch in range(args.epoch):
        loss, acc = train(epoch)
        
        if epoch % 5 == 0:
            print(f'Train Epoch {epoch:02d}, Loss: {loss:.4f}')
            val_acc, test_acc = test([valid_mask, test_mask])
            print(f'Evaluation Epoch:{epoch:02d}: Val: {val_acc:.4f}, '
                  f'Test: {test_acc:.4f}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
    test_accs.append(final_test_acc)
    end = time.perf_counter()
    Time.append(end-start)
    print('-----------------------')
    print('Consuming time{}:'.format(end-start))
    print('best acc{}:'.format(final_test_acc))

test_acc = torch.tensor(test_accs)
aver_time = torch.tensor(Time)
print('============================')
print(f'Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}')
print(f'aver time: {aver_time.mean():.4f} ± {aver_time.std():.4f}')
print(test_accs)
