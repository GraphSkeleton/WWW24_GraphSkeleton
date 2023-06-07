#!/usr/bin/env python
# coding: utf-8

# In[]:


import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch_cluster import random_walk
from torch_geometric.data import Data

from torch_geometric.typing import List, Optional, Tuple, Union
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.loader.neighbor_sampler import EdgeIndex, Adj

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborSampler

import numpy as np
import time

# compression grpah

# In[]:

file_path = './DGraphFin/dgraphfin.npz'
data = np.load(file_path)

# data = DGraphFin(root='./DGraphFin/', name='DGraphFin', transform=T.ToSparseTensor())
                                    
x = data['x']
y = data['y']
edge_index = data['edge_index'].T
print(edge_index.shape)

train_mask = data['train_mask']
valid_mask = data['valid_mask']
test_mask = data['test_mask']
star = np.concatenate((train_mask, valid_mask, test_mask), axis=0)


# In[]:
import graph_skeleton
from gs import *
import graph0429

print('*'*10, 'skeleton-alpha', '*'*10)
d = [2,1]


print(f"Zip nodes (num_edge: {edge_index.shape[1]})... ", end=" ", flush=True)
start_time = time.perf_counter()
n = int(np.max(edge_index) + 1)        # Num of vertex
edge_index = edge_index.astype(np.int32)

tmp = np.full((n,), False, dtype=np.bool_)
tmp[star] = True
target_node = tmp

graph0429.greet()
g = graph0429.Graph(edge_index, target_node)
n_id = g.zip(d[1], d[0], 2)
n_edge_index = g.reconstruct_edge(n_id);
# print(f"n_id: {n_id}")
# print(f"n_edge_index: {n_edge_index}")
print(f"Done! [{time.perf_counter() - start_time:.2f}s]")

n_x, cnt_x = reconstruct_x(x, n_id)
print(f"y: {y}")
n_y = mapping_label(y, n_id)
n_train_mask = mapping_mask(train_mask, n_id)
n_valid_mask = mapping_mask(valid_mask, n_id)
n_test_mask = mapping_mask(test_mask, n_id)

print('new node number:', n_x.shape[0])

zip_data = { 'x': n_x, 'y': n_y, 'edge_index': n_edge_index, 'train_mask': n_train_mask, 'valid_mask': n_valid_mask, 'test_mask': n_test_mask}
np.save(f'xinye_alpha.npy', zip_data)



# In[]:

print('*'*10, 'skeleton-beta', '*'*10)
d = [2,1]

print(f"Build... ", end=" ", flush=True)
start_time = time.perf_counter()
graph_skeleton.init()
edge_index32 = edge_index.astype(np.int32)
star32 = star.astype(np.int32)
num_node = np.max(edge_index) + 1
tmp = np.full((num_node,), False, dtype=np.bool_)
tmp[star] = True
star32 = tmp
print(f"edge_index: {edge_index32.shape}")
print(f"star: {star32.shape}")
g = graph_skeleton.Graph(edge_index32, star32)
print(f"Done! [{time.perf_counter() - start_time:.2f}s]")

print(f"Zip... ", end=" ", flush=True)
start_time = time.perf_counter()
n_id = g.extract_skeleton(d[1], d[0], 15, True, 16)
print(f"Done! [{time.perf_counter() - start_time:.2f}s]")

print(f"Reconstruct edge... ", end=" ", flush=True)
start_time = time.perf_counter()
# n_edge_index = g.reconstruct_edge(n_id)
n_edge_index, n_edge_weight = g.reconstruct_reweighted_edge(n_id)
print(f"Done! [{time.perf_counter() - start_time:.2f}s]")
print(f"Graph_2: #V: {np.max(n_id)+1}, #E: {n_edge_index.shape[1]}")

n_x, cnt_x = reconstruct_x(x, n_id)
n_y = mapping_label(y, n_id)

n_train_mask = mapping_mask(train_mask, n_id)
n_valid_mask = mapping_mask(valid_mask, n_id)
n_test_mask = mapping_mask(test_mask, n_id)
n_star = mapping_mask(star, n_id)

print('new node number:', n_x.shape[0])

zip_data = { 'x': n_x, 'y': n_y, 
            'edge_index': n_edge_index, 'edge_weight': n_edge_weight, 
            'train_mask': n_train_mask, 'valid_mask': n_valid_mask, 'test_mask': n_test_mask}
# np.save(f'xinye_beta.npy', zip_data)




# In[ ]:
# allfliation merge

print('*'*10, 'skeleton-gamma', '*'*10)

num_node2 = np.max(n_edge_index) + 1
n_star32 = np.full((num_node2,), False, dtype=np.bool_)
n_star32[n_star] = True


graph_skeleton.init()
g2 = graph_skeleton.Graph(n_edge_index, n_star32)

corr_mask = g2.get_corr_mask(1, 2)
# print(corr_mask)

nt = g2.nearest_target()
# print(nt)

g2.drop_corr()
n_id2 = g2.extract_skeleton(1, 2, 2, True, 1)

alpha = 0.7 # corr节点feature的占比
x_corr = np.zeros_like(n_x)
num_corr = np.zeros(x_corr.shape[0])
x2 = np.zeros_like(n_x)
for i in range(num_node2):
    if corr_mask[i]:
        x_corr[ nt[i] ] += n_x[i]
        num_corr[ nt[i] ] += 1

num_corr = np.clip(
    num_corr,
    a_min = 1,
    a_max = None,
    )

for i in range(num_node2):
    if n_star32[i]:
        x2[i] = (alpha * n_x[i] + (1-alpha) * x_corr[i]/num_corr[i])

n_x2, cnt_x2 = reconstruct_x(x2, n_id2)

n_edge_index2, n_edge_weight2 = g2.reconstruct_reweighted_edge(n_id2)
print(f"Done! [{time.perf_counter() - start_time:.2f}s]")
print(f"Graph_2: #V: {np.max(n_id2)+1}, #E: {n_edge_index2.shape[1]}")

n_y2 = mapping_label(n_y, n_id2)
n_train_mask2 = mapping_mask(n_train_mask, n_id2)
n_valid_mask2 = mapping_mask(n_valid_mask, n_id2)
n_test_mask2 = mapping_mask(n_test_mask, n_id2)
n_star2 = mapping_mask(n_star, n_id2)

print('new node number:', n_x2.shape[0])

zip_data = { 'x': n_x2, 'y': n_y2, 
            'edge_index': n_edge_index2, 
            'edge_weight': n_edge_weight2, 
            'train_mask': n_train_mask2, 'valid_mask': n_valid_mask2, 'test_mask': n_test_mask2}
np.save(f'xinye_gamma.npy', zip_data)
