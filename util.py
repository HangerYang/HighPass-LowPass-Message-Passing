from torch import zeros, from_numpy, tensor
from tqdm import tqdm
import numpy as np
import scipy

def lap_dinv(edge_index, num_nodes):
    adj = zeros((num_nodes,num_nodes))
    degree = zeros((num_nodes,num_nodes))
    for i in tqdm(range(edge_index.shape[1])):
        first = edge_index[0][i]
        second = edge_index[1][i]
        adj[first][second] = 1
        adj[second][first] = 1
    for i in tqdm(range(num_nodes)):
        degree[i][i] = sum(adj[i][:])
    lap = degree - adj
    inter = scipy.linalg.fractional_matrix_power(degree, (-1/2))
    d_inv = from_numpy(inter)
    return (lap, d_inv)

def data_sample(num_nodes):
  sample_num = int(num_nodes*0.6)
  sample_lst = np.random.choice(num_nodes,sample_num, replace = False)
  train_mask = [False] * num_nodes
  test_mask = [True] * num_nodes
  for i in sample_lst:
    train_mask[i] = True
    test_mask[i] = False
  train_mask = tensor(train_mask)
  test_mask = tensor(test_mask)
  return train_mask, test_mask