from torch import zeros, from_numpy
from tqdm import tqdm
import scipy

def adj_degree_dinv(edge_index, num_nodes):
    adj = zeros((num_nodes,num_nodes))
    degree = zeros((num_nodes,num_nodes))
    for i in tqdm(range(edge_index.shape[1])):
        first = edge_index[0][i]
        second = edge_index[1][i]
        adj[first][second] = 1
        adj[second][first] = 1
    for i in tqdm(range(num_nodes)):
        degree[i][i] = sum(adj[i][:])
    inter = scipy.linalg.fractional_matrix_power(degree, (-1/2))
    d_inv = from_numpy(inter)
    return (adj, degree, d_inv)
