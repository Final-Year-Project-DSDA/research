import torch
import torch_geometric
from torch_geometric.utils import from_scipy_sparse_matrix, dense_to_sparse
import numpy as np
import scipy.sparse as sp
import networkx as nx
import pickle as pkl

def adj_to_bias_pyg(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)

def load_data_pyg(dataset_str):
    """Load data and convert to PyTorch Geometric format."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for name in names:
        with open(f"data/ind.{dataset_str}.{name}", 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))
    
    x, y, tx, ty, allx, ally, graph = objects
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    
    # Convert adjacency matrix to edge_index format for PyTorch Geometric
    edge_index, edge_attr = from_scipy_sparse_matrix(adj)
    
    # Stack features
    features = sp.vstack((allx, tx)).tolil()
    
    # Convert labels and masks to PyTorch tensors
    labels = np.vstack((ally, ty))
    labels = torch.tensor(labels, dtype=torch.long)
    
    return edge_index, features, labels

def preprocess_adj_pyg(adj):
    """Preprocess adjacency matrix for GCN and return PyTorch Geometric compatible format."""
    adj_normalized = normalize_adj_pyg(adj + sp.eye(adj.shape[0]))
    return from_scipy_sparse_matrix(adj_normalized)

def normalize_adj_pyg(adj):
    """Normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
