import numpy as np
import json
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import StandardScaler
from networkx.readwrite import json_graph
import scipy.sparse as sp
import networkx as nx

def run_dfs(adj, msk, u, ind):
    if msk[u] == -1:
        msk[u] = ind
        for v in adj[u].nonzero()[1]:
            run_dfs(adj, msk, v, ind)

def dfs_split(adj):
    nb_nodes = adj.shape[0]
    ret = np.full(nb_nodes, -1, dtype=np.int32)

    graph_id = 0
    for i in range(nb_nodes):
        if ret[i] == -1:
            run_dfs(adj, ret, i, graph_id)
            graph_id += 1

    return ret

def test(adj, mapping):
    nb_nodes = adj.shape[0]
    for i in range(nb_nodes):
        for j in adj[i].nonzero()[1]:
            if mapping[i] != mapping[j]:
                return False
    return True

def find_split(adj, mapping, ds_label):
    nb_nodes = adj.shape[0]
    dict_splits = {}
    for i in range(nb_nodes):
        for j in adj[i].nonzero()[1]:
            if mapping[i] == 0 or mapping[j] == 0:
                dict_splits[0] = None
            elif mapping[i] == mapping[j]:
                if ds_label[i]['val'] == ds_label[j]['val'] and ds_label[i]['test'] == ds_label[j]['test']:
                    if mapping[i] not in dict_splits:
                        dict_splits[mapping[i]] = 'val' if ds_label[i]['val'] else ('test' if ds_label[i]['test'] else 'train')
                    else:
                        ind_label = 'test' if ds_label[i]['test'] else ('val' if ds_label[i]['val'] else 'train')
                        if dict_splits[mapping[i]] != ind_label:
                            print('Inconsistent labels within a graph. Exiting!')
                            return None
                else:
                    print('Labels of both nodes differ. Exiting!')
                    return None
    return dict_splits

def process_p2p():
    print('Loading G...')
    with open('p2p_dataset/ppi-G.json') as jsonfile:
        g_data = json.load(jsonfile)
    G = json_graph.node_link_graph(g_data)

    adj = nx.adjacency_matrix(G).tocoo()

    print('Loading id_map...')
    with open('p2p_dataset/ppi-id_map.json') as jsonfile:
        id_map = json.load(jsonfile)

    id_map = {int(k): int(v) for k, v in id_map.items()}
    id_map = {k: [v] for k, v in id_map.items()}

    print('Loading features...')
    features_ = np.load('p2p_dataset/ppi-feats.npy')
    scaler = StandardScaler()
    train_ids = [id_map[n][0] for n in G.nodes() if not G.nodes[n].get('val') and not G.nodes[n].get('test')]
    train_feats = features_[train_ids]
    scaler.fit(train_feats)
    features_ = scaler.transform(features_)
    features = torch.tensor(features_, dtype=torch.float)

    print('Loading class_map...')
    with open('p2p_dataset/ppi-class_map.json') as jsonfile:
        class_map = json.load(jsonfile)

    splits = dfs_split(adj)

    list_splits = splits.tolist()
    group_inc = 1
    for i in range(np.max(list_splits) + 1):
        if list_splits.count(i) >= 3:
            splits[np.array(list_splits) == i] = group_inc
            group_inc += 1
        else:
            ind_nodes = np.argwhere(np.array(list_splits) == i).flatten().tolist()
            split = None
            for ind_node in ind_nodes:
                if g_data['nodes'][ind_node]['val']:
                    if split is None or split == 'val':
                        splits[np.array(list_splits) == i] = 21
                        split = 'val'
                    else:
                        raise ValueError(f'New node is VAL but previously was {split}')
                elif g_data['nodes'][ind_node]['test']:
                    if split is None or split == 'test':
                        splits[np.array(list_splits) == i] = 23
                        split = 'test'
                    else:
                        raise ValueError(f'New node is TEST but previously was {split}')
                else:
                    if split is None or split == 'train':
                        splits[np.array(list_splits) == i] = 1
                        split = 'train'
                    else:
                        raise ValueError(f'New node is TRAIN but previously was {split}')

    nodes_per_graph = [list_splits.count(i) for i in range(1, np.max(list_splits) + 1)]

    adj_sub = np.empty((len(nodes_per_graph), np.max(nodes_per_graph), np.max(nodes_per_graph)))
    feat_sub = np.empty((len(nodes_per_graph), np.max(nodes_per_graph), features.shape[1]))
    labels_sub = np.empty((len(nodes_per_graph), np.max(nodes_per_graph), len(class_map[list(class_map.keys())[0]])))

    for i in range(1, np.max(list_splits) + 1):
        indexes = np.where(splits == i)[0]
        subgraph_ = adj[indexes, :][:, indexes]

        if subgraph_.shape[0] < np.max(nodes_per_graph) or subgraph_.shape[1] < np.max(nodes_per_graph):
            subgraph = np.identity(np.max(nodes_per_graph))
            feats = np.zeros([np.max(nodes_per_graph), features.shape[1]])
            labels = np.zeros([np.max(nodes_per_graph), len(class_map[list(class_map.keys())[0]])])
            subgraph[:subgraph_.shape[0], :subgraph_.shape[1]] = subgraph_.todense()
            adj_sub[i - 1] = subgraph
            feats[:len(indexes)] = features[indexes].todense()
            feat_sub[i - 1] = feats
            for j, node in enumerate(indexes):
                labels[j] = np.array(class_map[str(node)])
            labels[len(indexes):np.max(nodes_per_graph)] = 0
            labels_sub[i - 1] = labels
        else:
            adj_sub[i - 1] = subgraph_.todense()
            feat_sub[i - 1] = features[indexes].todense()
            for j, node in enumerate(indexes):
                labels[j] = np.array(class_map[str(node)])
            labels_sub[i - 1] = labels

    dict_splits = find_split(adj, splits, g_data['nodes'])

    print('Are sub-graphs isolated?')
    print(test(adj, splits))

    train_split = []
    val_split = []
    test_split = []
    for key, value in dict_splits.items():
        if value == 'train':
            train_split.append(int(key) - 1)
        elif value == 'val':
            val_split.append(int(key) - 1)
        elif value == 'test':
            test_split.append(int(key) - 1)

    train_adj = torch.tensor(adj_sub[train_split], dtype=torch.float)
    val_adj = torch.tensor(adj_sub[val_split], dtype=torch.float)
    test_adj = torch.tensor(adj_sub[test_split], dtype=torch.float)

    train_feat = torch.tensor(feat_sub[train_split], dtype=torch.float)
    val_feat = torch.tensor(feat_sub[val_split], dtype=torch.float)
    test_feat = torch.tensor(feat_sub[test_split], dtype=torch.float)

    train_labels = torch.tensor(labels_sub[train_split], dtype=torch.float)
    val_labels = torch.tensor(labels_sub[val_split], dtype=torch.float)
    test_labels = torch.tensor(labels_sub[test_split], dtype=torch.float)

    train_nodes = torch.tensor(nodes_per_graph[train_split], dtype=torch.long)
    val_nodes = torch.tensor(nodes_per_graph[val_split], dtype=torch.long)
    test_nodes = torch.tensor(nodes_per_graph[test_split], dtype=torch.long)

    tr_msk = torch.zeros((len(train_nodes), np.max(nodes_per_graph)), dtype=torch.float)
    vl_msk = torch.zeros((len(val_nodes), np.max(nodes_per_graph)), dtype=torch.float)
    ts_msk = torch.zeros((len(test_nodes), np.max(nodes_per_graph)), dtype=torch.float)

    for i in range(len(train_nodes)):
        tr_msk[i, :train_nodes[i]] = 1

    for i in range(len(val_nodes)):
        vl_msk[i, :val_nodes[i]] = 1

    for i in range(len(test_nodes)):
        ts_msk[i, :test_nodes[i]] = 1

    return train_adj, val_adj, test_adj, train_feat, val_feat, test_feat, train_labels, val_labels, test_labels, train_nodes, val_nodes, test_nodes, tr_msk, vl_msk, ts_msk
