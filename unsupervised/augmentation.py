import numpy as np
import torch
import copy
from torch_geometric.utils import subgraph
from utils import norm

def permute_edge(data, u, v):
    data = copy.deepcopy(data.cpu())
    node_num, _ = data.x.size()
    edge_index = data.edge_index.numpy()
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1

    aug_adj = copy.deepcopy(adj)

    if adj[u, v] == 0:
        return data
    
    aug_adj[u, v] = 0
    aug_adj[v, u] = 0
    edge_index = aug_adj.nonzero().t()
    
    data.dis = torch.tensor(2.0)
    data.edge_index = edge_index

    return data

def drop_node(data, idx):
    data = copy.deepcopy(data.cpu())
    node_num, _ = data.x.size()
    #_, edge_num = data.edge_index.size()
    drop_num = 1

    idx_drop = idx
    #idx_nondrop = [n for n in range(node_num) if  n != idx]
    #idx_dict = {idx_nondrop[n]: n for n in list(range(node_num - drop_num))}

    edge_index = data.edge_index.numpy()
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1

    aug_adj = copy.deepcopy(adj)

    aug_adj[idx_drop, :] = 0
    aug_adj[:, idx_drop] = 0
    edge_index = aug_adj.nonzero().t()
    
    #dis = norm(adj - aug_adj)
    
    data.dis = data.degree[idx_drop] + 1

    data.edge_index = edge_index

    return data


def mask_nodes(data):

    node_num, feat_dim = data.x.size()
    mask_num = int(node_num / 10)

    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(mask_num, feat_dim)), dtype=torch.float32)

    return data


def drop_nodes(data, rho):
    data = copy.deepcopy(data.cpu())
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num*(1.0-rho))

    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    idx_dict = {idx_nondrop[n]: n for n in list(range(node_num - drop_num))}

    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index

    return data


def permute_edges_prob(data, edge_score, rho, L):
    data = copy.deepcopy(data.cpu())
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num*(1.0-rho))

    edge_prob = edge_score.float()
    edge_prob += 0.001
    edge_prob = np.array(edge_prob.detach())
    edge_prob /= edge_prob.sum()

    idx_nondrop = np.random.choice(edge_num, edge_num-permute_num, replace=False, p=edge_prob)

    edge_index = data.edge_index.transpose(0, 1).numpy()
    edge_index = edge_index[idx_nondrop]
    data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    data.edge_weight = L[idx_nondrop]

    node_num = data.edge_index.max() + 1
    sl = torch.tensor([[n, n] for n in range(node_num)]).t()
    data.edge_index = torch.cat((data.edge_index, sl), dim=1)

    wl = torch.tensor([1.0 for n in range(node_num)], dtype=torch.half)
    data.edge_weight = torch.cat((data.edge_weight, wl))

    return data

def permute_edges_cp(data, edge_score, rho, L):
    data = copy.deepcopy(data.cpu())
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num*(1.0-rho))

    
    edge_prob = max(edge_score.float()) - edge_score.float()
    edge_prob += 0.001
    edge_prob = np.array(edge_prob.detach())
    edge_prob /= edge_prob.sum()

    idx_nondrop = np.random.choice(edge_num, edge_num-permute_num, replace=False, p=edge_prob)

    edge_index = data.edge_index.transpose(0, 1).numpy()
    edge_index = edge_index[idx_nondrop]
    data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    data.edge_weight= L[idx_nondrop]
    
    node_num = data.edge_index.max() + 1
    sl = torch.tensor([[n, n] for n in range(node_num)]).t()
    data.edge_index = torch.cat((data.edge_index, sl), dim=1)

    wl = torch.tensor([1.0 for n in range(node_num)], dtype=torch.half)
    data.edge_weight = torch.cat((data.edge_weight, wl))

    return data

def drop_nodes_prob(data, node_score, rho):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num*(1.0-rho))

    node_prob = node_score.float()
    node_prob += 0.001
    node_prob = np.array(node_prob)
    node_prob /= node_prob.sum()

    idx_nondrop = np.random.choice(node_num, node_num - drop_num, replace=False, p=node_prob)
    idx_drop = np.setdiff1d(np.arange(node_num), idx_nondrop)
    idx_nondrop.sort()

    idx_dict = {idx_nondrop[n]: n for n in list(range(node_num - drop_num))}

    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index

    return data


def drop_nodes_cp(data, node_score, rho):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num*(1.0-rho))

    node_prob = node_prob = max(node_score.float()) - node_score.float()
    node_prob += 0.001
    node_prob = np.array(node_prob)
    node_prob /= node_prob.sum()

    idx_nondrop = np.random.choice(node_num, node_num - drop_num, replace=False, p=node_prob)
    idx_drop = np.setdiff1d(np.arange(node_num), idx_nondrop)
    idx_nondrop.sort()

    idx_dict = {idx_nondrop[n]: n for n in list(range(node_num - drop_num))}

    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index

    return data