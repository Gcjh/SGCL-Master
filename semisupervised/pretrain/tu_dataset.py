from torch_geometric.datasets import TUDataset
import torch
from itertools import repeat, product
import numpy as np
from copy import deepcopy
from torch_geometric.utils import degree



class TUDatasetExt(TUDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <http://graphkernels.cs.tu-dortmund.de>`_.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name <http://graphkernels.cs.tu-dortmund.de>`_ of
            the dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node features (if present).
            (default: :obj:`False`)
    """

    url = 'https://ls11-www.cs.tu-dortmund.de/people/morris/' \
          'graphkerneldatasets'

    def __init__(self,
                 root,
                 name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 use_node_attr=False,
                 processed_filename='data.pt',
                 aug="none", aug_ratio=0.2):
        self.processed_filename = processed_filename
        self.aug = aug
        self.aug_ratio = aug_ratio
        super(TUDatasetExt, self).__init__(root, name, transform, pre_transform,
                                           pre_filter, use_node_attr)
        self.data, self.slices,_ = torch.load(self.processed_paths[0])
        self.node_score = torch.zeros(self.data['x'].shape[0])
        self.node_Lipschitz = torch.zeros(self.data['x'].shape[0])

    @property
    def processed_file_names(self):
        return self.processed_filename

    def get(self, idx):
        data = self.data.__class__()


        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.slices.keys():
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                                        item)] = slice(slices[idx],
                                                       slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        data.degree = degree(data.edge_index[0])
        node_num = data.edge_index.max()
        sl = torch.tensor([[n, n] for n in range(node_num)]).t()
        data.edge_index = torch.cat((data.edge_index, sl), dim=1)

        if self.aug == 'none':
            None
        
        elif self.aug == 'pre':
            data.node_score = self.node_score[self.slices['x'][idx]:self.slices['x'][idx + 1]]
            data.node_weight = self.node_Lipschitz[self.slices['x'][idx]:self.slices['x'][idx + 1]]
        
        elif self.aug == 'dropN':
            data.node_weight = self.node_Lipschitz[self.slices['x'][idx]:self.slices['x'][idx + 1]]
            nodes_l = data.node_weight
            data.node_score = self.node_score[self.slices['x'][idx]:self.slices['x'][idx + 1]]
            data.node_score = data.node_score * (1 - nodes_l) + nodes_l
            data = drop_nodes_prob(data, self.aug_ratio, data.node_score)

        elif self.aug == 'dropN_cp':
            data.node_weight = self.node_Lipschitz[self.slices['x'][idx]:self.slices['x'][idx + 1]]
            nodes_l = data.node_weight
            data.node_score = self.node_score[self.slices['x'][idx]:self.slices['x'][idx + 1]]
            data.node_score = data.node_score * (1 - nodes_l) + nodes_l
            data = drop_nodes_cp(data, self.aug_ratio, data.node_score)

        else:
            print('augmentation error')
            assert False

        return data


def drop_nodes_prob(data, aug_ratio, node_score):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * aug_ratio)

    node_prob = node_score.float()
    node_prob += 0.001
    node_prob = np.array(node_prob)
    node_prob /= node_prob.sum()

    idx_nondrop = np.random.choice(node_num, node_num-drop_num, replace=False, p=node_prob)
    idx_drop_set = set(np.setdiff1d(np.arange(node_num), idx_nondrop).tolist())
    idx_nondrop.sort()

    idx_dict = np.zeros((idx_nondrop[-1] + 1,), dtype=np.int64)
    idx_dict[idx_nondrop] = np.arange(len(idx_nondrop), dtype=np.int64)

    edge_index = data.edge_index.numpy()

    edge_mask = []
    for n in range(edge_num):
        if not (edge_index[0, n] in idx_drop_set or edge_index[1, n] in idx_drop_set):
            edge_mask.append(n)
    edge_mask = np.asarray(edge_mask, dtype=np.int64)
    edge_index = idx_dict[edge_index[:, edge_mask]]
    try:
        data.edge_index = torch.from_numpy(edge_index)
        data.x = data.x[idx_nondrop]
        data.edge_attr = data.edge_attr[edge_mask]
    except:
        data = data

    return data


def drop_nodes_cp(data, aug_ratio, node_score):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * aug_ratio)

    node_prob = max(node_score.float()) - node_score.float()
    node_prob += 0.001
    node_prob = np.array(node_prob)
    node_prob /= node_prob.sum()

    idx_nondrop = np.random.choice(node_num, node_num-drop_num, replace=False, p=node_prob)
    idx_drop_set = set(np.setdiff1d(np.arange(node_num), idx_nondrop).tolist())
    idx_nondrop.sort()

    idx_dict = np.zeros((idx_nondrop[-1] + 1,), dtype=np.int64)
    idx_dict[idx_nondrop] = np.arange(len(idx_nondrop), dtype=np.int64)

    edge_index = data.edge_index.numpy()

    edge_mask = []
    for n in range(edge_num):
        if not (edge_index[0, n] in idx_drop_set or edge_index[1, n] in idx_drop_set):
            edge_mask.append(n)
    edge_mask = np.asarray(edge_mask, dtype=np.int64)
    edge_index = idx_dict[edge_index[:, edge_mask]]
    try:
        data.edge_index = torch.from_numpy(edge_index)
        data.x = data.x[idx_nondrop]
        data.edge_attr = data.edge_attr[edge_mask]
    except:
        data = data

    return data


def drop_nodes(data, aug_ratio):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num  * aug_ratio)

    idx_perm = np.random.permutation(node_num)

    idx_drop = idx_perm[:drop_num]
    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]:n for n in list(range(idx_nondrop.shape[0]))}

    edge_index = data.edge_index.numpy()
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj = adj[idx_nondrop, :][:, idx_nondrop]
    edge_index = adj.nonzero().t()

    try:
        data.edge_index = edge_index
        data.x = data.x[idx_nondrop]
    except:
        data = data
    return data


def weighted_drop_nodes(data, aug_ratio, npower):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num  * aug_ratio)

    adj = np.zeros((node_num, node_num))
    adj[data.edge_index[0], data.edge_index[1]] = 1
    deg = adj.sum(axis=1)
    deg[deg==0] = 0.1
    # print(deg)
    # deg = deg ** (-1)
    deg = deg ** (npower)
    # print(deg)
    # print(deg / deg.sum())
    # assert False

    idx_drop = np.random.choice(node_num, drop_num, replace=False, p=deg / deg.sum())

    # idx_perm = np.random.permutation(node_num)
    # idx_drop = idx_perm[:drop_num]
    # idx_nondrop = idx_perm[drop_num:]

    idx_nondrop = np.array([n for n in range(node_num) if not n in idx_drop])

    # idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]:n for n in list(range(idx_nondrop.shape[0]))}

    edge_index = data.edge_index.numpy()
    ###
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj = adj[idx_nondrop, :][:, idx_nondrop]
    edge_index = adj.nonzero().t()

    ###
    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    try:
        data.edge_index = edge_index
        data.x = data.x[idx_nondrop]
    except:
        data = data
    return data


def permute_edges(data, aug_ratio):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)

    edge_index = data.edge_index.numpy()

    idx_add = np.random.choice(node_num, (2, permute_num))

    # idx_add = [[idx_add[0, n], idx_add[1, n]] for n in range(permute_num) if not (idx_add[0, n], idx_add[1, n]) in edge_index]
    # edge_index = [edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)] + idx_add

    edge_index = np.concatenate((edge_index[:, np.random.choice(edge_num, (edge_num - permute_num), replace=False)], idx_add), axis=1)
    data.edge_index = torch.tensor(edge_index)

    return data

def subgraph(data, aug_ratio):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * aug_ratio)

    edge_index = data.edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    idx_nondrop = idx_sub
    data.x = data.x[idx_nondrop]
    idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}

    edge_index = data.edge_index.numpy()
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[list(range(node_num)), list(range(node_num))] = 1
    adj = adj[idx_nondrop, :][:, idx_nondrop]
    edge_index = adj.nonzero().t()

    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
    data.edge_index = edge_index

    return data


def mask_nodes(data, aug_ratio):

    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)

    token = data.x.mean(dim=0)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(token, dtype=torch.float32)

    return data





