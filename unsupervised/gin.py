import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, GCNConv, GATConv, SAGEConv, global_add_pool
from torch_geometric.utils import softmax

import numpy as np
import torch_scatter


class Explainer(torch.nn.Module):
    def __init__(self, device, num_features, dim, num_gc_layers):
        super(Explainer, self).__init__()
        
        self.device = device
        self.num_gc_layers = num_gc_layers

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        self.dim = dim

        for i in range(num_gc_layers):
            if i and i < num_gc_layers - 1:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
                bn = torch.nn.BatchNorm1d(dim)
            elif i == num_gc_layers - 1:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, 1))
                bn = torch.nn.BatchNorm1d(1)
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
                bn = torch.nn.BatchNorm1d(dim)

            conv = GINConv(nn)
            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, data,  L = False):
        
        device = self.device
        
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)

        xs = []
        
        if L == True:
            mask = torch.ones(x.shape[0], x.shape[0]).to(device)
            mask_diag = torch.diag(mask) 
            mask_diag = torch.diag_embed(mask_diag)
            mask = mask - mask_diag[0]
            for i in range(self.num_gc_layers):
                if i !=  self.num_gc_layers - 1:
                    x = F.relu(self.bns[i](self.convs[i](x, edge_index)))
                else:
                    x = self.bns[i](self.convs[i](x, edge_index))
                xs.append(x)
            atten = F.softmax(xs[-1] * xs[-1].T, dim = -1)
            node_l = torch.matmul(mask*atten, xs[-1])
            node_prob = xs[-1]
            return node_prob, node_l

        else:
            for i in range(self.num_gc_layers):
                if i !=  self.num_gc_layers - 1:
                    x = F.relu(self.bns[i](self.convs[i](x, edge_index)))
                else:
                    x = self.bns[i](self.convs[i](x, edge_index))
                xs.append(x)
            node_prob = xs[-1]
            node_prob = softmax(node_prob / 5.0, batch)
            return node_prob

class Encoder(torch.nn.Module):
    def __init__(self, device, num_features, dim, num_gc_layers, pooling):
        super(Encoder, self).__init__()
        
        self.device = device
        self.num_gc_layers = num_gc_layers
        self.pooling = pooling

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.nns = torch.nn.ModuleList()
        self.dim = dim

        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
                #conv = SAGEConv(dim, dim, add_self_loops=False)
                conv = GINConv(nn)
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
                conv = SAGEConv(num_features, dim, add_self_loops=False)
            #conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)
            self.nns.append(nn)

    def forward(self, x, edge_index, batch, node_imp = None):
        device = self.device
        
        # mapping node_imp to [0.9,1.1]
        if node_imp is not None:
            node_imp = node_imp.reshape(-1, 1)
            out, _ = torch_scatter.scatter_max(torch.reshape(node_imp, (1, -1)), batch)
            out = out.reshape(-1, 1)
            out = out[batch]
            node_imp /= (out*10)
            node_imp += 0.9
            node_imp = node_imp.expand(-1, self.dim)
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)

        xs = []
        for i in range(self.num_gc_layers):

            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)

            if node_imp is not None:
                x_imp = x * node_imp
            else:
                x_imp = x

            xs.append(x_imp)

        if self.pooling == 'last':
            x = global_add_pool(xs[-1], batch)
        else:
            xpool = [global_add_pool(x, batch) for x in xs]
            x = torch.cat(xpool, 1)

        return x, torch.cat(xs, 1)

    def get_embeddings(self, loader):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                #data = data[0]
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                x, _ = self.forward(x, edge_index, batch, None)

                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y

    def get_embeddings_v(self, loader):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for n, data in enumerate(loader):
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                x_g, x = self.forward(x, edge_index, batch, None)
                x_g = x_g.cpu().numpy()
                ret = x.cpu().numpy()
                y = data.edge_index.cpu().numpy()
                print(data.y)
                if n == 1:
                   break

        return x_g, ret, y