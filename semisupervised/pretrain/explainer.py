import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv
from torch_geometric.utils import softmax

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
