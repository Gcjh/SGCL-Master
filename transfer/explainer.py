import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv
from torch_geometric.utils import softmax

class Explainer(torch.nn.Module):
    def __init__(self, device, num_features, dim, num_gc_layers, drop_ratio):
        super(Explainer, self).__init__()
        
        self.device = device
        self.num_gc_layers = num_gc_layers
        self.drop_ratio = drop_ratio

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        self.dim = [dim, 128, 64, 32]

        for i in range(num_gc_layers):
            nn = Sequential(Linear(self.dim[i], self.dim[i+1]), ReLU(), Linear(self.dim[i+1], self.dim[i+1]))
            bn = torch.nn.BatchNorm1d(self.dim[i+1])
            '''
            elif i == num_gc_layers - 1:
                nn = Sequential(Linear(self.dim[i], self.dim[i + 1]), ReLU(), Linear(self.dim[i + 1], 1))
                bn = torch.nn.BatchNorm1d(1)
            
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
                bn = torch.nn.BatchNorm1d(dim)
            '''

            conv = GINConv(nn)

            self.convs.append(conv)
            self.bns.append(bn)
            
            self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, batch,  L = False):
        
        device = self.device
        
        x = x
        edge_index = edge_index
        batch = batch
        
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)
        
        xs = []
        
        if L == True:
            mask = torch.ones(x.shape[0], x.shape[0]).to(device)
            mask_diag = torch.diag(mask) 
            mask_diag = torch.diag_embed(mask_diag)
            mask = mask - mask_diag[0]
            for i in range(self.num_gc_layers):
                x = self.bns[i](self.convs[i](x, edge_index))
                if i !=  self.num_gc_layers - 1:
                    x = F.dropout(F.relu(x), self.drop_ratio, training = self.training)
                else:
                    x = F.dropout(x, self.drop_ratio, training = self.training)
                xs.append(x)
            h = xs[-1]
            h = self.linear(h)
            atten = F.softmax(h * h.T, dim = -1)
            node_l = torch.matmul(mask*atten, h)
            node_prob = h
            return node_prob, node_l

        else:
            for i in range(self.num_gc_layers):
                x = self.bns[i](self.convs[i](x, edge_index))
                if i !=  self.num_gc_layers - 1:
                    x = F.dropout(F.relu(x), self.drop_ratio, training = self.training)
                else:
                    x = F.dropout(x, self.drop_ratio, training = self.training)
                xs.append(x)
            h = xs[-1]
            node_prob = self.linear(h)
            node_prob = softmax(node_prob / 5.0, batch)
            return node_prob
