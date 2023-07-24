from explainer import Explainer
import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import softmax

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 

class Generator(nn.Module):
    def __init__(self, args, dataset_num_features = 1):
        super(Generator, self).__init__()
        self.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
        
        self.hidden_dim = args.emb_dim
        self.num_gc_layers = args.num_layer
        
        self.encoder = Explainer(self.device, dataset_num_features, self.hidden_dim, 3, args.dropout_ratio)
        
        self.x_embedding1 = torch.nn.Embedding(num_atom_type, self.hidden_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, self.hidden_dim)
        
        #torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        #torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        
        self.init_emb()
    
    def init_emb(self):
        # initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
                    
    def Node_Lipschitz(self, data):
        
        x, edge_index, degree, batch = data.x, data.edge_index, data.degree, data.batch
        
        if x is None:
            y = torch.ones((batch.shape[0], 1)).to(device)
        else:
            y = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        
        h, aug_h = self.encoder(y, edge_index, batch, L = True)

        L = softmax(h / 5.0, batch)

        h = global_add_pool(h, batch)
        h = h.reshape(-1)
        aug_h = torch.squeeze(aug_h)

        delta_h = torch.abs(h[batch] - aug_h).div(degree + 1)

        delta_h_list = delta_h.reshape(-1, 1)

        avg_delta_h = global_mean_pool(delta_h_list, batch)
        avg_delta_h = avg_delta_h.reshape(-1)

        bool_L = (delta_h >= avg_delta_h[batch]).float().reshape(-1, 1)

        return L, bool_L
        
        
    def explain(self, data):
    
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        if x is None:
            y = torch.ones((batch.shape[0], 1)).to(device)
        else:
            y = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        z = self.encoder(y, edge_index, batch, L = False)

        return z