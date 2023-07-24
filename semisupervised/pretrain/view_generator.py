from explainer import Explainer
import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import softmax


class Generator(nn.Module):
    def __init__(self, device, hidden, dataset_num_features = 1):
        super(Generator, self).__init__()
        self.device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")
        self.hidden_dim = hidden
        self.num_gc_layers = 3
        self.encoder = Explainer(self.device, dataset_num_features, self.hidden_dim, 3)
        self.init_emb()
    
    def init_emb(self):
        # initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
                    
    def Node_Lipschitz(self, data):
        
        degree, batch = data.degree, data.batch
        
        h, aug_h = self.encoder(data, L = True)

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

        y = self.encoder(data, L = False)

        return y