from gin import Encoder, Explainer
import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import softmax
import torch.nn.functional as F

class simclr(nn.Module):
    def __init__(self, args, dataset_num_features = 1, alpha=0.5, beta=1., gamma=.1):
        super(simclr, self).__init__()
        
        self.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

        self.hidden_dim = args.hidden_dim
        self.num_gc_layers = args.num_gc_layers
        self.pooling = args.pooling

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior

        if self.pooling == 'last':
            self.embedding_dim = self.hidden_dim
        else:
            self.embedding_dim = self.hidden_dim*self.num_gc_layers
        self.encoder = Encoder(self.device, dataset_num_features, self.hidden_dim, self.num_gc_layers, self.pooling)
        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))

        self.init_emb()

    def init_emb(self):
        # initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data, w = None):
        
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        
        if x is None:
            x = torch.ones(batch.shape[0]).to(self.device)

        y, M = self.encoder(x, edge_index, batch, w)

        y = self.proj_head(y)

        return y

    def loss_cal(self, x, y, x_cp):

        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        y_abs = y.norm(dim=1)
        x_cp_abs = x_cp.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, y) / torch.einsum('i,j->ij', x_abs, y_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        

        sim_matrix_cp_x = torch.einsum('ik,jk->ij', x, x_cp) / torch.einsum('i,j->ij', x_abs, x_cp_abs)
        sim_matrix_cp_x = torch.exp(sim_matrix_cp_x / T)

        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss1 = pos_sim / (sim_matrix.sum(dim=1))
        loss2 = pos_sim / (sim_matrix_cp_x.sum(dim=1) + pos_sim)

        loss = loss1 + 0.01*loss2   ###0.01
        loss = - torch.log(loss).mean()

        return loss
        
        
class Generator(nn.Module):
    def __init__(self, args, dataset_num_features = 1):
        super(Generator, self).__init__()
        self.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
        self.hidden_dim = args.hidden_dim
        self.num_gc_layers = args.num_gc_layers
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