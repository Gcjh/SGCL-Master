import random
import torch
import numpy as np
import os.path as osp
from dataset import TUDataset_aug as TUDataset
from torch_geometric.loader import DataLoader
from arguments import arg_parse
from model import simclr, Generator
from evaluate_embedding import evaluate_embedding
from tqdm import tqdm
import json
from torch_geometric.data import Batch

def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':

    args = arg_parse()

    setup_seed(args.seed)

    accuracies = {'test': [], 'std': []}
    imp_batch_size = 16
    epochs = args.epochs
    log_interval = args.log_interval
    batch_size = args.batch_size
    lr = args.lr
    DS = args.DS
    num_workers = args.num_workers
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)

    dataset = TUDataset(path, name=DS, aug=args.aug, rho=args.rho)
    dataset_eval = TUDataset(path, name=DS, aug='none')
    print(len(dataset))
    print(dataset.get_num_feature())

    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size, num_workers=2*num_workers)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    model = simclr(args, dataset_num_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('================')
    print('dataset: {}'.format(DS))
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('pooling: {}'.format(args.pooling))
    print('================')
    
    gen = Generator(args, dataset_num_features).to(device)
    view_optimizer = torch.optim.Adam(gen.parameters(),lr=lr,weight_decay=0.0)

    best_acc = 0
    best_std = 0
    
    for epoch in range(1, epochs+1):

        dataset.aug = "none"
        dataloader = DataLoader(dataset, batch_size=imp_batch_size, num_workers=num_workers, shuffle=False)
        model.eval()
        torch.set_grad_enabled(False)
        print('Lipschitz Computation Begin!')
        for step, data in enumerate(tqdm(dataloader)):
            node_index_start = step * imp_batch_size
            node_index_end = min(node_index_start + imp_batch_size - 1, len(dataset) - 1)
            data = data.to(device)
            nodes_imp, nodes_Lipschitz = gen.Node_Lipschitz(data)
            nodes_imp = nodes_imp.detach()
            nodes_Lipschitz = nodes_Lipschitz.detach()
            dataset.node_score[dataset.slices['x'][node_index_start]:dataset.slices['x'][node_index_end + 1]] = \
                torch.squeeze(nodes_imp)
            dataset.node_Lipschitz[dataset.slices['x'][node_index_start]:dataset.slices['x'][node_index_end + 1]] = \
                torch.squeeze(nodes_Lipschitz)
        print(' ')
        print('Lipschitz Computation Completed!')
        print('================')
        loss_all = 0
        dataset.aug = args.aug
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=num_workers, shuffle=True)
        model.train()
        torch.set_grad_enabled(True)
        for data in tqdm(dataloader):

            data, data_aug, data_cp = data
            
            num = int(len(data)*0.8)
            if num == 0:
                num = 1
                
            data = Batch.from_data_list(data[:num])
            data_aug = Batch.from_data_list(data_aug[:num])
            data_cp = Batch.from_data_list(data_cp[:num])
            
            optimizer.zero_grad()
            view_optimizer.zero_grad()

            data = data.to(device)
            data_imp = gen.explain(data)
            L = data.node_weight.reshape(-1, 1)
            data_imp = data_imp * (1 - L) + L
            x = model(data, data_imp)

            data_aug = data_aug.to(device)
            data_cp = data_cp.to(device)

            data_aug_imp = gen.explain(data_aug)
            x_aug = model(data_aug, data_aug_imp)

            data_cp_imp = gen.explain(data_cp)
            x_cp = model(data_cp, data_cp_imp)

            loss = model.loss_cal(x, x_aug, x_cp)
            loss_all += loss.item() * data.num_graphs

            regularization_loss = 0
            for param in gen.parameters():
                regularization_loss += torch.sum(torch.abs(param))
            
            loss += args.l * regularization_loss
            loss_all += args.l * regularization_loss

            loss.backward()
            optimizer.step()
            view_optimizer.step()
            
        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))
        
        if (epoch) % log_interval == 0:
            model.eval()
            emb, y = model.encoder.get_embeddings(dataloader_eval)
            acc, std = evaluate_embedding(emb, y)
            if acc > best_acc:
                best_acc = acc
                best_std = std
            accuracies['std'].append(std)
            accuracies['test'].append(acc)
    print('Best Acc: {:.2f} ± {:.2f}'.format(best_acc*100, best_std*100))
    tpe = ('local' if args.local else '') + ('prior' if args.prior else '')
    with open('logs/0.8_log_' + args.DS + '.txt', 'a+') as f:
        s1 = json.dumps(accuracies['test'])
        s2 = json.dumps(accuracies['std'])
        f.write('DS:{}, seed:{}, layer:{}\n'.format(args.DS, args.seed, args.num_gc_layers))
        f.write('{}\n'.format(s1))
        f.write('Best Acc: {:.2f} ± {:.2f}\n'.format(best_acc*100, best_std*100))
        f.write('\n')
