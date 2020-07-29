import sys, os.path as osp, os, math, numpy as np, argparse
import torch
import torch_sparse # Not 100% sure if needed
import gc
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import EdgeConv

class EdgeNetWithCategoriesJittable(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=8, output_dim=4, n_iters=1, aggr='add',
             norm=torch.tensor([1./500., 1./500., 1./54., 1/25., 1./1000.])):
        super(EdgeNetWithCategoriesJittable, self).__init__()

        self.datanorm = nn.Parameter(norm)
    
        start_width = 2 * (hidden_dim + input_dim)
        middle_width = (3 * hidden_dim + 2*input_dim) // 2
    
        self.n_iters = n_iters
            
        self.inputnet =  nn.Sequential(
            nn.Linear(input_dim, 2*hidden_dim),
            nn.Tanh(),
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            nn.Tanh(),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.edgenetwork = nn.Sequential(
            nn.Linear(2*n_iters*hidden_dim, 2*hidden_dim),
            nn.ELU(),
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            nn.ELU(),
            nn.Linear(2*hidden_dim, output_dim),
            nn.LogSoftmax(dim=-1),
        )
        
        convnn = nn.Sequential(
            nn.Linear(start_width, middle_width),
            nn.ELU(),
            #nn.Dropout(p=0.5, inplace=False),
            nn.Linear(middle_width, hidden_dim),
            nn.ELU()
        )
        self.firstnodenetwork = EdgeConv(nn=convnn, aggr=aggr).jittable()
        self.nodenetwork = nn.ModuleList()
        for i in range(n_iters - 1):
            convnn = nn.Sequential(
                nn.Linear(start_width, middle_width),
                nn.ELU(),
                #nn.Dropout(p=0.5, inplace=False),
                nn.Linear(middle_width, hidden_dim),
                nn.ELU()
            )
            self.nodenetwork.append(EdgeConv(nn=convnn, aggr=aggr).jittable())
    
    def forward(self, x, edge_index):
        row = edge_index[0]
        col = edge_index[1]
        x_norm = self.datanorm * x
        H = self.inputnet(x_norm)
        H = self.firstnodenetwork(torch.cat([H, x_norm], dim=-1), edge_index)
        H_cat = H
        for nodenetwork in self.nodenetwork:
            H = nodenetwork(torch.cat([H, x_norm], dim=-1), edge_index)
            H_cat = torch.cat([H, H_cat], dim=-1)
        return self.edgenetwork(torch.cat([H_cat[row],H_cat[col]],dim=-1)).squeeze(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint.pt', default='model_checkpoint_EdgeNetWithCategories_264403_5b5c05404f_csharma.best.pth.tar')
    parser.add_argument('--n_iters', type=int, default=6)
    parser.add_argument('--input_dim', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=64)
    args = parser.parse_args()

    nonjit_model = EdgeNetWithCategoriesJittable(
        n_iters=args.n_iters,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim
        )
    model = torch.jit.script(nonjit_model)

    pretrained_dict = torch.load(args.checkpoint, map_location=torch.device('cpu'))['model']
    model_dict = model.state_dict()

    map_pretrained_to_jitmodel = {
        'nodenetwork0.nn.0.weight' : 'firstnodenetwork.nn.0.weight',
        'nodenetwork0.nn.0.bias'   : 'firstnodenetwork.nn.0.bias',
        'nodenetwork0.nn.2.weight' : 'firstnodenetwork.nn.2.weight',
        'nodenetwork0.nn.2.bias'   : 'firstnodenetwork.nn.2.bias',
        'nodenetwork1.nn.0.weight' : 'nodenetwork.0.nn.0.weight',
        'nodenetwork1.nn.0.bias'   : 'nodenetwork.0.nn.0.bias',
        'nodenetwork1.nn.2.weight' : 'nodenetwork.0.nn.2.weight',
        'nodenetwork1.nn.2.bias'   : 'nodenetwork.0.nn.2.bias',
        'nodenetwork2.nn.0.weight' : 'nodenetwork.1.nn.0.weight',
        'nodenetwork2.nn.0.bias'   : 'nodenetwork.1.nn.0.bias',
        'nodenetwork2.nn.2.weight' : 'nodenetwork.1.nn.2.weight',
        'nodenetwork2.nn.2.bias'   : 'nodenetwork.1.nn.2.bias',
        'nodenetwork3.nn.0.weight' : 'nodenetwork.2.nn.0.weight',
        'nodenetwork3.nn.0.bias'   : 'nodenetwork.2.nn.0.bias',
        'nodenetwork3.nn.2.weight' : 'nodenetwork.2.nn.2.weight',
        'nodenetwork3.nn.2.bias'   : 'nodenetwork.2.nn.2.bias',
        'nodenetwork4.nn.0.weight' : 'nodenetwork.3.nn.0.weight',
        'nodenetwork4.nn.0.bias'   : 'nodenetwork.3.nn.0.bias',
        'nodenetwork4.nn.2.weight' : 'nodenetwork.3.nn.2.weight',
        'nodenetwork4.nn.2.bias'   : 'nodenetwork.3.nn.2.bias',
        'nodenetwork5.nn.0.weight' : 'nodenetwork.4.nn.0.weight',
        'nodenetwork5.nn.0.bias'   : 'nodenetwork.4.nn.0.bias',
        'nodenetwork5.nn.2.weight' : 'nodenetwork.4.nn.2.weight',
        'nodenetwork5.nn.2.bias'   : 'nodenetwork.4.nn.2.bias',
        }

    print('Loading checkpoint {0} into model'.format(args.checkpoint))
    for key in pretrained_dict:
        model_key = map_pretrained_to_jitmodel.get(key, key)

        if not model_key in model_dict:
            print('Skipping key {0}, not in model_dict'.format(model_key))
            continue

        if key != model_key:
            print('Mapping {0} --> {1}'.format(key, model_key))
        
        model_dict[model_key] = pretrained_dict[key]
        
    model.load_state_dict(model_dict)
    torch.jit.save(model, 'edgenetwithcats.pt')


if __name__ == '__main__':
    main()
