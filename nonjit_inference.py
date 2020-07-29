"""
Script to run the inference of the testdata locally.
NOTE: Requires an environment with torch and torch dependencies installed
"""
import os, os.path as osp, math, numpy as np, argparse, glob
import torch, torch.nn as nn
from torch_geometric.nn import EdgeConv
from torch_geometric.data import Data

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device', device)

def build_edge_index(n_nodes, Ri_rows, Ri_cols, Ro_rows, Ro_cols):
    # Warning: not a very well optimized function
    n_edges = Ri_rows.shape[0]
    spRi_idxs = np.stack([Ri_rows.astype(np.int64), Ri_cols.astype(np.int64)])
    spRi_vals = np.ones((Ri_rows.shape[0],), dtype=np.float32)
    spRi = (spRi_idxs,spRi_vals,n_nodes,n_edges)

    spRo_idxs = np.stack([Ro_rows.astype(np.int64), Ro_cols.astype(np.int64)])
    spRo_vals = np.ones((Ro_rows.shape[0],), dtype=np.float32)
    spRo = (spRo_idxs,spRo_vals,n_nodes,n_edges)

    Ro = spRo[0].T.astype(np.int64)
    Ri = spRi[0].T.astype(np.int64)
    
    i_out = Ro[Ro[:,1].argsort(kind='stable')][:,0]
    i_in  = Ri[Ri[:,1].argsort(kind='stable')][:,0]
    edge_index = np.stack((i_out,i_in))
    return edge_index

class EdgeNetWithCategories(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=8, output_dim=4, n_iters=1, aggr='add',
                 norm=torch.tensor([1./500., 1./500., 1./54., 1/25., 1./1000.])):
        super(EdgeNetWithCategories, self).__init__()

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

        self.edgenetwork = nn.Sequential(nn.Linear(2*n_iters*hidden_dim, 2*hidden_dim),
                                         nn.ELU(),  
                                         nn.Linear(2*hidden_dim, 2*hidden_dim),                                         
                                         nn.ELU(),
                                         nn.Linear(2*hidden_dim, output_dim),
                                         nn.LogSoftmax(dim=-1),
        )
        
        for i in range(n_iters):
            convnn = nn.Sequential(nn.Linear(start_width, middle_width),
                                   nn.ELU(),
                                   #nn.Dropout(p=0.5, inplace=False),
                                   nn.Linear(middle_width, hidden_dim),                                             
                                   nn.ELU()                                   
                                  )
            setattr(self, 'nodenetwork%d' % i, EdgeConv(nn=convnn, aggr=aggr))
        
    def forward(self, data):        
        row,col = data.edge_index
        x_norm = self.datanorm * data.x
        H = self.inputnet(x_norm)
        H = getattr(self,'nodenetwork0')(torch.cat([H, x_norm], dim=-1), data.edge_index)
        H_cat = H
        for i in range(1,self.n_iters):            
            H = getattr(self,'nodenetwork%d' % i)(torch.cat([H, x_norm], dim=-1), data.edge_index)
            H_cat = torch.cat([H, H_cat], dim=-1)                    
        return self.edgenetwork(torch.cat([H_cat[row],H_cat[col]],dim=-1)).squeeze(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint.pt', default='make_jit_model/model_checkpoint_EdgeNetWithCategories_264403_5b5c05404f_csharma.best.pth.tar')
    parser.add_argument('--n_iters', type=int, default=6)
    parser.add_argument('--input_dim', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=64)
    args = parser.parse_args()

    model = EdgeNetWithCategories(
        n_iters=args.n_iters,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim
        )
    model.load_state_dict(torch.load(args.checkpoint)['model'])
    model.to(device)
    print(model)

    with torch.no_grad():
        for npzfile in sorted(glob.glob('hgcal_testdata/*.npz')):
            print(npzfile)
            with np.load(npzfile) as npzdata:
                x = npzdata['X'].astype(np.float32)
                edge_index = build_edge_index(
                    x.shape[0],
                    npzdata['Ri_rows'], npzdata['Ri_cols'], npzdata['Ro_rows'], npzdata['Ro_cols']
                    )
                y = npzdata['y']
                print(x.shape, edge_index.shape)
                data = Data(
                    x=torch.from_numpy(x),
                    edge_index=torch.from_numpy(edge_index),
                    y=torch.from_numpy(y)
                    ).to(device)

            output = model(data)
            print(output)
            del data



if __name__ == '__main__':
    main()