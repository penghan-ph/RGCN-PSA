# Time Shift Attention

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.TimeEncoder import TimeEncoder

class TSA(nn.Module):
    def __init__(self, args):
        super(TSA, self).__init__()
        self.args = args
        self.Wq = nn.Parameter(torch.FloatTensor(self.args.dim_tsa_hidden, self.args.num_nodes, self.args.dim_short_hidden+self.args.dim_clusters_hidden)) # [dim_tsa_hidden*N*(self.args.dim_short_hidden+self.dim_clusters_hidden)]
        self.Wk = nn.Parameter(torch.FloatTensor(self.args.dim_tsa_hidden, self.args.num_nodes, self.args.yearly_window)) # [dim_tsa_hidden*N*self.args.yearly_window]
        self.V = nn.Parameter(torch.FloatTensor(self.args.dim_tsa_hidden, self.args.num_nodes, self.args.yearly_window)) # [dim_tsa_hidden*N*self.args.yearly_window]
        # self.gru = nn.GRU(input_size=self.args.num_nodes, hidden_size=self.args.num_nodes, batch_first=True)
        self.encoder = TimeEncoder(self.args.yearly_window, self.args.dim_long_hidden, self.args.num_nodes)

    def forward(self, x_long, x_short):
        # print(x_long.shape, x_short.shape) #bnty bnd
        Q = torch.einsum('hnd,bnd->bhn', self.Wq, x_short).unsqueeze(dim=-1).unsqueeze(dim=-1)
        # print(Q.shape)
        K = torch.einsum('hny,bnty->bhnty', self.Wk, x_long)
        # print(K.shape)
        A = torch.einsum('hny,bhnty->bnty', self.V, torch.tanh(Q+K))
        # print(A.shape)
        A_ = F.softmax(A, dim=2) #bnty
        x_long_new = (A_ * x_long).sum(dim=2) #bny
        # print(x_long_new.shape)
        # x_long_pre, _ = self.gru(x_long_new.permute(0,2,1)) #byn

        x_long_hidden = self.encoder(x_long_new) #b,n,dim_long_hidden
        # print(x_long_pre.shape)
        # return x_long_pre[:,-1,:]
        # return x_long_pre
        return x_long_hidden