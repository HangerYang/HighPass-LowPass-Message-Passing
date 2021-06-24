from torch import mm, nn, rand
from torch_geometric.nn import GATConv
import torch.nn.functional as F

class FBGAT_Layer(nn.Module):
    def __init__(self,n_head, in_dim, out_dim, dropout, concat):
        super().__init__()
      
        self.high = nn.Linear(in_dim, out_dim * n_head, bias = False)
        self.gat = GATConv(in_dim, out_dim, heads = n_head, dropout = dropout, concat = concat)
        self.dropout = dropout

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.high.weight, gain)
        
        self.aL = nn.Parameter(rand(1))
        self.aH = nn.Parameter(rand(1))

        self.gat.reset_parameters()

    def forward(self, x, edge_index, lap, d_inv):
        
        Lhp = mm(mm(d_inv, lap), d_inv)
        Hh = mm(Lhp, F.relu(self.high(x)))
        
        Hl = F.elu(self.gat(x, edge_index))
        Hl = F.dropout(Hl, training=self.training)

        return (self.aL * Hl + self.aH * Hh)