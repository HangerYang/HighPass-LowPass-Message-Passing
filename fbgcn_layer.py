from torch import mm, nn, rand
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class FBGCN_Layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
      
        self.high = nn.Linear(in_dim, out_dim, bias = False)
        self.conv = GCNConv(in_dim, out_dim)
        
    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.high.weight, gain)
        
        self.aL = nn.Parameter(rand(1))
        self.aH = nn.Parameter(rand(1))
        
        self.conv.reset_parameters()

    def forward(self, x, edge_index, lap, d_inv):
        
        Lhp = mm(mm(d_inv, lap), d_inv)
        Hh = mm(Lhp, F.relu(self.high(x)))
        
        Hl = self.conv(x, edge_index)

        return (self.aL * Hl + self.aH * Hh)