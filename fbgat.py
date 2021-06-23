from torch.nn import functional as F
from torch import nn
from fbgat_layer import FBGAT_Layer as fbgat

class FBGAT(nn.Module):
    def __init__(self, n_head, in_dim, hi_dim, out_dim, dropout):
        """
        :param in_dim: input dimension
        :param hi_dim: hidden dimension
        :param out_dim: output dimension
        :param dropout: dropout ratio
        """
        super().__init__()
        self.fbgats = nn.ModuleList()
        # first layer
        self.fbgats.append(fbgat(in_dim, hi_dim, heads = n_head, dropout = dropout, concat = True))
        # last layer
        self.fbgats.append(fbgat(hi_dim * n_head, out_dim, heads = 1, dropout = dropout, concat = False))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for fbgat in self.fbgats:
            fbgat.reset_parameters()

    def forward(self, x, edge_index, lap, d_inv):
        # first layer
        x = self.fbgats[0](x, edge_index, lap, d_inv)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # last layer
        return self.fbgats[-1](x, edge_index, lap, d_inv)