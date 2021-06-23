from torch.nn import functional as F
from torch import nn
from fbgcn_layer import FBGCN_Layer as fbgcn

class FBGCN(nn.Module):
    def __init__(self, n_layer, in_dim, hi_dim, out_dim, dropout):
        """
        :param n_layer: number of layers
        :param in_dim: input dimension
        :param hi_dim: hidden dimension
        :param out_dim: output dimension
        :param dropout: dropout ratio
        """
        super().__init__()
        assert(n_layer > 0)
        self.num_layers = n_layer
        self.fbgcns = nn.ModuleList()
        # first layer
        self.fbgcns.append(fbgcn(in_dim, hi_dim, dropout))
        # inner layers
        for _ in range(n_layer - 2):
            self.fbgcns.append(fbgcn(hi_dim, hi_dim, dropout))
        # last layer
        self.fbgcns.append(fbgcn(hi_dim, out_dim, dropout))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for fbgcn in self.fbgcns:
            fbgcn.reset_parameters()
    def forward(self, x, edge_index, lap, d_inv):
        # first layer
        x = self.fbgcns[0](x, edge_index, lap, d_inv)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # inner layers
        if self.num_layers > 2:
            for layer in range(self.num_layers - 1):
                 x = self.fbgcns[0](x, edge_index, lap, d_inv)
                 x = F.dropout(x, p=self.dropout, training=self.training)
        # last layer
        return self.fbgcns[-1](x, edge_index, lap, d_inv)

