from util import data_sample
from torch_geometric.datasets import WebKB
from util import lap_dinv as util
from fbgcn import FBGCN
import torch
from tqdm import tqdm
import torch.nn.functional as F

name_data = 'Cornell'
dataset = WebKB(root= '/tmp/' + name_data, name = name_data)
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

data = dataset[0]
features = data.x
num_nodes = features.shape[0]
num_features = features.shape[1]
out_dim = dataset.num_classes
lap, d_inv = util(data.edge_index, num_nodes)
train_mask, test_mask = data_sample(num_nodes)
d_inv = d_inv.to(device)
lap = lap.to(device)
x = data.x.to(device)
data = data.to(device)
edge_index = data.edge_index.to(device)

# model = FBGCN(2, num_features, 16, out_dim, 0.5).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-5)
# model.train()
# train_mask, test_mask = data_sample(num_nodes)
# for epoch in tqdm(range(20)):
#     optimizer.zero_grad()
#     out = model(x, edge_index, lap, d_inv)
#     loss = F.nll_loss(out[train_mask], data.y[train_mask]) 
#     loss.backward()
#     optimizer.step()
# model.eval()
# _, pred = model(x, edge_index, lap, d_inv).max(dim=1)
# correct = int(pred[test_mask].eq(data.y[test_mask]).sum() .item())
# acc = correct / int(test_mask.sum())
# print(acc)

