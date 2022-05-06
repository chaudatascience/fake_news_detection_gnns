import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
import pandas as pd

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False,
                             dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(dataset.num_features, dataset.num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    train_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    train_loss.backward()
    optimizer.step()
    return train_loss.item()


@torch.no_grad()
def test(data):
    model.eval()
    optimizer.zero_grad()
    out, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        acc = float((out[mask].argmax(-1) == data.y[mask]).sum() / mask.sum())
        accs.append(acc)
    val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
    accs.append(val_loss)
    test_loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask])
    accs.append(test_loss)
    optimizer.step()
    return accs


record_metrics = pd.DataFrame(columns = ['train_loss', 'train_acc', 'val_loss','val_acc', 'test_acc', 'test_loss','epoch'])
for epoch in range(1, 201):
    train_loss = train(data)
    train_acc, val_acc, test_acc, val_loss, test_loss = test(data)
    record_metrics = record_metrics.append({'train_loss' : f'{train_loss:.4f}', 'train_acc' : f'{train_acc:.4f}', 'val_loss' : f'{val_loss:.4f}', 'val_acc' : f'{val_acc:.4f}', 'test_acc' : f'{test_acc:.4f}', 'test_loss' : f'{test_loss:.4f}', 'epoch' : f'{epoch:03d}'}, ignore_index=True)
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')

record_metrics.to_csv(f'{dataset}_GAT_results.csv', index=True)