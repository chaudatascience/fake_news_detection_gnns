import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_max_pool
from torch_geometric.transforms import ToUndirected
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='gossipcop',
                    choices=['politifact', 'gossipcop'])
parser.add_argument('--feature', type=str, default='spacy',
                    choices=['profile', 'spacy', 'bert', 'content'])
parser.add_argument('--model', type=str, default='GAT',
                    choices=['GCN', 'GAT', 'SAGE'])
args = parser.parse_args()

#path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'UPFD')
path = './'
train_dataset = UPFD(path, args.dataset, args.feature, 'train', ToUndirected())
val_dataset = UPFD(path, args.dataset, args.feature, 'val', ToUndirected())
test_dataset = UPFD(path, args.dataset, args.feature, 'test', ToUndirected())

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


class Net(torch.nn.Module):
    def __init__(self, model, in_channels, hidden_channels, out_channels,
                 concat=False):
        super().__init__()
        self.concat = concat

        if model == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
        elif model == 'SAGE':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
        elif model == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels)

        if self.concat:
            self.lin0 = Linear(in_channels, hidden_channels)
            self.lin1 = Linear(2 * hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index).relu()
        h = global_max_pool(h, batch)

        if self.concat:
            # Get the root node (tweet) features of each graph:
            root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
            root = torch.cat([root.new_zeros(1), root + 1], dim=0)
            news = x[root]

            news = self.lin0(news).relu()
            h = self.lin1(torch.cat([news, h], dim=-1)).relu()

        h = self.lin2(h)
        return h.log_softmax(dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(args.model, train_dataset.num_features, 128,
            train_dataset.num_classes, concat=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_correct = total_examples = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch).argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
        total_examples += data.num_graphs

    return total_correct / total_examples

@torch.no_grad()
def test_loss_function(passed_data):
    model.eval()
    total_loss = 0
    for data in passed_data:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        #loss.backward() # this may not be required
        optimizer.step()
        total_loss += float(loss) * data.num_graphs

    return total_loss / len(passed_data.dataset)

record_metrics = pd.DataFrame(columns = ['Epoch', 'train_acc', 'train_loss', 'val_acc','test_acc'])
for epoch in range(1, 201):
    loss = train()
    train_acc = test(train_loader)
    val_acc = test(val_loader)
    test_acc = test(test_loader)
    val_loss, test_loss = test_loss_function(val_loader), test_loss_function(test_loader)
    record_metrics = record_metrics.append({'Epoch': f'{epoch:03d}', 'train_acc': f'{train_acc:.4f}', 'train_loss': f'{loss:.4f}','val_acc': f'{val_acc:.4f}', 'val_loss':f'{val_loss:.4f}','test_acc': f'{test_acc:.4f}','test_loss': f'{test_loss:.4f}'}, ignore_index=True)
    print(f'Epoch: {epoch:03d}, train_acc: {train_acc:.4f}, train_loss: {loss:.4f},val_acc: {val_acc:.4f}, val_loss: {val_loss:.4f}, test_acc: {test_acc:.4f}, test_loss: {test_loss:.4f}'
          f'test_acc: {test_acc:.4f}')

record_metrics.to_csv('gossipcop_spacy_GAT_results.csv', index=False)