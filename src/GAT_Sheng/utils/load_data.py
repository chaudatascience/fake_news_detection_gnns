#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 16:34:49 2022

download and load dataset Politifact and Gossipcop via PyG

@author: hs2015
"""
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F

import argparse

from torch_geometric.datasets import UPFD  ## dataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToUndirected  ## transform
import torch_geometric.nn as nn  

## download all data
if False:
    for name in ['politifact', 'gossipcop']:
        for feature in ['profile', 'spacy', 'bert', 'content']:
            dataset = UPFD(root='./data', name=name, feature=feature)


## load data
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='politifact', 
                    choices=['politifact', 'gossipcop'])
parser.add_argument('--feature', default='spacy', 
                    choices=['profile', 'spacy', 'bert', 'content'])
parser.add_argument('--model', default='GCN', 
                    choices=['GCN', 'GAT', 'SAGE'])
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()
# args = {'dataset':'politifact',  ## politifact, gossipcop
#         'feature':'bert',  ## profile, spacy, bert, content
#         'batch_size':128, 
#         'model':'GCN'}  ## GCN, GAT, SAGE
# for key in args:
#     setattr(args, key, args[key])

train_dataset = UPFD(root='./data', name=args.dataset, feature=args.feature, split='train', transform=ToUndirected())
val_dataset = UPFD(root='./data', name=args.dataset, feature=args.feature, split='val', transform=ToUndirected())
test_dataset = UPFD(root='./data', name=args.dataset, feature=args.feature, split='test', transform=ToUndirected())

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

## check data shape
if True:
    data = next(iter(train_loader))
    print(dir(data))
    print(data.num_graphs)
    from torch_scatter import scatter_mean
    x = scatter_mean(data.x, data.batch, dim=0)
    print(data.x.shape, x.shape)

'''
demo network and train process
'''
class Net(torch.nn.Module):
    def __init__(self, model, in_channels, hidden_channels, out_channels, concat=False):
        super().__init__()
        self.concat = concat
        self.conv1 = getattr(nn, model+'Conv')(in_channels, hidden_channels)
        if self.concat:
            self.lin0 = torch.nn.Linear(in_channels, hidden_channels)
            self.lin1 = torch.nn.Linear(2*hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index).relu()
        h = nn.global_max_pool(h, batch)
        
        if self.concat:
            ## get the root node feature of each graph:
            root = (batch[1:]-batch[:-1]).nonzero(as_tuple=False).view(-1)
            root = torch.cat([root.new_zeros(1), root+1], dim=0)
            news = x[root]
            
            news = self.lin0(news).relu()
            h = self.lin1(torch.cat([news, h], dim=-1)).relu()
        
        h = self.lin2(h)
        return h.log_softmax(dim=-1)

model = Net(args.model, train_dataset.num_features, 128, train_dataset.num_classes, concat=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)

def train():
    model.train()
    loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss_ = F.nll_loss(out, data.y)
        loss_.backward()
        optimizer.step()
        loss += float(loss_)*data.num_graphs
    return loss/len(train_loader.dataset)

loss = train()