#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 14:47:26 2022

@author: hs2015
"""
import argparse
import time, sys

import torch
from torch_geometric.datasets import UPFD  ## dataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToUndirected  ## transform
from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt

from GAT.gat_net import GATNet
from GAT.gat_utils import *

def train(args):
    ## load data
    dataset, loader, features, labels, masks = {}, {}, {}, {}, {}
    for split in ['train', 'val', 'test']:
        dataset[split] = UPFD(root='./data', name=args['dataset'], feature=args['feature'], 
                              split=split, transform=ToUndirected())
        if 'data' in dataset[split].__dict__.keys():
            for key in ['x', 'y', 'edge_index']:
                setattr(dataset[split], key, getattr(dataset[split].data, key))
            setattr(dataset[split], 'num_nodes', len(dataset[split].x))  ## need it?
        features[split] = dataset[split].x
        labels[split] = dataset[split].y
        masks[split] = convert_edge_list_to_mask(dataset[split].edge_index.T.tolist(), dataset[split].num_nodes)
        loader[split] = DataLoader(dataset[split], batch_size=128, shuffle=True)
    ## set up model
    num_classes = dataset['train'].num_classes
    node_dim = dataset['train'].num_features
    gat_config = {
                    'node_dim': node_dim,
                    'num_layers': 2,
                    'layer_dims': [64, num_classes],
                    'num_heads_list': [8, 8],
                    "dropout": 0.2
                }
    gat_net = GATNet(**gat_config)
    print(f"#params in GAT NET: {count_parameters(gat_net):,}")
    analyze_state_dict_shapes_and_names(gat_net)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(gat_net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    
    ## train model
    time_start = time.time()
    for epoch in range(args["num_epochs"]):
        # Training loop
        print('epoch', epoch)
        gat_net.train()
        
        for data in loader['train']:
            feature, label, edge_index, batch = data.x, data.y, data.edge_index, data.batch
            mask = convert_edge_list_to_mask(edge_index.T.tolist(), data.num_nodes)
            score = gat_net((feature, mask, batch))
            # score = gat_net((features['train'], masks['train']))
            loss = loss_fn(score, label)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            train_pred = torch.argmax(score, dim=-1)
            train_acc = torch.sum(torch.eq(train_pred, label).float()).item()/len(label)
        # Valid loop
        with torch.no_grad():
            gat_net.eval()
            for data in loader['val']:
                feature, label, edge_index, batch = data.x, data.y, data.edge_index, data.batch
                mask = convert_edge_list_to_mask(edge_index.T.tolist(), data.num_nodes)
                score = gat_net((feature, mask, batch))
                val_pred = torch.argmax(score, dim=-1)
                val_acc = torch.sum(torch.eq(val_pred, label).float()).item()/len(label)

        print(
            f"epoch: {epoch + 1} | time elapsed = {(time.time() - time_start): .2f}[s] | train_loss:{loss: .2f} | train_acc:{train_acc: .2f} | val_acc:{val_acc: .2f}")

    ## Test on test set
    gat_net.eval()
    for data in loader['test']:
        feature, label, edge_index, batch = data.x, data.y, data.edge_index, data.batch
        mask = convert_edge_list_to_mask(edge_index.T.tolist(), data.num_nodes)
        score = gat_net((feature, mask, batch))
        test_pred = torch.argmax(score, dim=-1)
        test_acc = torch.sum(torch.eq(test_pred, label).float()).item()/len(label)
        print(f"test_acc:{test_acc: .2f}")

    # print(f"accuracy_reported_in_the_paper: {data_config['accuracy_reported_in_the_paper']}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='politifact', 
                        choices=['politifact', 'gossipcop'])
    parser.add_argument('--feature', default='spacy', 
                        choices=['profile', 'spacy', 'bert', 'content'])
    parser.add_argument('--model', default='GCN', 
                        choices=['GCN', 'GAT', 'SAGE'])
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    
    args = {'dataset':'politifact',  ## politifact, gossipcop
            'feature':'bert',  ## profile, spacy, bert, content
            'model':'GCN', ## GCN, GAT, SAGE
            'batch_size':128, 
            'num_epochs':1000, 
            'lr':5e-3, 
            'weight_decay':1e-3
            }  
    # for key in args:
    #     setattr(args, key, args[key])
    
    train(args)