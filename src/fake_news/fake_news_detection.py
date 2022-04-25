import argparse
import time
from typing import List

import torch
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader

from src.fake_news.fake_news_net import FakeNewsNet
from src.utils.fake_news_utils import train, test, get_device


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="politifact",
                        help="name of the dataset, either 'gossipcop' or 'politifact' ")
    parser.add_argument('--cuda', type=int, default=0,
                        help="choose x for 'cuda:x', default using cuda 0")
    parser.add_argument('--batch', type=int, default=128,
                        help="batch size for training")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help="weight decay for lr")
    parser.add_argument('--eval', action='store_true',
                        help="evaluation mode")
    parser.add_argument('--epochs', type=int, default=300,
                        help="training epochs")
    parser.add_argument('--pooling', type=str, default="global_max_pool",
                        help="global_mean_pool or global_max_pool")
    parser.add_argument('--gat_layer', type=str, default="GATConv",
                        help="one of 'GATConv', 'GATv2Conv', 'SuperGATConv'")
    parser.add_argument('--hid_dims', type=List[int], default=[128,128],
                        help="hidden dimensions for GATs")
    parser.add_argument('--num_heads', type=int, default=1,
                        help="num attention heads for each GAT layer")
    parser.add_argument('--save_name', type=str, default=None,
                        help="name for saved model")
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    config = parse_config()

    train_data = UPFD(root=".", name=config.dataset, feature="content", split="train")
    val_data = UPFD(root=".", name=config.dataset, feature="content", split="val")
    test_data = UPFD(root=".", name=config.dataset, feature="content", split="test")
    print(f"The number of train dataset: {len(train_data):,}")
    print(f"The number of val dataset: {len(val_data):,}")
    print(f"The number of test dataset: {len(test_data):,}")

    train_loader = DataLoader(train_data, batch_size=config.batch, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=config.batch, shuffle=False)

    device = get_device(f"cuda:{config.cuda}")
    print(f"using {device}...")
    model = FakeNewsNet(config.gat_layer, config.pooling, train_data.num_features, config.hid_dims, 1, config.num_heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fnc = torch.nn.BCELoss()

    start = time.time()
    for epoch in range(config.epochs):
        train_loss = train(model, train_loader, loss_fnc, device, optimizer)
        val_loss, val_acc, val_f1 = test(model, val_loader, loss_fnc, device)
        test_loss, test_acc, test_f1 = test(model, test_loader, loss_fnc, device)
        running_time = (time.time() - start) /60
        print(f"Epoch: {epoch+1} | time: {running_time:.2f}[m] | train_loss: {train_loss: .2f} | "
              f"val_loss: {val_loss:.2f} | val_acc: {val_acc:.2f} | val_f1: {val_f1:.2f} | "
                f"test_loss: {test_loss:.2f} | test_acc: {test_acc:.2f} | test_f1: {test_f1:.2f}")


