import os
import time
from collections import namedtuple

from typing import Dict
import torch
import yaml
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToUndirected

from src.GAT.gat_ultils import count_parameters
from src.custom_logs import ExperimentLog
from src.fake_news.argparse_config import parse_argparse_config
from src.fake_news.fake_news_net import FakeNewsNet
from src.utils.fake_news_utils import train, test, get_device, get_hyperpram_grid_configs
from src import custom_logs


def train_model(config: Dict, logger: ExperimentLog):
    train_data = UPFD(root=".", name=config.dataset, feature=config.feature, split="train", transform=ToUndirected())
    val_data = UPFD(root=".", name=config.dataset, feature=config.feature, split="val", transform=ToUndirected())
    test_data = UPFD(root=".", name=config.dataset, feature=config.feature, split="test", transform=ToUndirected())
    print(f"The number of train dataset: {len(train_data):,}")
    print(f"The number of val dataset: {len(val_data):,}")
    print(f"The number of test dataset: {len(test_data):,}")

    train_loader = DataLoader(train_data, batch_size=config.batch, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=config.batch, shuffle=False)

    device = get_device(f"cuda:{config.cuda}")
    print(f"using {device}...")
    model = FakeNewsNet(config.gat_layer, config.pooling, train_data.num_features, config.hid_dims, 1, config.num_heads,
                        config.readout_dim, config.news_dim, config.dropout).to(device)

    count_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fnc = torch.nn.BCELoss()

    best_val_loss = float("inf")
    _final_test_acc = -float("inf")
    _final_test_f1 = -float("inf")
    patience = 0

    train_loss_list = []
    val_acc_list, val_f1_list, val_loss_list = [], [], []

    start = time.time()
    for epoch in range(config.epochs):
        train_loss = train(model, train_loader, loss_fnc, device, optimizer)
        val_loss, val_acc, val_f1 = test(model, val_loader, loss_fnc, device)
        _, test_acc, test_f1 = test(model, test_loader, loss_fnc, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _final_test_acc = test_acc
            _final_test_f1 = test_f1
        else:
            patience += 1

        if patience > config.early_stopping:
            logger.info(f"early stopping, best epoch:{epoch - patience}")
            logger.info(f"final_test_acc: {_final_test_acc}; final_test_f1: {_final_test_f1}")
            break

        running_time = (time.time() - start) / 60
        logger.info(f"Epoch: {epoch + 1} | time: {running_time:.2f}[m] | train_loss: {train_loss: .2f} | "
              f"val_loss: {val_loss:.2f} | val_acc: {val_acc:.2f} | val_f1: {val_f1:.2f}")

        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)
        val_f1_list.append(val_f1)
        val_loss_list.append(val_loss)

    logger.time_mins = running_time
    logger.running_epochs = epoch-patience
    logger.final_test_acc = _final_test_acc
    logger.final_test_f1 = _final_test_f1
    logger.train_loss = train_loss_list
    logger.val_f1 = val_f1_list
    logger.val_acc = val_acc_list
    logger.val_loss = val_loss_list

    logger.log_all()
    return

if __name__ == '__main__':
    # config = parse_argparse_config()

    with open("../../configs/gossipcop.yml", "r") as f:
        main_config = yaml.safe_load(f)

    config_list = get_hyperpram_grid_configs(main_config)

    print(f"there are total {len(config_list)} config files!")
    for i, config in enumerate(config_list):
        print(f"running config {i+1}...")
        logger = custom_logs.ExperimentLog({"logger_name": "%Y-%m-%d--%Hh%Mm%Ss.%f",
                                            "logger_path": "logs/{dataset_name}"},
                                           dataset=config["dataset"])
        logger.info(config)
        logger.config = config

        MyConfig = namedtuple('MyConfig', config)
        config = MyConfig(**config)

        train_model(config, logger)
