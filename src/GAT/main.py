import time

import pandas as pd
from torch import nn

from src.GAT.gat_net import GATNet
from src.GAT.gat_ultils import *


def train(config):
    gat_config = config["gat_net"]
    data_config = config["dataset"]

    data = get_dataset(data_config["dataset_name"])
    node_features, node_labels = data.x, data.y

    mask = convert_edge_list_to_mask(data.edge_index.T.tolist(), data.num_nodes)

    gat_net = GATNet(**gat_config)

    print(f"#params in GAT NET: {count_parameters(gat_net):,}")
    analyze_state_dict_shapes_and_names(gat_net)

    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(gat_net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    train_indices = torch.arange(data_config["train_range"][0], data_config["train_range"][1], dtype=torch.long)
    val_indices = torch.arange(data_config["val_range"][0], data_config["val_range"][1], dtype=torch.long)
    test_indices = torch.arange(data_config["test_range"][0], data_config["test_range"][1], dtype=torch.long)

    train_labels = node_labels.index_select(0, train_indices)
    val_labels = node_labels.index_select(0, val_indices)
    test_labels = node_labels.index_select(0, test_indices)

    # node_features size = (num_nodes N, node_dim F), adj_matrix size = (num_nodes N, num_nodes N)
    graph_data = (node_features, mask)
    time_start = time.time()

    train_loss_list, train_acc_list = [], []
    val_loss_list, val_acc_list = [], []
    test_acc_list = []

    for epoch in range(config["num_epochs"]):
        # Training loop
        gat_net.train()

        scores = gat_net(graph_data)
        train_scores = scores.index_select(0, train_indices)
        loss = loss_fn(train_scores, train_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_pred = torch.argmax(train_scores, dim=-1)
        train_acc = 100 * torch.sum(torch.eq(train_pred, train_labels).float()).item() / len(train_labels)

        # Valid loop
        with torch.no_grad():
            gat_net.eval()
            scores = gat_net(graph_data)
            val_scores = scores.index_select(0, val_indices)
            val_loss = loss_fn(val_scores, val_labels).item()
            val_pred = torch.argmax(val_scores, dim=-1)
            val_acc = 100 * torch.sum(torch.eq(val_pred, val_labels).float()).item() / len(val_labels)

        print(
            f"epoch: {epoch + 1} | time elapsed = {(time.time() - time_start) // 60: .2f}[m] | train_loss:{loss: .2f} | train_acc:{train_acc: .2f} | val_acc:{val_acc: .2f}")

        train_loss_list.append(loss.item())
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

        ## Test on test set
        gat_net.eval()
        test_scores = scores.index_select(0, test_indices)
        test_pred = torch.argmax(test_scores, dim=-1)
        test_acc = 100 * torch.sum(torch.eq(test_pred, test_labels).float()).item() / len(test_labels)
        print(f"test_acc:{test_acc: .2f}")
        test_acc_list.append(test_acc)

    print(f"accuracy_reported_in_the_paper: {data_config['accuracy_reported_in_the_paper']}")
    store = {"train_loss_list": train_loss_list,
             "train_acc_list": train_acc_list,
             "val_loss_list": val_loss_list,
             "val_acc_list": val_acc_list,
             "test_acc_list": test_acc_list
             }

    df = pd.DataFrame(store)
    df.to_csv("logs/gat_cora.csv")


def train_cora():
    cora_num_classes = 7
    cora_node_dim = 1433

    cora_config = {
        "dataset":
            {
                "dataset_name": "Cora",
                "accuracy_reported_in_the_paper": 83.0,  # our accuracy should be at least 81%
                "train_range": [0, 140],  # train set contains nodes whose indices from 0 to 140
                "val_range": [140, 140 + 500],  # 500 val nodes: index from 140 to 640
                "test_range": [1708, 1708 + 1000]
            },
        "num_epochs": 2000,
        "weight_decay": 5e-4,
        "lr": 5e-3,
        "gat_net":
            {
                'node_dim': cora_node_dim,
                'num_layers': 2,
                'layer_dims': [8, cora_num_classes],
                'num_heads_list': [8, 1],
                "dropout": 0.6
            }
    }

    train(cora_config)


def train_citeseer():
    ##  Addition to Cora, we need to test on another dataset (e.g., Citeseer)
    # to prove our implementation is correct.

    ## TODO 4: Should be the same as `train_cora()` function, only change some arguments. Dataset info: https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
    raise NotImplementedError("Not implemented yet!")


if __name__ == '__main__':
    train_cora()
