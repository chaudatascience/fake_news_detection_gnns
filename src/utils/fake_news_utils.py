import os

import networkx as nx
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def convert_to_networkx(data, node_attrs=None, edge_attrs=None, to_undirected=False,
                        remove_self_loops=False):
    if to_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()
    G.add_nodes_from(range(data.num_nodes))
    node_attrs, edge_attrs = node_attrs or [], edge_attrs or []
    values = {}
    for key, item in data(*(node_attrs + edge_attrs)):
        if torch.is_tensor(item):
            values[key] = item.squeeze().tolist()
        else:
            values[key] = item
        if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
            values[key] = item[0]
    for i, (u, v) in enumerate(data.edge_index.t().tolist()):
        if to_undirected and v > u:
            continue
        if remove_self_loops and u == v:
            continue
        G.add_edge(u, v)
        for key in edge_attrs:
            G[u][v][key] = values[key][i]
    for key in node_attrs:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})
    return G


def train(model, train_loader, loss_fnc, device, optimizer):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = loss_fnc(torch.reshape(out, (-1,)), data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(model, test_loader, loss_fnc, device):
    total_loss = 0
    all_predictions = []
    all_labels = []
    model.eval()
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = loss_fnc(torch.reshape(out, (-1,)), data.y.float())
        total_loss += float(loss) * data.num_graphs
        all_predictions.append(torch.reshape(out, (-1,)))
        all_labels.append(data.y.float())

    # Calculate Metrics
    acc, f1_score = metrics(all_predictions, all_labels)

    return total_loss / len(test_loader.dataset), acc, f1_score


def metrics(pred, labels):
    pred = torch.round(torch.cat(pred))
    labels = torch.cat(labels)
    acc = accuracy_score(pred, labels)*100
    f1 = f1_score(pred, labels)*100
    return acc, f1


def get_device(device: str = None):
    def get_most_available_gpu():
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Total >tmp_total')
        memory_total = [int(x.split()[2]) for x in open('tmp_total', 'r').readlines()]

        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp_used')
        memory_used = [int(x.split()[2]) for x in open('tmp_used', 'r').readlines()]

        memory_available = [total - used for total, used in zip(memory_total, memory_used)]
        return np.argmax(memory_available)

    if device:
        if ("cuda" in device) and (not torch.cuda.is_available()):
            device = "cpu"
    else:
        device = f'cuda:{get_most_available_gpu()}' if torch.cuda.is_available() else 'cpu'

    return device
