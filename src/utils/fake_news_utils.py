import copy
import os
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid


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

    acc, f1_score = metrics(all_predictions, all_labels)
    return total_loss / len(test_loader.dataset), acc, f1_score


def metrics(pred, labels):
    pred = torch.round(torch.cat(pred))
    labels = torch.cat(labels)
    acc = accuracy_score(pred, labels) * 100
    f1 = f1_score(pred, labels) * 100
    return acc, f1


def get_device(device: str = "cuda:auto"):
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
        if "auto" in device:
            device = f'cuda:{get_most_available_gpu()}' if torch.cuda.is_available() else 'cpu'
    return device


def get_hyperpram_grid_configs(config: Dict) -> List[Dict]:
    """
    Return all possible combinations of parameters
    :param config: the config read from YAML file
    :return: return a list of configs:
    """

    PARSE_BY = "--PARSE_BY-->"

    def get_params(config: Dict) -> Dict:
        """
        return params we need to tune and their values
        :param config:
        :return:
        """
        hyperparams = dict()
        for config_type, config_value in config.items():
            if isinstance(config_value, dict):
                for k1, v1 in config_value.items():
                    if isinstance(v1, dict):
                        for k2, v2 in v1.items():
                            if isinstance(v2, list):
                                hyperparams[f"{config_type}{PARSE_BY}{k1}{PARSE_BY}{k2}"] = v2
                    elif isinstance(v1, list):
                        hyperparams[f"{config_type}{PARSE_BY}{k1}"] = v1
            elif isinstance(config_value, list):
                hyperparams[f"{config_type}"] = config_value
        return hyperparams

    def update_dict(config: Dict, params: Dict) -> Dict:
        """
        Update a `config` dict using `params`
        :param config:
        :param params:
        :return:
        """
        for param_k, param_v in params.items():
            key_list = param_k.split(PARSE_BY)
            if len(key_list) == 1:
                config[key_list[0]] = param_v
            elif len(key_list) == 2:
                config[key_list[0]][key_list[1]] = param_v
            elif len(key_list) == 3:
                config[key_list[0]][key_list[1]][key_list[2]] = param_v
            else:
                raise ValueError("Should be no more than 3 nested layers!")
        return config

    params = get_params(config)
    tuning_params_grid = ParameterGrid(params)
    config_list = []
    for i, params in enumerate(tuning_params_grid):
        new_config = update_dict(copy.deepcopy(config), params)
        config_list.append(new_config)
    print(f"Hypergrid: Create {len(config_list)} different configs!")
    return config_list


def update_config(config, main_config):
    if main_config["debug"]:
        config["train"]["batch_size"] = 8
        config["dataset"]["force_recreate"] = True
        config["dataset"]["write_to_file"] = False
        config["debug"] = True
        config["train"]["num_epochs"] = 1
    else:
        config["dataset"]["force_recreate"] = main_config["force_recreate"]
        config["dataset"]["write_to_file"] = main_config["write_to_file"]

    config["dataset"]["pd_output"] = main_config["pd_output"]
    config["device"] = main_config["device"]
    return config


def plot_graph(g: nx.Graph, node_labels: Dict = None, figsize: Tuple = (25, 12), options: Dict = None,
               file_name="_tmp"):
    if options is None:
        options = {
            'node_size': 500,
            'width': 1,
            'node_color': 'gray',
        }
    plt.figure(figsize=figsize)
    if node_labels:
        nx.draw(g, labels=node_labels, with_labels=True, **options)
    else:
        nx.draw(g, **options, with_labels=True)
    plt.savefig(f'output/{file_name}_pic.png')
    plt.show()
    return g


