import torch
from torch_geometric.datasets import Planetoid
from os.path import join as os_join
from typing import List


def count_parameters(model):
    """
    Count how many parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyze_state_dict_shapes_and_names(model):
    print(model.state_dict().keys())
    for name, param in model.named_parameters():
        print(name, param.shape)
        if not param.requires_grad:
            raise Exception('Some params are not trainable!')


def convert_edge_list_to_mask(edges: List[List], num_nodes: int):
    """
    convert an edge list (list of pairs of edges) into an adjacency matrix, then add Identity matrix to generate a mask
    :param edges:list of pairs [source_node, target_node]
    :param num_nodes: number of nodes in the graph
    :return: mask, shape (num_nodes, num_nodes)
    """
    adj_matrix = [[False] * num_nodes for _ in range(num_nodes)]
    for src, dest in edges:
        adj_matrix[src][dest] = True
    adj_matrix = torch.Tensor(adj_matrix)
    adj_matrix.fill_diagonal_(True)
    return adj_matrix.bool()


def get_dataset(name):
    """
    get datasets using pytorch-geometric.
    Reference: https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
    :param name: one of ["Cora", "Citeseer", "Pubmed"]
    :return: pytorch-geometric object
    """
    assert name in ["Cora", "Citeseer", "Pubmed"], f'expected one of ["Cora", "Citeseer", "Pubmed"], but received {name}'
    dataset = Planetoid(root=os_join('/tmp', name), name=name)
    data = dataset[0]

    return data
