import torch
from torch import nn
from typing import List

from gat import GAT


class GATNet(nn.Module):
    def __init__(self, node_dim: int, num_layers: int, layer_dims: List[int], num_heads_list: List[int]):
        super().__init__()

        assert num_layers == len(layer_dims) == len(num_heads_list), "Check on the dimensions!"
        layer_dims = [node_dim] + layer_dims
        in_out_num_heads = zip(layer_dims[:-1], layer_dims[1:], num_heads_list)
        gat_layers = [
            GAT(in_dim, out_dim, num_heads, True if i == num_layers else False)
            for i, in_dim, out_dim, num_heads in enumerate(in_out_num_heads, 1)
        ]
        self.gat_net = nn.Sequential(gat_layers)
