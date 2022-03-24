import torch
from torch import nn
from typing import List

from gat import GAT


class GATNet(nn.Module):
    def __init__(self, node_dim: int, num_layers: int, layer_dims: List[int], num_heads_list: List[int], dropout: int):
        super().__init__()

        assert num_layers == len(layer_dims) == len(num_heads_list), "Check on the dimensions!"
        layer_dims = [node_dim] + layer_dims
        num_heads_list = [1] + num_heads_list
        gat_layers = []
        for i in range(len(layer_dims)-1):
            layer = GAT(layer_dims[i] * num_heads_list[i], layer_dims[i+1], num_heads= num_heads_list[i+1],
                        is_final_layer=True if i == len(layer_dims)-2 else False, dropout=dropout)
            gat_layers.append(layer)
        self.gat_net = nn.Sequential(*gat_layers)

    def forward(self, data):
        output, _ = self.gat_net(data)
        return output
