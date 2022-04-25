from typing import List

import torch
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.nn import GATConv, GATv2Conv, SuperGATConv
from torch.nn import Linear, ModuleList, Sequential, ReLU


class FakeNewsNet(torch.nn.Module):
    def __init__(self, gnn_layer: str, pooling: str, in_dim: int, hidden_dims: List[int], out_dim: int, num_heads: List[int], dropout: int =0):
        super().__init__()

        if gnn_layer == "GATConv":
            GNN = GATConv
        elif gnn_layer == "GATv2Conv":
            GNN = GATv2Conv
        elif gnn_layer == "SuperGATConv":
            GNN = SuperGATConv
        else:
            raise ValueError(f"{gnn_layer} is not valid! expected one of 'GATConv', 'GATv2Conv', 'SuperGATConv'")

        if pooling == "global_max_pool":
            self.pooling = global_max_pool
        elif pooling == "global_mean_pool":
            self.pooling = global_mean_pool
        else:
            raise ValueError(f"{pooling} is not valid! expected 'global_mean_pool' or 'global_max_pool'")

        # Graph Attention Networks
        gat_dims = list(zip([in_dim] + hidden_dims, hidden_dims[:-1], [num_heads]*(len(hidden_dims)-1)))
        gat_layers = [GNN(in_d, out_d, heads, dropout=dropout) for in_d, out_d, heads in gat_dims]
        self.gats = ModuleList(gat_layers)

        # Readout
        self.lin_news = Linear(in_dim, hidden_dims[-1])
        self.lin0 = Linear(hidden_dims[-1], hidden_dims[-1])
        self.lin1 = Linear(2 * hidden_dims[-1], out_dim)

    def forward(self, x, edge_index, batch):
        # Message passing using GATs
        h = x
        for gat in self.gats:
            h = gat(h, edge_index).relu()

        # Pooling
        h = self.pooling(h, batch)

        # Readout
        h = self.lin0(h).relu()

        # get root node for each graph
        root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
        root = torch.cat([root.new_zeros(1), root + 1], dim=0) # list of indices
        news = x[root]
        news = self.lin_news(news).relu()

        # Concat raw word2vec embeddings of news and readout from the graph
        out = self.lin1(torch.cat([h, news], dim=-1))
        return torch.sigmoid(out)