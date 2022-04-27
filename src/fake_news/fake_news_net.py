from typing import List

import torch
import torch_geometric
from torch import nn
from torch.nn import Linear, ModuleList
from torch_geometric.nn import global_max_pool, global_mean_pool, GlobalAttention

from src.GAT.gat import GAT
from src.GAT.gat_net import GATNet
from src.GAT.gat_ultils import convert_edge_list_to_mask


class FakeNewsNet(torch.nn.Module):
    def __init__(self, gnn_layer: str, pooling: str, in_dim: int, hidden_dims: List[int], out_dim: int,
                 num_heads: int, readout_dim: int = None, news_dim: int = None, dropout: int = 0):
        super().__init__()

        assert gnn_layer in ["OurGATNet", "GATConv", "GATv2Conv", "SuperGATConv"]
        self.gnn_layer = gnn_layer

        if gnn_layer == "OurGATNet":
            GNN = GAT
        else:
            GNN = getattr(torch_geometric.nn, gnn_layer)

        if pooling == "global_max_pool":
            self.pooling = global_max_pool
        elif pooling == "global_mean_pool":
            self.pooling = global_mean_pool
        elif pooling == "global_attention":
            attention_readout = nn.Sequential(Linear(hidden_dims[-1] * num_heads, 1))
            self.pooling = GlobalAttention(attention_readout)
        elif pooling == "global_attention_with_relu":
            attention_readout = nn.Sequential(Linear(hidden_dims[-1] * num_heads, 1), nn.ReLU())
            self.pooling = GlobalAttention(attention_readout)
        elif pooling == "global_attention_with_relu_linear":
            attention_readout = nn.Sequential(Linear(hidden_dims[-1] * num_heads, 1), nn.ReLU())
            attention_linear = Linear(hidden_dims[-1] * num_heads, hidden_dims[-1] * num_heads)
            self.pooling = GlobalAttention(attention_readout, attention_linear)
        else:
            raise ValueError(f"{pooling} is not valid! expected 'global_mean_pool' or 'global_max_pool'")

        # Graph Attention Networks
        if gnn_layer == "OurGATNet":  # our implementation
            gat_config = {
                'node_dim': in_dim,
                'num_layers': len(hidden_dims),
                'layer_dims': hidden_dims,
                'num_heads_list': [num_heads] * len(hidden_dims),
                "dropout": 0.6
            }
            self.gats = GATNet(**gat_config)
        else:
            gat_dims = list(zip([in_dim] + [h * num_heads
                                            for h in hidden_dims[:-1]], hidden_dims, [num_heads] * (len(hidden_dims))))
            gat_layers = [GNN(in_d, out_d, heads, dropout=dropout) for in_d, out_d, heads in gat_dims]
            self.gats = ModuleList(gat_layers)

        # Readout
        self.linear_news = Linear(in_dim, news_dim)
        self.linear_readout = Linear(hidden_dims[-1] * num_heads, readout_dim)
        self.linear_concat = Linear(news_dim + readout_dim, out_dim)

    def forward(self, x, edge_index, batch):
        """
        :param x: input node features, shape (X, feature_dim), feature_dim=310 in case of "content" feature)
        :param edge_index: list of edges in the graphs
        :param batch: shape (X,), contains number from 0 -> batch_size-1 to indicate which node belongs to which graph in batch
        :return:
        """
        # Message passing using GATs
        if self.gnn_layer == "OurGATNet":
            edge_index_2 = [[s.item(), t.item()] for s, t in zip(edge_index[0], edge_index[1])]
            h = self.gats((x, convert_edge_list_to_mask(edge_index_2, num_nodes=len(x))))
        else:
            h = x
            for gat in self.gats:
                h = gat(h, edge_index).relu()

        # Pooling: reduce all nodes in a graph into 1 node
        h = self.pooling(h, batch)  ## (batch_size, hid_dim)

        # Readout
        h = self.linear_readout(h).relu()

        # get root node for each graph
        root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
        root = torch.cat([root.new_zeros(1), root + 1], dim=0)  # list of indices
        news = x[root]  # shape (batch_size, feature_dim)
        news = self.linear_news(news).relu()

        # Concat raw word2vec embeddings of news and readout from the graph
        out = self.linear_concat(torch.cat([h, news], dim=-1))
        return torch.sigmoid(out)
