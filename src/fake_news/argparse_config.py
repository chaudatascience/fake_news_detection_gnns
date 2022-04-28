import argparse
from typing import List


def parse_argparse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="politifact",
                        help="name of the dataset, either 'gossipcop' or 'politifact' ")
    parser.add_argument('--early_stopping', type=int, default=50,
                        help="stop training after `early_stopping` non-decreasing val loss epochs")
    parser.add_argument('--cuda', type=str, default="auto",
                        help="choose x for 'cuda:x', or using most available GPU by default")
    parser.add_argument('--batch', type=int, default=128,
                        help="batch size for training")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help="weight decay for lr")
    parser.add_argument('--epochs', type=int, default=300,
                        help="training epochs")
    parser.add_argument('--dropout', type=float, default=0,
                        help="dropout")
    parser.add_argument('--pooling', type=str, default="global_max_pool",
                        help="one of [global_mean_pool, global_max_pool, global_attention, global_attention_with_relu, global_attention_with_relu_linear]")
    parser.add_argument('--gat_layer', type=str, default="GATConv",
                        help="one of ['OurGATNet', GATConv', 'GATv2Conv', 'SuperGATConv']")
    parser.add_argument('--hid_dims', type=List[int], default=[32],
                        help="hidden dimensions for GATs")
    parser.add_argument('--news_dim', type=int, default=64,
                        help="dimensions for news")
    parser.add_argument('--readout_dim', type=int, default=64,
                        help="dimensions for graph readout")
    parser.add_argument('--num_heads', type=int, default=1,
                        help="num attention heads for each GAT layer")
    parser.add_argument('--only_gat', type=bool, default=False,
                        help="Only use GAT (not use news features)")
    # https://github.com/safe-graph/GNN-FakeNews/issues/13
    parser.add_argument('--feature', type=str, default='content',
                        help="feature type: [profile, spacy, bert, content],`content` means 300-d word2vec+10-d profile")

    args = parser.parse_args()
    return args