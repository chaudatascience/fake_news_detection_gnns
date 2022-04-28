# A GNN-based model for Fake News Detection

# Setup

```
git clone git@github.com:chaudatascience/fake_news_detection_gnns.git
cd fake_news_detection_gnns
conda env create -f environment.yml
conda activate fake_news
```

# Usage

```
python -m src.fake_news.fake_news_detection --dataset gossipcop --epochs 300 
```
<br>

**Other arguments**
```
'--dataset', type=str, default="gossipcop",
                    help="name of the dataset, either 'gossipcop' or 'politifact' "
'--early_stopping', type=int, default=50,
                    help="stop training after `early_stopping` non-decreasing val loss epochs"
'--cuda', type=str, default="auto",
                    help="choose x for 'cuda:x', or using most available GPU by default"
'--batch', type=int, default=128,
                    help="batch size for training"
'--lr', type=float, default=0.001,
                    help="learning rate"
'--weight_decay', type=float, default=0.01,
                    help="weight decay for lr"
'--epochs', type=int, default=300,
                    help="training epochs"
'--dropout', type=float, default=0,
                    help="dropout"
'--pooling', type=str, default="global_max_pool",
                    help="one of [global_mean_pool, global_max_pool, global_attention, global_attention_with_relu, global_attention_with_relu_linear]"
'--gat_layer', type=str, default="GATConv",
                    help="one of ['OurGATNet', GATConv', 'GATv2Conv', 'SuperGATConv']"
'--hid_dims', type=List[int], default=[128, 128],
                    help="hidden dimensions for GATs"
'--news_dim', type=int, default=128,
                    help="dimensions for news"
'--readout_dim', type=int, default=128,
                    help="dimensions for graph readout"
'--num_heads', type=int, default=1,
                    help="num attention heads for each GAT layer"

'--feature', type=str, default='content',
                    help="feature type: [profile, spacy, bert, content],`content` means 300-d word2vec+10-d profile"
```

# Datasets
<img src="plots/data_stats.png" width="500">

<br>
<br>
<br>
<img src="plots/a_graph_in_Gossipcop_dataset.png" width="500" >
<br>
<br>
<br>
Train and val losses on **Gossipcop** dataset
<br>
<img src="plots/gossipcop_loss.png" width="600">

<br>
<br>
**Result table**
<br>
<img src="plots/res_table.png" width="436">


## References

Graph Attention Networks &ensp; [[ICLR 2018]](https://arxiv.org/abs/1710.10903)

A Generalization of Transformer Networks to Graphs &ensp; [[AAAI 2021]](https://arxiv.org/abs/2012.09699)

User Preference-aware Fake News Detection (UPFD) &ensp; [[SIGIR 2021]](https://arxiv.org/abs/2104.12259)

UPFD Code &ensp; [[Github]](https://github.com/safe-graph/GNN-FakeNews) 


