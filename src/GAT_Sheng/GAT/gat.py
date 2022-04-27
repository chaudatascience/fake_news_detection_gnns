## paper Graph Attention Networks - https://arxiv.org/abs/1710.10903
import enum

import torch
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_max_pool


class GAT(nn.Module):
    def __init__(self, node_dim: int, out_dim: int, num_heads: int, 
                 is_final_layer: bool, dropout: float):
        """
        :param node_dim: F, the number of features in each node
        :param out_dim: F', new node feature dimension
        :param num_heads: K, number of heads in multi-head attention
        :param is_final_layer: True if the final layer, False otherwise
        :param dropout: dropout prob for dropout layer
        """
        super().__init__()
        self.node_dim = node_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_classes = 2
        self.W = nn.Linear(node_dim, out_dim * num_heads)  # Shared linear transformations

        self.is_final_layer = is_final_layer  # used for choosing equation (5) (if False) or equation (6) (if True)

        ## Vector `a` in equation 3. Instead of using a vector `a` with a shape of 2F',
        # we split `a` into 2 halves: `a_left` and `a_right`. This is only for implementation purpose
        # i.e., easier to stack vectors to use matrix multiplication.
        self.a_left = nn.Parameter(torch.Tensor(1, num_heads, out_dim))  # left part of vector `a`
        self.a_right = nn.Parameter(torch.Tensor(1, num_heads, out_dim))  # the right part

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(dropout)
        
        self.linear = nn.Linear(self.num_heads*self.out_dim, self.num_classes)

        self._init_params()  # initialize some parameters

    def _init_params(self):
        nn.init.xavier_uniform_(self.a_left)
        nn.init.xavier_uniform_(self.a_right)
        nn.init.xavier_uniform_(self.W.weight)


    def _compute_attention(self, x, mask):
        """
         Compute attention matrix which contains alpha_ij as described in equation (3), page 3
         :param e: energy matrix containing e_ij in equation (1), shape of (num_heads K, num_nodes N, num_nodes N)
         :param mask: used to mask out the nodes that are not in the neighborhood, shape (num_nodes N, num_nodes N)
         :return: attention matrix `attention`, shape of (num_heads K, num_nodes N, num_nodes N)
         """
        ## TODO 2: get attention matrix a
        
        x_proj = self.W(x).view(-1, self.num_heads, self.out_dim)
        x_proj = self.dropout(x_proj)  # in the official GAT imp they did dropout here as well
        score_source = (x_proj*self.a_left).sum(dim=-1)  ## (N, num_heads, 1)
        score_target = (x_proj*self.a_right).sum(dim=-1)
        
        ## to simplify calculation
        ## score[i, j] = score_source[i]+score_target[j]
        score = score_source.unsqueeze(1)+score_target.unsqueeze(0)
        score = self.leaky_relu(score)
        score = (score-score.mean())/score.std()  ## numerical stability
        score = torch.exp(score)
        mask2 = mask.reshape(len(mask), len(mask), -1)
        denominator = (score*mask2).sum(1, keepdim=True)  ## append mask with num_heads dim, sum over dim 1

        ## aij: (score_source[i]+score_target[j])/(score_source[i]+score_source[i1]+score_source[i]+score_source[i2])
        aij = score/denominator  ## it can be greater than 1 as aij might be greater than aik+ail+aim
        aij = aij*mask2  ## keep coefficients where j is neibour to Ni
        # aij = self.dropout(aij)
        aij = torch.permute(aij, (2, 0, 1))
        
        return aij

    def _compute_final_node_features(self, attention, x):
        """
        Compute final output features `h_prime` in equation (4)
        :param x: input, shape (num_nodes N, node_dim F)
        :param attention: attention matrix in equation (3), shape (num_heads K, num_nodes N, num_nodes N)
        :return: (num_nodes N, num_heads K, out_dim F')
        """
        x_encoded = self.W(x).reshape(-1, self.num_heads, self.out_dim)  # shape: (N, K, F')
        x_encoded = self.dropout(x_encoded)  # (N, K, F')
        # note: we can also cache x_encoded in `_compute_energy()` to avoid re-calculating it here.

        h_prime = torch.einsum("knm,mkf->nkf", [attention, x_encoded])  # shape (num_nodes N, num_heads K, out_dim F')
        h_prime = self.elu(h_prime)
        return h_prime

    def _concat_multi_head_features(self, node_features):
        """
        concatenate features from all attention heads, equation (5)
        :param node_features: updated node features from equation (4), shape (num_nodes N, num_heads K, out_dim F')
        :return: output. We just need to reshape to 2-D: (num_nodes N, num_heads K * out_dim F')
        """
        ## TODO 3: use self.elu, then reshape to return the ouput
        node_features = node_features.reshape(node_features.shape[0], -1)
        return node_features

    def _average_multi_head_features(self, node_features):
        """
        For the final layer, we employ averaging as described in equation (6)
        :param node_features: updated node features from equation (4), shape  (num_nodes N, num_heads K, out_dim F')
        :return: (num_nodes N, out_dim F'), for the last layer, we use F'=C (#classes)
        """
        output = torch.mean(node_features, dim=1)  # No need softmax() here as it will be used in cross entropy loss
        return output

    def forward(self, data):
        """
        The flow of GAT network, following the description in the paper
        :param data: tuple (input, mask), shape of (num_nodes N, node_dim F) and (num_nodes N, num_nodes N) respectively
        :return: output, mask
        """
        ## unpack data
        # x: node features, shape (num_nodes N, node_dim F)
        # mask: bool 2D-tensor,  mask = adjacency_matrix + identity_matrix; shape (num_nodes, num_nodes)
        x, mask, batch = data

        ## apply dropout to the input
        x = self.dropout(x)

        ## compute energy matrix `e` containing e_ij in the equation (1), page 3
        ## Sheng: I don't think paper use e
        # e = self._compute_energy(x)  # shape of e: (num_heads K, num_nodes N, num_nodes N)

        ## compute attention matrix `attention` containing alpha_ij in the equation (3)
        attention = self._compute_attention(x, mask)   # shape (num_heads K, num_nodes N, num_nodes N)
        
        ## compute final output features `h_prime` in equation (4)
        ## you also need mask to do aggregate over node
        final_features = self._compute_final_node_features(attention, x)  # (num_nodes N, num_heads K, out_dim F')

        # collect features from multi heads
        if not self.is_final_layer: # # intermediate layer  
            output = self._concat_multi_head_features(final_features)  #eq (5), (num_nodes N, num_heads K * out_dim F')
        else:  ## last layer
            output = self._average_multi_head_features(final_features)  # eq (6), (num_nodes N, out_dim F')
            output = global_max_pool(output, batch)  ## final classification
            
        ## in addition to output, we return mask for the next layer
        return output, mask, batch  
