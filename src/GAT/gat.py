import torch
from torch import nn


class GAT(nn.Module):
    def __init__(self, node_dim: int, out_dim: int, num_heads: int, is_final_layer: bool):
        """
        :param node_dim: F, the number of features in each node
        :param out_dim: F', new node feature dimension
        :param num_heads: K, number of heads in multi-head attention
        :param is_final_layer: True if the final layer, False otherwise
        """
        super().__init__()
        self.node_dim = node_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.W = nn.Linear(node_dim, out_dim * num_heads)  # Shared linear transformations

        self.is_final_layer = is_final_layer  # used for choosing equation (5) (if False) or equation (6) (if True)

        ## Attention `a` in equation 3. Instead of using a vector `a` with a shape of 2F',
        # we split `a` into 2 halves: `a_left` and `a_right`. This is only for implementation purpose
        # i.e., making use of matrix multiplication.
        self.a_left = nn.Parameter(torch.Tensor(1, num_heads, out_dim))  # left part of the attention mechanism
        self.a_right = nn.Parameter(torch.Tensor(1, num_heads, out_dim))  # the right part

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

        self._init_params()  # initialize some parameters

    def _init_params(self):
        nn.init.xavier_uniform_(self.a_left)
        nn.init.xavier_uniform_(self.a_right)

    def _compute_energy(self, x):
        """
        Compute matrix e containing e_ij in equation (1)
        :param x: input, size of (num_nodes N, node_dim F)
        :return: e: size of (num_heads K, num_nodes N, num_nodes N)
        """
        raise NotImplementedError("Need to be implemented!")

    def _compute_attention(self, e, mask):
        """
         Compute matrix a containing a_ij in equation (3)
         :param e: energy matrix containing e_ij in equation (1), size of (num_heads K, num_nodes N, num_nodes N)
         :param mask: binary matrix to mask out the nodes that are not neighbors, size (1, num_nodes N, num_nodes N)
         :return: attention weights, size of (num_heads K, num_nodes N, num_nodes N)
         """
        raise NotImplementedError("Need to be implemented!")

    def _compute_final_node_features(self, x, a):
        """
        Compute final output h_prime in equation (4)
        :param x: input, size (num_nodes N, node_dim F)
        :param a: attention matrix in equation (3), size (num_heads K, num_nodes N, num_nodes N)
        :return: (num_nodes N, num_heads K, out_dim F')
        """
        # to avoid re-calculating this, cache `h1` when computing energy (`_compute_energy()`)
        h1 = self.W(x).reshape(-1, self.out_dim, self.num_heads)  # size (num_nodes N, out_dim F', num_heads K)
        h_prime = torch.einsum("knn,nkf->nkf", [a, h1])  # size (num_nodes N, num_heads K, out_dim F')
        h_prime = self.sigmoid(h_prime)
        return h_prime

    def _concate_multi_head_attention(self, h_prime):
        """
        concatenate features from all attention heads, equation (5)
        :param h_prime: updated node features from equation (4), size (num_nodes N, num_heads K, out_dim F')
        :return: just need to reshape to 2-D: (num_nodes N, num_heads K * out_dim F')
        """
        h_prime = h_prime.reshape(-1, self.num_heads * self.out_dim)
        return h_prime

    def _averaging_final_multi_head_layer(self, h_prime):
        """
        For the final layer, we employ averaging as described in equation (6)
        :param h_prime: updated node features from equation (4), size (num_nodes N, num_heads K, out_dim F')
        :return: (num_nodes N, out_dim F')
        """
        h_prime = torch.mean(h_prime, dim=1)
        return h_prime

    def forward(self, x, mask):
        """
        Forward pass
        :param x: input, size of (num_nodes N, node_dim F)
        :param mask: inject graph structure into the attention mechanism, size of (num_nodes, num_nodes)
        :return:
        """
        ## compute matrix `e` containing e_ij in the equation (1)
        e = self._compute_energy(x)  # size (num_heads K, num_nodes N, num_nodes N)

        ## compute matrix `a` containing a_ij in the equation (3)
        a = self._compute_attention(e, mask)   # size (num_heads K, num_nodes N, num_nodes N)

        ## compute matrix `h_prime` in equation (4)
        h_prime = self._compute_final_node_features(x, a)  # (num_nodes N, num_heads K, out_dim F')

        ## collect features from multi heads
        if self.is_final_layer:
            h_prime = self._concate_multi_head_attention(h_prime)  # eq (5), (num_nodes N, num_heads K * out_dim F')
        else:
            h_prime = self._averaging_final_multi_head_layer(h_prime)  # eq (6), (num_nodes N, out_dim F')

        return h_prime






















