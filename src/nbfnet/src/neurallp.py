"""
All code here is adapted from TorchDrug

See here - https://github.com/DeepGraphLearning/torchdrug/
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add


class NeuralLogicProgramming(nn.Module):
    """
    Neural Logic Programming proposed in `Differentiable Learning of Logical Rules for Knowledge Base Reasoning`_.

    .. _Differentiable Learning of Logical Rules for Knowledge Base Reasoning:
        https://papers.nips.cc/paper/2017/file/0e55666a4ad822e0e34299df3591d979-Paper.pdf

    Parameters:
        num_relation (int): number of relations
        hidden_dim (int): dimension of hidden units in LSTM
        num_step (int): number of recurrent steps
        num_lstm_layer (int, optional): number of LSTM layers
    """

    eps = 1e-10

    def __init__(self, num_relation, hidden_dim, num_step, num_lstm_layer=1):
        super(NeuralLogicProgramming, self).__init__()

        num_relation = int(num_relation) // 2
        self.num_relation = num_relation
        self.num_step = num_step

        self.query = nn.Embedding(num_relation * 2 + 1, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_lstm_layer)
        self.weight_linear = nn.Linear(hidden_dim, num_relation * 2)
        self.linear = nn.Linear(1, 1)

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        # convert p(h | t, r) to p(t' | h', r')
        # h' = t, r' = r^{-1}, t' = h
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation)
        return new_h_index, new_t_index, new_r_index


    def get_t_output(self, data, h_index, r_index):
        end_index = torch.ones_like(r_index) * self.num_relation
        q_index = torch.stack([r_index] * (self.num_step - 1) + [end_index], dim=0)
        query = self.query(q_index)


        hidden, hx = self.lstm(query)
        memory = one_hot(h_index, data.num_nodes+1).unsqueeze(0)

        edge_weight = torch.ones(data.num_edges, device=h_index.device).unsqueeze(-1)

        node_in, node_out = data.edge_index
        relation = data.edge_type


        for i in range(self.num_step):
            key = hidden[i]
            value = hidden[:i + 1]
            x = torch.einsum("bd, tbd -> bt", key, value)
            attention = F.softmax(x, dim=-1)
            input = torch.einsum("bt, tbn -> nb", attention, memory)
            weight = F.softmax(self.weight_linear(key), dim=-1).t()

            # if data.num_nodes * self.num_relation < data.num_edges:
            #     # O(|V|d) memory
            #     node_out = node_out * self.num_relation + relation
            #     adjacency = torch.sparse_coo_tensor(torch.stack([node_in, node_out]), edge_weight,
            #                                         (data.num_nodes, data.num_nodes * self.num_relation))
            #     output = adjacency.t() @ input
            #     output = output.view(data.num_nodes, self.num_relation, -1)
            #     output = (output * weight).sum(dim=1)
            # else:
            #     # O(|E|) memory
            message = input[node_in]
            message = message * weight[relation]
            output = scatter_add(message * edge_weight, node_out, dim=0, dim_size=data.num_nodes+1)
        
            output = output / output.sum(dim=0, keepdim=True).clamp(self.eps)
            memory = torch.cat([memory, output.t().unsqueeze(0)])

        return output

    def forward(self, data, batch, **kwargs):
        """
        Compute the score for triplets.
        """
        # assert graph.num_relation == self.num_relation
        # graph = graph.undirected(add_inverse=True)

        h_index, t_index, r_index = batch.unbind(-1)

        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        hr_index = h_index * self.num_relation + r_index
        hr_index_set, hr_inverse = hr_index.unique(return_inverse=True)
        h_index_set = torch.div(hr_index_set, self.num_relation, rounding_mode="floor")
        r_index_set = hr_index_set % self.num_relation

        output = self.get_t_output(data, h_index_set, r_index_set)

        score = output[t_index, hr_inverse]
        score = self.linear(score.unsqueeze(-1)).squeeze(-1)
        return score


def one_hot(index, size):
    """
    Expand indexes into one-hot vectors.

    Parameters:
        index (Tensor): index
        size (int): size of the one-hot dimension
    """
    shape = list(index.shape) + [size]
    result = torch.zeros(shape, device=index.device)
    if index.numel():
        assert index.min() >= 0
        assert index.max() < size
        result.scatter_(-1, index.unsqueeze(-1), 1)
    return result



def get_mem():
    """
    Print all params and memory usage

    **Used for debugging purposes**
    """
    import gc
    import warnings
    import sys

    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)) and "cuda" in obj.device:
                print(obj.__class__.__name__, obj.shape, type(obj), sys.getsizeof(obj.storage()), obj.device)
        except: pass
