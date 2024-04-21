"""
    Performs graph structure learning on the MSG as outlined in the
    paper.
"""
import os
import time
import json
import glob
import numpy as np
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from collections import Counter
from sklearn.metrics import r2_score
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


# from .model import Model
# from .utils.generic_utils import to_cuda
# from .utils.data_utils import prepare_datasets, DataStream, vectorize_input
# from .utils import Timer, DummyLogger, AverageMeter
# from .utils import constants as Constants
# from .layers.common import dropout
# from .layers.anchor import sample_anchors, batch_sample_anchors, batch_select_from_tensor,

# from .models.graph_clf import GraphClf
# from .models.text_graph import TextGraphRegression, TextGraphClf
# from .utils.text_data.vocab_utils import VocabModel
# from .utils import constants as Constants
# from .utils.generic_utils import to_cuda, create_mask
# from .utils.constants import INF
# from .utils.radam import RAdam

# from .model import Model
# from .utils.generic_utils import to_cuda
# from .utils.data_utils import prepare_datasets, DataStream, vectorize_input
# from .utils import Timer, DummyLogger, AverageMeter
# from .utils import constants as Constants
# from .layers.common import dropout
# from .layers.anchor import sample_anchors, batch_sample_anchors, batch_select_from_tensor, compute_anchor_adj

VERY_SMALL_NUMBER = 1e-12
INF = 1e20


def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm


class GraphLearner(nn.Module):
    """
        Implements GSL.
    """

    def __init__(self):
        def __init__(self, input_size, hidden_size, topk=None, epsilon=None, num_pers=16, metric_type='weighted_cosine', device=None):
            super(GraphLearner, self).__init__()
            self.device = device
            self.topk = topk
            self.epsilon = epsilon
            self.metric_type = metric_type

            # if metric_type == 'weighted_cosine'
            self.weight_tensor = torch.Tensor(num_pers, input_size)
            self.weight_tensor = nn.Parameter(
                nn.init.xavier_uniform_(self.weight_tensor))
            print(
                '[ Multi-perspective {} GraphLearner: {} ]'.format(metric_type, num_pers))

    def forward(self, context, ctx_mask=None):
        # if metric_type == 'weighted_cosine'
        expand_weight_tensor = self.weight_tensor.unsqueeze(1)
        if len(context.shape) == 3:
            expand_weight_tensor = expand_weight_tensor.unsqueeze(1)

        context_fc = context.unsqueeze(0) * expand_weight_tensor
        context_norm = F.normalize(context_fc, p=2, dim=-1)
        attention = torch.matmul(
            context_norm, context_norm.transpose(-1, -2)).mean(0)
        markoff_value = 0

        # if ctx_mask is not None:
        #     attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(1), markoff_value)
        #     attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(-1), markoff_value)

        if self.epsilon is not None:
            attention = self.build_epsilon_neighbourhood(
                attention, self.epsilon, markoff_value)

        # if self.topk is not None:
        #     attention = self.build_knn_neighbourhood(attention, self.topk, markoff_value)

        return attention

    def build_epsilon_neighbourhood(self, attention, epsilon, markoff_value):
        mask = (attention > epsilon).detach().float()
        weighted_adjacency_matrix = attention * \
            mask + markoff_value * (1 - mask)
        return weighted_adjacency_matrix
