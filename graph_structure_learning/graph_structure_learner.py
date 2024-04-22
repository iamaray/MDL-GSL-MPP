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


class CosineSimilarityModule(nn.Module):
    """
        Performs the weighted cosine computation and the graph update.
    """

    def __init__(self):
        def __init__(
                self,
                num_nodes,
                num_pers=16):
            super(CosineSimilarityModule, self).__init__()
            self.device = device

            self.weight_tensor = torch.Tensor(num_pers, num_nodes)

            self.weight_tensor = nn.Parameter(
                nn.init.xavier_uniform_(self.weight_tensor))

        def forward(
                self,
                node_features,
                adj):
            """
            Parameters
                :node_features, (num_pers, input_size)
                :adjacency_matrix, (input_size, input_size)

            Returns
                :cosine_similarity_matrix, (input_size, input_size)
            """
            # Ensure the adjacency matrix is a float tensor
            adjacency_matrix = adjacency_matrix.float()

            # Compute the weighted node features
            weighted_node_features = node_features * self.weight_tensor

            # Normalize the weighted node features
            weighted_node_features_norm = F.normalize(
                weighted_node_features, p=2, dim=1)

            # Compute the cosine similarity matrix
            cosine_similarity_matrix = torch.matmul(
                weighted_node_features_norm, weighted_node_features_norm.t())

            # Set the diagonal to zero (self-similarity is not defined)
            cosine_similarity_matrix.fill_diagonal_(0)

            return cosine_similarity_matrix


class MatrixRefineModule(nn.Module):
    def __init__(
            self,
            curr_node_features,
            adj,
            adj_0,
            adj_1=None,
            input_size=1000,
            lam=0,
            eta=0):
        super(MatrixRefineModule, self).__init__()
        self.lam = lam
        self.eta = eta
        self.curr_node_features = curr_node_features
        self.adj = adj
        self.adj_0 = adj_0
        self.adj_1 = adj_1
        self.cosineSim = CosineSimilarityModule(input_size)

    def forward(self):
        combined_refined_adj = None
        refinedMat = self.cosineSim(self.curr_node_features, self.adj)
        if adj_1 != None:
            combined_refined_adj = (self.lam * self.adj_0) + (1 - self.lam) * \
                (self.eta * refinedMat + (1 - self.eta) * self.adj_1)

        return refinedMat, combined_refined_adj


class InterMolecularGNN(nn.Module):
    def __init__(self):
        super(InterMolecularGNN, self).__init__()
        pass

    def forward(self):
        pass


class PredictionModule(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, gradient=True):
        super(PredictionModule, self).__init__()
        self.gradient = gradient
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.relu = nn.ReLU()

    def forward(self, node_features):
        x = node_features
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        out = self.layers[-1](x)

        output = {"output": out}

        if self.gradient and out.requires_grad:
            # Assuming `node_features` is part of a larger data object that includes `pos` and `displacement`
            # This part is adapted from the provided model and might need adjustments based on your specific use case
            volume = torch.einsum("zi,zi->z", node_features[:, 0, :], torch.cross(
                node_features[:, 1, :], node_features[:, 2, :], dim=1)).unsqueeze(-1)
            grad = torch.autograd.grad(
                out,
                # Assuming `node_features` is the tensor you want to compute gradients for
                [node_features],
                grad_outputs=torch.ones_like(out),
                create_graph=self.training
            )
            forces = -1 * grad[0]
            stress = forces  # Assuming `stress` is equivalent to `forces` in this context
            stress = stress / volume.view(-1, 1, 1)

            output["node_grad"] = forces
            output["stress"] = stress
        else:
            output["node_grad"] = None
            output["stress"] = None

        return output


class GraphLearner(nn.Module):
    def __init__(
            self,
            data,
            iters,
            laplace_weight=0,
            initAdj_weight=0,
            intermolec_gnn='STGNN'):
        # Assuming `data` is your PyTorch Geometric graph data object
        edge_index = data.edge_index
        edge_weights = data.edge_attr

        # Number of nodes in the graph
        num_nodes = data.num_nodes
        # Initialize matrix of node embeddings
        self.init_embedding_matrix = data.x
        # Initialize an adjacency matrix filled with zeros
        self.init_adjacency_matrix = torch.zeros((num_nodes, num_nodes))
        # Fill the adjacency matrix with edge weights
        for i, (src, tgt) in enumerate(edge_index.t()):
            self.init_adjacency_matrix[src, tgt] = edge_weights[i]

        self.T = iters
        self.laplace_weight = laplaceWeight
        self.init_adj_weight = initAdjWeight

        self.intermolec_gnn = InterMolecularGNN(intermolec_gnn)
        self.refine_matrix = MatrixRefineModule()
        self.refine_embeddings = EmbeddingsRefineModule()
        self.predict = PredictionModule()
        self.refined_mats = [(self.epsilonMaskedInitAdj, None)]
        self.learned_embedding_mats = [self.init_embedding_matrix]

    def forward(self):
        t = 1
        while t <= self.T:
            refined_adj, combined_refined_adj = self.refineMatrix(
                self.refinedMats[t-1][0], self.learned_embedding_mats[t-1])
            self.refinedMats += [(refinedAdj, combinedRefinedAdj)]

            curr_embedding_mat = self.learned_embedding_mats[0]

            curr_learned_embedding_mat = self.intermolec_gnn(
                curr_embedding, combined_refined_adj)

            self.learned_embedding_mats += [curr_learned_embedding_mat]

            t += 1

        return self.predict(self.learned_embedding_mats[T-1])


"""
    TRAINING LOOP
"""


def loss_gsl():
    pass


def loss_pred():
    pass


def loss(mu):
    return (mu * loss_gsl) + ((1 - mu) * loss_pred)


