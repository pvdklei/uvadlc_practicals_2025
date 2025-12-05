import torch.nn as nn
import torch

import torch.nn.functional as F
from torch_geometric.utils import add_self_loops


class MatrixGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(MatrixGraphConvolution, self).__init__()
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.B = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.B)

    def make_adjacency_matrix(self, edge_index, num_nodes):
        """
        Creates adjacency matrix from edge index.

        :param edge_index: [source, destination] pairs defining directed edges nodes. dims: [2, num_edges]
        :param num_nodes: number of nodes in the graph.
        :return: adjacency matrix with shape [num_nodes, num_nodes]

        Hint: A[i,j] -> there is an edge from node j to node i
        """
        adjacency_matrix = torch.zeros(
            (num_nodes, num_nodes), device=edge_index.device
        )
        adjacency_matrix[edge_index[1], edge_index[0]] = 1.0
        return adjacency_matrix

    def make_inverted_degree_matrix(self, edge_index, num_nodes):
        """
        Creates inverted degree matrix from edge index.

        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :param num_nodes: number of nodes in the graph.
        :return: inverted degree matrix with shape [num_nodes, num_nodes]. Set degree of nodes without an edge to 1.
        """
        degree_vector = torch.sum(
            self.make_adjacency_matrix(edge_index, num_nodes), dim=1, dtype=torch.float
        )
        inverted_degree_vector = 1.0 / degree_vector
        inverted_degree_matrix = torch.diag(inverted_degree_vector)
        return inverted_degree_matrix

    def forward(self, x, edge_index):
        """
        Forward propagation for GCNs using efficient matrix multiplication.

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :return: activations for the GCN
        """
        A = self.make_adjacency_matrix(edge_index, x.size(0))
        D_inv = self.make_inverted_degree_matrix(edge_index, x.size(0))
        out = D_inv @ A @ x @ self.W.T + x @ self.B.T
        return out


class MessageGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(MessageGraphConvolution, self).__init__()
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.B = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.B)

    @staticmethod
    def message(x, edge_index):
        """
        message step of the message passing algorithm for GCNs.

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :return: message vector with shape [num_nodes, num_in_features]. Messages correspond to the old node values.

        Hint: check out torch.Tensor.index_add function
        """

        # Finding the messages correlating to each origin node (the neighbors)
        messages = torch.zeros((edge_index.size(0), x.size(1)), device=x.device)
        messages = x[edge_index[0]]

        # Sum all messages for each destination node (the node with neighbors)
        aggregated_messages = torch.zeros_like(x)
        aggregated_messages = torch.index_add(
            aggregated_messages, 0, edge_index[1], messages
        )

        # Normalize with number of incoming messages (taking the mean over neighbors)
        sum_weight = torch.zeros((x.size(0), 1), device=x.device)
        ones = torch.ones((edge_index.size(1), 1), device=x.device)
        sum_weight = torch.index_add(sum_weight, 0, edge_index[1], ones)
        aggregated_messages = aggregated_messages / sum_weight.clamp(min=1.0)

        return aggregated_messages

    def update(self, x, messages):
        """
        update step of the message passing algorithm for GCNs.

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param messages: messages vector with shape [num_nodes, num_in_features]
        :return: updated values of nodes. shape: [num_nodes, num_out_features]
        """
        x = x @ self.B.T + messages @ self.W.T
        return x

    def forward(self, x, edge_index):
        message = self.message(x, edge_index)
        x = self.update(x, message)
        return x


class GraphAttention(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttention, self).__init__()
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.a = nn.Parameter(torch.Tensor(out_features * 2))

        nn.init.xavier_uniform_(self.W)
        nn.init.uniform_(self.a, 0, 1)

    def forward(self, x, edge_index, debug=False):
        """
        Forward propagation for GATs.
        Follow the implementation of Graph attention networks (Veličković et al. 2018).

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :param debug: used for tests
        :return: updated values of nodes. shape: [num_nodes, num_out_features]
        :return: debug data for tests:
                 messages -> messages vector with shape [num_nodes, num_out_features], i.e. Wh from Veličković et al.
                 edge_weights_numerator -> unnormalized edge weightsm i.e. exp(e_ij) from Veličković et al.
                 softmax_denominator -> per destination softmax normalizer

        Hint: the GAT implementation uses only 1 parameter vector and edge index with self loops
        Hint: It is easier to use/calculate only the numerator of the softmax
              and weight with the denominator at the end.

        Hint: check out torch.Tensor.index_add function
        """
        edge_index, _ = add_self_loops(edge_index)

        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        in_features = x.size(1)
        out_features = self.W.size(0)


        sources, destinations = edge_index

        assert sources.size() == (num_edges,)

        activations = x @ self.W.T  # shape: [num_nodes, out_features]
        messages = activations[sources]  # shape: [num_edges, out_features]

        assert activations.size() == (num_nodes, out_features)
        assert messages.size() == (num_edges, out_features)

        attention_inputs = torch.cat(
            [activations[destinations], activations[sources]], dim=1
        )  # shape: [num_edges, 2 * out_features]

        assert attention_inputs.size() == (num_edges, 2 * out_features)

        edge_weights_numerator = torch.exp(F.leaky_relu(
            attention_inputs @ self.a, negative_slope=0.2
        ))  # shape: [num_edges]
        weighted_messages = messages * edge_weights_numerator.unsqueeze(
            -1
        )  # shape: [num_edges, out_features]

        assert edge_weights_numerator.size() == (num_edges,)

        softmax_denominator = torch.zeros(num_nodes, device=x.device)
        softmax_denominator = torch.index_add(
            softmax_denominator, 0, destinations, edge_weights_numerator
        )  # shape: [num_nodes]

        aggregated_messages = torch.zeros((num_nodes, out_features))
        aggregated_messages = torch.index_add(
            input=aggregated_messages, dim=0, index=destinations, source=weighted_messages
        )
        aggregated_messages = aggregated_messages / softmax_denominator.unsqueeze(-1)

        return aggregated_messages, {
            "edge_weights": edge_weights_numerator,
            "softmax_weights": softmax_denominator,
            "messages": messages,
        }
