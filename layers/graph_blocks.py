import torch
import torch.nn as nn
from layers.graphs import GraphsTuple


def broadcast_receiver_nodes_to_edges(graph: GraphsTuple):
    return graph.nodes.index_select(index=graph.receivers.long(), dim=0)


def broadcast_sender_nodes_to_edges(graph: GraphsTuple):
    return graph.nodes.index_select(index=graph.senders.long(), dim=0)


def unsorted_sum_agg(data, segment_ids):
    num_edges = torch.unique(segment_ids).shape[0]
    output = torch.zeros((num_edges, *data.shape[1:])).to(data.device)
    if len(data.shape) == 3:
        output = output.scatter_add(0, segment_ids.unsqueeze(-1).unsqueeze(-1).expand(data.shape), data)
    else:
        output.scatter_add(0, segment_ids.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(data.shape), data)
    return output


def unsorted_mean_agg(data, segment_ids):
    unique_ids, counts = torch.unique(segment_ids, return_counts=True)
    num_edges = len(unique_ids)
    output = torch.zeros((num_edges, *data.shape[1:])).to(data.device)

    if len(data.shape) == 3:
        output = output.scatter_add(0, segment_ids.unsqueeze(-1).unsqueeze(-1).expand(data.shape), data) / counts[:, None, None]
    elif len(data.shape) == 2:
        output = output.scatter_add(0, segment_ids.unsqueeze(-1).expand(data.shape), data) / counts[:, None]
    else:
        raise NotImplementedError

    return output


def unsorted_softmax(data, segment_ids):
    data = torch.exp(data)
    num_edges = torch.unique(segment_ids).shape[0]

    denom = torch.zeros((num_edges, *data.shape[1:])).to(data.device)
    denom = denom.scatter_add(0, segment_ids.unsqueeze(-1).unsqueeze(-1).expand(data.shape), data)
    denom = denom.index_select(index=segment_ids.long(), dim=0)

    data = data / denom
    return data


class _EdgesToNodesAggregator(nn.Module):
    """Aggregates sent or received edges into the corresponding nodes."""

    def __init__(self, reducer, use_sent_edges=False):
        super(_EdgesToNodesAggregator, self).__init__()
        self._reducer = reducer
        self._use_sent_edges = use_sent_edges

    def forward(self, graph):
        indices = graph.senders if self._use_sent_edges else graph.receivers

        return [self._reducer(graph.edges, indices)]


class EdgeBlock(nn.Module):
    def __init__(self,
                 update_fn,
                 d_model,
                 use_edges=True,
                 use_receiver_nodes=True,
                 use_sender_nodes=True,
                 num_node_series=1,
                 num_edge_series=1):
        super(EdgeBlock, self).__init__()
        self._use_edges = use_edges
        self._use_receiver_nodes = use_receiver_nodes
        self._use_sender_nodes = use_sender_nodes

        if num_edge_series == 1:
            d_in = int(d_model * (use_edges * num_edge_series + use_receiver_nodes * num_node_series + use_sender_nodes * num_node_series))
            self.project = nn.Linear(d_in, d_model, bias=False)
        elif num_edge_series == 2:
            d_in = int(d_model * (use_edges + use_receiver_nodes + use_sender_nodes))
            self.project_freq = nn.Linear(d_in, d_model, bias=False)
            self.project_trend = nn.Linear(d_in, d_model, bias=False)
        else:
            raise NotImplementedError

        self.update_fn = update_fn
        self.num_node_series = num_node_series
        if self.num_node_series not in [1, 2]:
            raise NotImplementedError
        self.num_edge_series = num_edge_series

    def forward(self, graph: GraphsTuple, **kwargs):
        edges_to_collect = []

        if self._use_edges:
            if self.num_edge_series != 1:
                edges_to_collect += torch.unbind(graph.edges, dim=-1)
            else:
                edges_to_collect.append(graph.edges)

        if self._use_receiver_nodes:
            if self.num_node_series != 1:
                edges_to_collect += torch.unbind(broadcast_receiver_nodes_to_edges(graph), dim=-1)
            else:
                edges_to_collect.append(broadcast_receiver_nodes_to_edges(graph))  # (50,5)

        if self._use_sender_nodes:
            if self.num_node_series != 1:
                edges_to_collect += torch.unbind(broadcast_sender_nodes_to_edges(graph), dim=-1)
            else:
                edges_to_collect.append(broadcast_sender_nodes_to_edges(graph))  # (50,5)

        if self.num_edge_series == 1:
            collected_edges = torch.cat(edges_to_collect, dim=-1)
            collected_edges = self.project(collected_edges)
            updated_edges, attn_edges = self.update_fn(collected_edges, **kwargs)
        else:
            updated_edges = [
                self.project_freq(torch.cat(edges_to_collect[::2], dim=-1)),
                self.project_trend(torch.cat(edges_to_collect[1::2], dim=-1))
            ]

            updated_edges, attn_edges = self.update_fn(updated_edges, **kwargs)
            updated_edges = torch.stack(updated_edges, -1)
        if isinstance(updated_edges, list):
            assert len(updated_edges) == 2
            graph = [graph.replace(edges=updated_edges[0]), updated_edges[-1]]
        else:
            graph = graph.replace(edges=updated_edges)

        return graph, attn_edges


class NodeBlock(nn.Module):

    def __init__(self,
                 update_fn,
                 d_model,
                 use_received_edges=True,
                 use_sent_edges=False,
                 use_nodes=True,
                 edges_agg=unsorted_mean_agg,
                 num_node_series=1,
                 num_edge_series=1):
        super(NodeBlock, self).__init__()
        self._use_received_edges = use_received_edges
        self._use_sent_edges = use_sent_edges
        self._use_nodes = use_nodes

        d_in = int(d_model*(use_received_edges + use_sent_edges + use_nodes))
        if num_node_series == 1:
            self.project = nn.Linear(d_in, d_model, bias=False)
        elif num_node_series == 2:
            self.project_freq = nn.Linear(d_in, d_model, bias=False)
            self.project_trend = nn.Linear(d_in, d_model, bias=False)
        else:
            raise NotImplementedError

        self.update_fn = update_fn
        self._received_edges_aggregator = _EdgesToNodesAggregator(edges_agg, use_sent_edges=False)
        self._sent_edges_aggregator = _EdgesToNodesAggregator(edges_agg, use_sent_edges=True)
        self.num_node_series = num_node_series
        self.num_edge_series = num_edge_series

    def forward(self, graph, **kwargs):

        nodes_to_collect = []
        if self._use_received_edges:
            nodes_to_collect += self._received_edges_aggregator(graph)  # (24,10)

        if self._use_sent_edges:
            nodes_to_collect += self._sent_edges_aggregator(graph)

        if self._use_nodes:
            if self.num_node_series == 1:
                nodes_to_collect.append(graph.nodes)
            else:
                nodes_to_collect += torch.unbind(graph.nodes, dim=-1)
        if self.num_node_series == 1:
            collected_nodes = torch.cat(nodes_to_collect, dim=-1)  # 24,19
            collected_nodes = self.project(collected_nodes)
            updated_nodes, attn_nodes = self.update_fn(collected_nodes, **kwargs)  # 24,11
        else:
            updated_nodes = [
                self.project_freq(torch.cat([*nodes_to_collect[:-2], nodes_to_collect[-2]], dim=-1)),
                self.project_trend(torch.cat([*nodes_to_collect[:-2], nodes_to_collect[-1]], dim=-1))
            ]
            updated_nodes, attn_nodes = self.update_fn(updated_nodes, **kwargs)
            updated_nodes = torch.stack(updated_nodes, -1)
        if isinstance(updated_nodes, list):
            assert len(updated_nodes) == 2
            graph = [graph.replace(nodes=updated_nodes[0]), updated_nodes[-1]]
        else:
            graph = graph.replace(nodes=updated_nodes)
        return graph, attn_nodes


class Aggregator(nn.Module):
    def __init__(self, mode):
        super(Aggregator, self).__init__()
        self.mode = mode

    def forward(self, graph):
        edges = graph.edges
        nodes = graph.nodes
        if self.mode == 'receivers':
            indeces = graph.receivers
        elif self.mode == 'senders':
            indeces = graph.senders
        else:
            raise AttributeError("invalid parameter `mode`")
        N_edges, N_features = edges.shape
        N_nodes=nodes.shape[0]
        aggrated_list = []
        for i in range(N_nodes):
            aggrated = edges[indeces == i]
            if aggrated.shape[0] == 0:
                aggrated = torch.zeros(1, N_features)
            aggrated_list.append(torch.sum(aggrated, dim=0))
        return torch.stack(aggrated_list, dim=0)
