import torch
from layers.graphs import GraphsTuple
import numpy as np


def data_dicts_to_graphs_tuple(graph_dicts: dict, device=None):
    for k, v in graph_dicts.items():
        if k in ['senders', 'receivers', 'n_node', 'n_edge', 'graph_mapping']:
            graph_dicts[k] = torch.tensor(v, dtype=torch.int64)
        elif k == 'station_names':
            continue
        else:
            graph_dicts[k] = torch.tensor(v, dtype=torch.float32)
        if device is not None:
            graph_dicts[k] = graph_dicts[k].to(device)
    return GraphsTuple(**graph_dicts)


# In development. Should work fine, but might contain some bugs.
def split_torch_graph(graph, target_gpus):
    target_gpus = ['cuda:' + str(gpu) for gpu in target_gpus]
    bs = graph.n_node.shape[0]

    sub_bs = np.array_split(np.arange(bs), len(target_gpus))

    sum_node_prev = 0
    sum_edge_prev = 0
    graph_list = []
    for gpu_i, sub_i in zip(target_gpus, sub_bs):
        sub_i = torch.from_numpy(sub_i).to(graph.nodes.device).long()
        graph_i = graph
        end_node = torch.sum(graph.n_node[sub_i]) + sum_node_prev
        end_edge = torch.sum(graph.n_edge[sub_i]) + sum_edge_prev

        graph_i = graph_i.replace(
            nodes=graph.nodes[sum_node_prev:end_node].to(gpu_i),
            edges=graph.edges[sum_edge_prev:end_edge].to(gpu_i),
            senders=(graph.senders[sum_edge_prev:end_edge] - sum_node_prev).to(gpu_i),
            receivers=(graph.receivers[sum_edge_prev:end_edge] - sum_node_prev).to(gpu_i),
            n_node=graph.n_node[sub_i].to(gpu_i),
            n_edge=graph.n_edge[sub_i].to(gpu_i),
            station_names=graph.station_names[sum_node_prev:end_node],
        )
        sum_node_prev = end_node
        sum_edge_prev = end_edge
        graph_list.append(graph_i)

    return graph_list, sub_bs, target_gpus
