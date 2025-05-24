import os
import torch
import shutil
import numpy as np
import os.path as osp
from itertools import repeat
from feature_expansion import FeatureExpander
from torch_geometric.datasets import TUDataset
from torch_geometric.data.collate import collate
from torch_geometric.utils import degree
from torch_geometric.data import Dataset
import torch_geometric.utils as tg_utils
from torch_geometric.data import InMemoryDataset, download_url, extract_zip, Batch
from torch_geometric.io import read_tu_data
from torch_geometric.data.data import BaseData
from collections import defaultdict
import random

constant_num_FeatureExpander = 102
def get_dataset(name, feature_params, root=None):
    if root is None or root == '':
        path = 'datasets/'
    else:
        path = osp.join(root, name)

    pre_transform = FeatureExpander(feature_params).transform

    dataset = CombinedTUDataset(path, name, pre_transform=pre_transform, use_node_attr=True)

    return dataset


class CombinedTUDataset(InMemoryDataset):
    def __init__(self, root, dataset_name, transform=None, pre_transform=None, pre_filter=None, use_node_attr=True, use_edge_attr=True):
        """
        Initialize the combined dataset class
        :param root: Root path for data storage
        :param dataset_name: Dataset name or combination flag
        :param transform: Data transformation
        :param pre_transform: Preprocessing transformation
        :param pre_filter: Preprocessing filter
        """

        self.set_aug_mode('none')
        self.set_aug_ratio(0.2)
        self.set_aug_prob(np.ones(20) / 20)
        self.augmentations = [node_drop, subgraph, edge_pert, attr_mask, lambda x, y: x]
        self.get_k = 2
        self.name = dataset_name

        if dataset_name == "combined":
            # Define datasets to be combined
            # self.dataset_names = ['NCI109', 'NCI1']
            self.dataset_names = ['NCI1']
        else:
            # If not a combined dataset, load the single dataset directly
            self.dataset_names = [dataset_name]

        # Temporary data storage
        self.data_list = []
        max_node_attr_dim = 1
        max_edge_attr_dim = 1

        for name in self.dataset_names:
            dataset = TUDataset(root, name, transform = transform, pre_transform = pre_transform,
                                pre_filter = pre_filter, use_node_attr = use_node_attr, use_edge_attr = use_edge_attr)

            for data in dataset:
                # Update maximum node and edge attribute dimensions
                if data.x is not None:
                    max_node_attr_dim = max(max_node_attr_dim, data.x.size(1))

                if data.edge_attr is not None:
                    max_edge_attr_dim = max(max_edge_attr_dim, data.edge_attr.size(1))

                # data.edge_attr = None
                # Mark data source
                # data.dataset_name = name
                self.data_list.append(data)

        # Handle inconsistent node and edge attribute dimensions
        for data in self.data_list:
            if data.x is None:
                # If node attributes don't exist, fill with zero vectors
                data.x = torch.zeros(data.num_nodes, max_node_attr_dim or 1)
            elif data.x.size(1) < max_node_attr_dim:
                # If node attribute dimension is less than maximum, pad
                padding = max_node_attr_dim - data.x.size(1)
                tmp_padding = torch.cat([data.x[:, :-constant_num_FeatureExpander],torch.zeros(data.x.size(0), padding)], dim=1)
                data.x = torch.cat([tmp_padding, data.x[:,-constant_num_FeatureExpander:]], dim=1)

            # Handle edge attributes, but not considering edge attribute augmentation
            if data.edge_attr is None:
                data.edge_attr = torch.zeros(data.edge_index.size(1), max_edge_attr_dim or 1)
            elif data.edge_attr.size(1) < max_edge_attr_dim:
                padding = max_edge_attr_dim - data.edge_attr.size(1)
                data.edge_attr = torch.cat([data.edge_attr, torch.zeros(data.edge_attr.size(0), padding)], dim=1)

        # self.edge_prob = get_edge_sp_prob_m(self.data_list)
        # self.node_probs_all = get_g_n_prob(self.data_list, self.edge_prob)
        # for idx, data in enumerate(self.data_list):
        #     data.node_probs = self.node_probs_all[idx]


        super(CombinedTUDataset, self).__init__(root, transform, pre_transform, pre_filter)

        # Combine data and create slices
        self.data, self.slices = self.collate(self.data_list)
        self.all_labels = torch.unique(self.data.y)



    @staticmethod
    def collate(data_list):
        r"""Collates a Python list of :obj:`torch_geometric.data.Data` objects
        to the internal storage format of
        :class:`~torch_geometric.data.InMemoryDataset`."""
        if len(data_list) == 1:
            return data_list[0], None

        data, slices, _ = collate(
            data_list[0].__class__,
            data_list=data_list,
            increment=False,
            add_batch=False,)

        return data, slices


    def len(self):
        return len(self.data_list)

    def set_aug_mode(self, aug_mode='none'):
        self.aug_mode = aug_mode

    def set_aug_ratio(self, aug_ratio=0.2):
        self.aug_ratio = aug_ratio

    def set_aug_prob(self, prob):
        if prob.ndim == 2:
            prob = prob.reshape(-1)
        self.aug_prob = prob

    def set_get_k(self, k=0):
        self.get_k = k

    @property
    def num_edge_attributes(self):
        return self[0][0].edge_attr.size(1)
    @property
    def num_node_features(self):
        r"""Returns the number of features per node in the dataset."""
        return self[0][0].num_node_features

    def get(self, idx):
        get_k = self.get_k
        data_out = []
        data = self.data.__class__()
        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]
        for key in self.data.keys():
            if key == "num_nodes":
                continue
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                                        item)] = slice(slices[idx],
                                                       slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        data_out.append(data)

        for i in range(get_k):
            datak = self.data.__class__()
            aug_ratio = self.aug_ratio
            for key, item in data_out[0]:
                if isinstance(item, torch.Tensor):
                    datak[key] = item.clone()
                else: 
                    datak[key] = item

            # pre-defined augmentations
            if self.aug_mode == 'uniform':
                n_aug = np.random.choice(25, 1)[0]
                n_aug1 = n_aug // 5
                datak = self.augmentations[n_aug1](datak, aug_ratio)
            elif self.aug_mode == "base":
                datak = self.augmentations[0](datak,aug_ratio)

            else:
                datak = self.augmentations[-1](datak, aug_ratio)


            data_out.append(datak)

        return data_out

def _get_flattened_data_list(data_list):
    outs = []
    for data in data_list:
        if isinstance(data, BaseData):
            outs.append(data)
        elif isinstance(data, (tuple, list)):
            outs.extend(_get_flattened_data_list(data))
        elif isinstance(data, dict):
            outs.extend(_get_flattened_data_list(data.values()))
    return outs



def node_drop(data, aug_ratio, node_probs=None):
    node_num, _ = data.x.size()

    # Calculate number of nodes to remove
    drop_num = int(node_num * aug_ratio)
    if drop_num < 1:
        return data

    if node_probs==None:
        node_probs = torch.ones(node_num)

    # Use torch.multinomial for non-repetitive node sampling
    prob = 1 / (node_probs + 1e-5)

    drop_nodes = torch.multinomial(prob, drop_num, replacement=False)

    # Remove nodes
    idx_nondrop = list(set(range(node_num)) - set(drop_nodes.tolist()))
    # idx_nondrop.sort()

    # Use tg_utils.subgraph function to remove nodes and their associated edges
    edge_index, _ = tg_utils.subgraph(idx_nondrop, data.edge_index, relabel_nodes=True, num_nodes=node_num)

    # Update node features
    data.x = data.x[idx_nondrop]

    # Update edge indices
    data.edge_index = edge_index

    # Update node count
    data.__num_nodes__, _ = data.x.shape

    return data


def subgraph(data, aug_ratio):
    G = tg_utils.to_networkx(data)

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * (1-aug_ratio))

    idx_sub = [np.random.randint(node_num, size=1)[0]]

    idx_neigh = set([n for n in G.neighbors(idx_sub[-1])])

    while len(idx_sub) <= sub_num:
        if len(idx_neigh) == 0:
            idx_unsub = list(set([n for n in range(node_num)]).difference(set(idx_sub)))
            idx_neigh = set([np.random.choice(idx_unsub)])
        sample_node = np.random.choice(list(idx_neigh))

        idx_sub.append(sample_node)
        idx_neigh = idx_neigh.union(set([n for n in G.neighbors(idx_sub[-1])])).difference(set(idx_sub))


    idx_nondrop = idx_sub
    idx_nondrop.sort()

    edge_index, _ = tg_utils.subgraph(idx_nondrop, data.edge_index, relabel_nodes=True, num_nodes=node_num)

    data.x = data.x[idx_nondrop]
    data.edge_index = edge_index
    data.__num_nodes__, _ = data.x.shape
    return data


def edge_pert(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    pert_num = int(edge_num * aug_ratio)

    edge_index = data.edge_index[:, np.random.choice(edge_num, (edge_num - pert_num), replace=False)]

    idx_add = np.random.choice(node_num, (2, pert_num))

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_add[0], idx_add[1]] = 1
    adj[np.arange(node_num), np.arange(node_num)] = 0
    edge_index = adj.nonzero(as_tuple=False).t()

    data.edge_index = edge_index
    return data


def attr_mask(data, aug_ratio):
    node_num, _ = data.x.size()
    mask_num = int(node_num * aug_ratio)
    _x = data.x.clone()

    token = data.x.mean(dim=0)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)

    _x[idx_mask] = token
    data.x = _x
    return data


def custom_collate(data_list):
    batch = Batch.from_data_list([d[0] for d in data_list], follow_batch=['edge_index', 'edge_index_neg'])
    batch_1 = Batch.from_data_list([d[1] for d in data_list])
    batch_2 = Batch.from_data_list([d[2] for d in data_list])
    return batch, batch_1, batch_2







def get_edge_sp_prob_m(dataset):
    # Count edge occurrences between atoms
    edge_count = defaultdict(int)
    total_edge = 0
    max =-1

    # Extract edges and node labels from all graphs
    for graph in dataset:
        # x is node features, usually one-hot encoded
        node_labels = torch.argmax(graph.x[:,:-constant_num_FeatureExpander], dim=1)  # Get type label for each node (assuming one-hot encoding)
        edge_index = graph.edge_index
        total_edge += edge_index.size(1)
        if max < node_labels.max():
            max = node_labels.max()

        # Traverse edges, record edges between each pair of nodes
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            u_label, v_label = node_labels[u].item(), node_labels[v].item()  # Get atom labels for nodes u and v
            edge_count[(u_label, v_label)] += 1

    # Calculate bond formation probability between atoms
    edge_probability = {}
    for edge, count in edge_count.items():
        edge_probability[edge] = count / total_edge

    # Prepare indices and values for sparse matrix
    indices = []
    values = []

    for edge, prob in edge_probability.items():
        indices.append(list(edge))
        values.append(prob)

    # Convert to PyTorch tensors
    indices = torch.tensor(indices).T  # Transpose to match COO format
    values = torch.tensor(values)

    # Create sparse tensor
    probability_matrix_sparse = torch.sparse_coo_tensor(indices, values, size=(max+1, max+1))

    return probability_matrix_sparse

def get_g_n_prob(dataset, probability_matrix_sparse):
    node_probs_all = []

    # Extract edges and node labels from all graphs
    for graph in dataset:
        # x is node features, usually one-hot encoded
        node_labels = torch.argmax(graph.x[:,:-constant_num_FeatureExpander], dim=1)  # temp code, only used default config NCI1.
        edge_index = graph.edge_index
        node_num = graph.x.size(0)
        node_probs = torch.zeros(node_num)

        # Traverse edges, record edges between each pair of nodes
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            u_label, v_label = node_labels[u].item(), node_labels[v].item()

            # Get corresponding probability from probability matrix
            edge_prob = probability_matrix_sparse[u_label, v_label].item()  # Get bond formation probability from u to v

            node_probs[u] += edge_prob  # Accumulate to node u's deletion probability
            node_probs[v] += edge_prob  # Accumulate to node v's deletion probability

        # node_probs = 1/torch.softmax(node_probs, dim=-1)

        node_probs_all.append(node_probs)

    return node_probs_all
