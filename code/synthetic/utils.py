import torch
import random
import numpy as np
from torch_geometric.data import Data,Dataset
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import subgraph
from torch_geometric.data.collate import collate

def init_similarity_log(n: int, log_path: str = "out.log"):
    """
    Initialize CSV log file and write header.
    Each call will overwrite the original file.
    """
    col_names = [f"m{i}{j}" for i in range(n) for j in range(n)]  # including diagonal elements
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + col_names)

def write_similarity_row(epoch: int, sim_mat, log_path: str = "out.log"):
    """
    Append a row of all elements from sim_mat to the CSV file (including diagonal).
    """
    n = sim_mat.shape[0]
    assert sim_mat.shape == (n, n), "`sim_mat` must be a square matrix"

    # Flatten the entire matrix in row-major order, including diagonal
    row_vals = [sim_mat[i, j].item() for i in range(n) for j in range(n)]

    # Write to file
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch] + row_vals)

def create_random_pyg_graph(
    num_nodes=20,
    avg_edges=40,
    feat_dim=16,
    class_idx=None
) -> Data:
    """
    Generate a random PyG Data:
    - number of nodes = num_nodes
    - average number of edges ~ avg_edges
    - node feature dimension = feat_dim (randomly initialized)
    - optional class_idx
    """
    # Calculate connection probability p to achieve expected edge count ~ avg_edges
    # For undirected graph, max edges = num_nodes*(num_nodes - 1)/2
    # p = avg_edges / max_edges
    max_edges = num_nodes * (num_nodes - 1) // 2
    if max_edges == 0:
        p = 0.0
    else:
        p = avg_edges / max_edges
        p = min(p, 1.0)  # avoid too large probability

    # Generate a [num_nodes x num_nodes] random adjacency matrix (upper triangle is sufficient)
    # Note: only need to perform Bernoulli(p) for (i<j), then make it symmetric
    adj_matrix = (torch.rand(num_nodes, num_nodes) < p).int()

    # Ensure diagonal is 0 (no self-loops) and make it symmetric
    adj_matrix = torch.triu(adj_matrix, diagonal=1)
    adj_matrix = adj_matrix + adj_matrix.t()

    # Convert to edge_index
    edge_index, _ = dense_to_sparse(adj_matrix)

    # Randomly initialize node features
    x = torch.randn(num_nodes, feat_dim)

    data = Data(x=x, edge_index=edge_index)
    if class_idx is not None:
        data.class_idx = class_idx  # Used to know which class this graph belongs to during augmentation
    return data


def generate_base_graphs(class_num=5, num_nodes=20, avg_edges=40, feat_dim=16):
    """
    Generate class_num random base graphs (PyG Data),
    where each graph has different number of nodes/features/edges (random).
    """
    base_graphs = []
    for cidx in range(class_num):
        data = create_random_pyg_graph(
            num_nodes=num_nodes,
            avg_edges=avg_edges,
            feat_dim=feat_dim,
            class_idx=cidx
        )
        base_graphs.append(data)
    return base_graphs


def add_random_mask(data: Data, max_mask_rate=0.2) -> Data:
    num_nodes = data.num_nodes
    if num_nodes == 0:
        return data

    # Randomly generate mask_rate
    mask_rate = random.random() * max_mask_rate
    mask_count = int(mask_rate * num_nodes)
    if mask_count == 0:
        return data

    # Get nodes to keep
    all_nodes = list(range(num_nodes))
    del_nodes = random.sample(all_nodes, mask_count)
    keep_nodes = list(set(all_nodes) - set(del_nodes))

    # Use subgraph to get new edge_index, automatically relabel nodes
    new_edge_index, _ = subgraph(
        keep_nodes,
        data.edge_index,
        relabel_nodes=True
    )

    # Select feature subset and reorder according to relabel_nodes
    # Mapping from kept nodes to new indices
    keep_nodes_sorted = sorted(keep_nodes)
    index_map = {old: new for new, old in enumerate(keep_nodes_sorted)}
    new_x = data.x[keep_nodes_sorted]

    new_data = Data(x=new_x, edge_index=new_edge_index)

    if hasattr(data, 'y'):
        new_data.y = data.y
    if hasattr(data, 'class_idx'):
        new_data.class_idx = data.class_idx

    return new_data


def sample_and_build_dataset(base_graphs, n, P, max_mask_rate=0.2):
    """
    - base_graphs: [Data_0, Data_1, ..., Data_(class_num-1)]
    - n: number of samples needed
    - P: sampling probability distribution, length=class_num
    - max_mask_rate: maximum mask ratio when adding noise
    Returns a list of noisy PyG Data (each data has .class_idx)
    """
    class_num = len(base_graphs)
    dataset_list = []

    for _ in range(n):
        cidx = np.random.choice(class_num, p=P)
        base_data = base_graphs[cidx]

        # Add random node mask to original graph
        noisy_data = add_random_mask(base_data, max_mask_rate)

        # Record class information
        noisy_data.class_idx = cidx
        dataset_list.append(noisy_data)

    return dataset_list

###############################################################################
# Custom Dataset: Add noise during construction, perform augmentation during data retrieval
###############################################################################


class AugmentedGraphDataset(Dataset):
    def __init__(self, data_list, base_graphs, transition_matrix, max_mask_rate=0.2, k=2):
        """
        - data_list: List of PyG Data with added noise, each Data has data.class_idx
        - base_graphs: Original graph set needed for augmentation
        - transition_matrix: State transition matrix
        - max_mask_rate: Maximum node mask ratio when adding noise during augmentation
        - k: Return k augmented graphs each time
        """
        super().__init__()
        self.data_list = data_list
        self.base_graphs = base_graphs
        self.transition_matrix = transition_matrix
        self.max_mask_rate = max_mask_rate
        self.k = k
        self.class_num = len(base_graphs)

    def __len__(self):
        return len(self.data_list)

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
            add_batch=False, )

        return data, slices

    def __getitem__(self, idx):
        """
        Returns: [source_graph, aug_1, aug_2, ..., aug_k]
        Source graph comes from data_list[idx] (already has noise),
        Augmented graphs are obtained from base_graphs of transition class, then add noise.
        """
        data = self.data_list[idx]
        original_class = data.class_idx

        # Generate k augmented graphs
        aug_list = [data]
        for _ in range(self.k):
            # Sample new class using transition matrix
            new_class = np.random.choice(self.class_num, p=self.transition_matrix[original_class])

            # Get original graph of corresponding class from base_graphs, then add random mask
            base_data = self.base_graphs[new_class]
            aug_data = add_random_mask(base_data, self.max_mask_rate)

            # Record the class after augmentation
            aug_data.class_idx = new_class
            aug_list.append(aug_data)

        return aug_list


def get_synthetic_graph_dataset(class_num=3, graph_num=10, max_mask_rate=0.2, feat_dim=16, k=2, P=None, transition_matrix=None):


    base_graphs = generate_base_graphs(
        class_num=class_num,
        num_nodes=20,
        avg_edges=40,
        feat_dim=feat_dim
    )

    noisy_dataset_list = sample_and_build_dataset(
        base_graphs,
        n=graph_num,
        P=P,
        max_mask_rate=max_mask_rate
    )

    dataset = AugmentedGraphDataset(
        data_list=noisy_dataset_list,
        base_graphs=base_graphs,
        transition_matrix=transition_matrix,
        max_mask_rate=max_mask_rate,
        k=k
    )
    return dataset




###############################################################################
# Main process example
###############################################################################
if __name__ == "__main__":

    dataset = get_synthetic_graph_dataset()

    # Test: Get first 3 samples, each returns [source_graph, aug_graph1, aug_graph2]
    for i in range(3):
        sample_list = dataset[i]
        print(f"\n=== Sample {i} ===")
        print(f"-  Source: num_nodes={sample_list[0].num_nodes}, num_edges={sample_list[0].num_edges}, label={sample_list[0].class_idx}")
        for j, aug_data in enumerate(sample_list[1:], start=1):
            print(f"   Aug {j}: num_nodes={aug_data.num_nodes}, num_edges={aug_data.num_edges}, label={aug_data.class_idx}")


