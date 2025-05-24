import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import csv
from transformers import get_cosine_schedule_with_warmup
from utils import *
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
import argparse



import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(2025)

class FMICLLoss(nn.Module):
    """Fast, self‑contained implementation of f‑MICL contrastive loss.

    Args:
        div_type (str): one of {"kl", "chi2", "js", "sh", "tsallis", "vlc"}
        sigma (float): Gaussian kernel σ (default 0.5, matching paper)
        alpha (float): weighting for negative term (default 1.0)
        tsallis_alpha (float): α for Tsallis‑α divergence (only when div_type=="tsallis")
        eps (float): numerical stability constant
    Input:
        z1, z2: L2‑normalised feature tensors of shape (N, d)
    Output:
        scalar loss value (shape: ())
    """

    def __init__(self, div_type: str = "kl", *, sigma: float = 1.0, alpha: float = 1.5,
                 tsallis_alpha: float = 1.5, eps: float = 1e-8):
        super().__init__()
        self.sigma, self.alpha, self.eps = sigma, alpha, eps
        self.f_prime, self.f_star = self._get_funcs(div_type.lower(), tsallis_alpha)

    # ---------------------------------------------------------------------
    def _get_funcs(self, div: str, a: float):
        e = self.eps  # shortcut for numerical eps
        if div == "kl":  # Kullback–Leibler
            f_p = lambda u: torch.log(u + e) + 1.0
            f_s = lambda t: torch.exp(t - 1.0)
        elif div == "chi2":  # Pearson χ²
            f_p = lambda u: 2.0 * (u - 1.0)
            f_s = lambda t: 0.25 * t ** 2 + t
        elif div == "js":  # Jensen–Shannon
            f_p = lambda u: torch.log(torch.tensor(2.0, dtype=u.dtype, device=u.device)) + torch.log(u + e) - torch.log(1.0 + u)
            f_s = lambda t: -torch.log(2.0 - torch.exp(t))
        elif div == "sh":  # Squared‑Hellinger
            f_p = lambda u: 1.0 - torch.rsqrt(u + e)
            f_s = lambda t: t / (1.0 - t + e)
        elif div == "tsallis":  # Tsallis‑α
            f_p = lambda u, a=a: -(a / (a - 1.0)) * u ** (a - 1.0)
            f_s = lambda t, a=a: ((a - 1.0) * t / a).pow(a / (a - 1.0))
        elif div == "vlc":  # Variational lower‑bound contrastive (paper)
            f_p = lambda u: 1.0 - 4.0 / (u + 1.0) ** 2
            f_s = lambda t: 4.0 - t - 4.0 * torch.sqrt(1.0 - t + e)
        else:
            raise ValueError(f"Unknown divergence type: {div}")
        return f_p, f_s

    # ---------------------------------------------------------------------
    def forward(self, z_list):
        z1, z2 = z_list[1], z_list[2]

        z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
        N = z1.size(0)

        # ------------------ positive term ------------------
        d_pos = (z1 - z2).pow(2).sum(dim=1)  # squared L2 distance, shape (N,)
        g_pos = torch.exp(-d_pos / (2.0 * self.sigma ** 2))
        s_pos = self.f_prime(g_pos)

        # ------------------ negative term ------------------
        # pairwise squared distances on the first view (saves FLOPs)
        d2 = torch.cdist(z1, z1, p=2).pow(2)  # (N, N)
        g_neg = torch.exp(-d2 / (2.0 * self.sigma ** 2))
        s_neg = self.f_prime(g_neg)

        # exclude diagonal (i == j)
        star_neg = self.f_star(s_neg[~torch.eye(N, dtype=torch.bool, device=z1.device)])

        loss = -s_pos.mean() + self.alpha * star_neg.mean()

        return loss, 0.0


# -------- convenient factory function -------------------------------------------------

def get_f_micl_loss(**kwargs):
    """Tiny helper so you can write: loss_fn = get_f_micl_loss(div_type='chi2')"""
    return FMICLLoss(**kwargs)

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, proj_dim, alpha=0.5, beta=-0.1, tau=0.2):
        super().__init__()
        self.conv1 = GINConv(nn.Linear(in_channels, hidden_channels))
        self.conv2 = GINConv(nn.Linear(hidden_channels, hidden_channels))
        self.pool = global_mean_pool

        self.alpha = alpha
        self.beta = beta
        self.T = tau
        self.LARGE_NUMBER = 1e9

        self.proj_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, proj_dim)
        )

    def forward(self, x, edge_index, batch):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h_graph = self.pool(h, batch)
        z = self.proj_head(h_graph)
        return z

    def InfoNCE(self, z_list):

        x0, x1, x2 = z_list[0],z_list[1],z_list[2]

        z = torch.cat([x1,x2], dim=0)

        n = z.shape[0]
        z = F.normalize(z, p=2, dim=1) / np.sqrt(self.T)

        logits = z @ z.t()
        # logits[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER

        logprob = F.softmax(logits, dim=1)
        # logprob = symmetric_normal(logits)
        #exponential_kernel_normalization

        logprob = torch.log(logprob)

        # choose all positive objects for an example, for i it would be (i + k * n/m), where k=0...(m-1)
        m = 2
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n // m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)

        # TODO: maybe different terms for each process should only be computed here...
        loss = -logprob[np.repeat(np.arange(n), m - 1), labels].sum() / n / (m - 1)

        x0_abs = x0.norm(dim=1)
        matrix_x0 = torch.einsum('ik,jk->ij', x0, x0) / torch.einsum('i,j->ij', x0_abs, x0_abs)
        prob = F.softmax(matrix_x0, dim=-1)
        prob = prob[0]

        return loss, prob[:10]

    def DCL(self, z_list):

        x0, x1, x2 = z_list[0],z_list[1],z_list[2]
        T = self.T

        x0_abs = x0.norm(dim=1)
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        matrix_x0 = torch.einsum('ik,jk->ij', x0, x0) / torch.einsum('i,j->ij', x0_abs, x0_abs)
        prob = F.softmax(matrix_x0, dim=-1)
        prob = prob[0]


        matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs) / T

        sim_matrix = torch.exp(matrix)
        pos = torch.diag(sim_matrix)
        p_ij = pos / (sim_matrix.sum(dim=1) - pos)

        loss = - torch.log(p_ij)

        return loss.mean(), prob[:10]

    def simple_cl(self, z_list):

        x0, x1, x2 = z_list[0],z_list[1],z_list[2]

        x0_abs = x0.norm(dim=1)
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        matrix_x0 = torch.einsum('ik,jk->ij', x0, x0) / torch.einsum('i,j->ij', x0_abs, x0_abs)
        # matrix_x0[np.arange(batch_size), np.arange(batch_size)] = -1e9
        prob = F.softmax(matrix_x0, dim=-1)
        prob = prob[0]

        matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        pos_sim = torch.diag(matrix)
        neg_sim = (matrix.sum(dim=1) - pos_sim)/(x1.size(0)-1)

        loss = - pos_sim + neg_sim

        return loss.mean(), prob[:10]

    def SC_InfoNCE(self, z_list):

        x0, x1, x2 = z_list[0],z_list[1],z_list[2]

        T = self.T

        x0_abs = x0.norm(dim=1)
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)


        matrix_x0 = torch.einsum('ik,jk->ij', x0, x0) / torch.einsum('i,j->ij', x0_abs, x0_abs)
        prob = F.softmax(matrix_x0, dim=-1)
        prob = prob[0]


        matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs) / T
        pos_sim = torch.diag(matrix)
        neg_sim = (matrix.sum(dim=1) - pos_sim) / (x1.size(0) - 1)

        # p_ij = symmetric_normal(matrix)

        sim_matrix = torch.exp(matrix)
        pos = torch.diag(sim_matrix)
        p_ij = pos / (sim_matrix.sum(dim=1) - pos)

        with torch.no_grad():
            alpha = p_ij - 1 + self.alpha  # v5

        loss = - torch.log(p_ij) - alpha * pos_sim + neg_sim * (self.beta)

        return loss.mean(), prob[:10]

    def IS_InfoNCE(self, z_list):

        x1, x2, x3 = z_list[1],z_list[2],z_list[3]

        T = self.T

        c1 = torch.cosine_similarity(x1, x2)
        pos_sim = torch.exp(c1 / T)

        x3_abs = x3.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x3, x3) / torch.einsum('i,j->ij', x3_abs, x3_abs)
        sim_matrix = torch.exp(sim_matrix / T)

        p_ij = pos_sim / (sim_matrix.sum(dim=1) - torch.diag(sim_matrix))

        loss = - torch.log(p_ij).mean()

        return loss, 0.0

    def DHEL(self, z_list, include_augs=True):#4
        """
        Decoupled Hyperspherical Energy Loss (DHEL) from https://arxiv.org/abs/2405.18045.
        """

        z_1, z_2 = z_list[1], z_list[2]

        z_1 = F.normalize(z_1, p=2, dim=1)
        z_2 = F.normalize(z_2, p=2, dim=1)



        sim_matrix_anchor = torch.exp(torch.mm(z_1, z_1.t()) / self.T)

        # Create a mask to exclude self-similarity
        mask = torch.eye(z_1.size(0), device=z_1.device).bool()
        sim_matrix_anchor = sim_matrix_anchor.masked_fill(mask, 0)

        # Compute the positive similarities between anchor and positive samples
        pos_sim = torch.exp(torch.sum(z_1 * z_2, dim=-1) / self.T)

        if include_augs:
            # Compute the similarity matrix for the positive samples
            sim_matrix_pos = torch.exp(torch.mm(z_2, z_2.t()) / self.T)
            sim_matrix_pos = sim_matrix_pos.masked_fill(mask, 0)

            # Compute the contrastive loss including augmentations
            loss = -torch.log(pos_sim / (sim_matrix_anchor.sum(dim=-1) * sim_matrix_pos.sum(dim=-1))).mean()
        else:
            # Compute the contrastive loss without including augmentations
            loss = -torch.log(pos_sim / sim_matrix_anchor.sum(dim=-1)).mean()



        return loss, 0.0

    def GKCL(self, z_list):#5, Results not available
        """
        Gaussian-Kernel Contrastive Loss (KCL) from https://arxiv.org/abs/2405.18045.
        """

        z_0, z_1, z_2 = z_list[0], z_list[1], z_list[2]

        def gaussian_kernel(x, t=2):
            """
            Compute the pairwise potential (energy) based on the Gaussian kernel.

            Args:
                x (Tensor): Input tensor of shape (M, d) where M is the number of samples and d is the embedding dimension.
                t (float): Scaling parameter. Default is 2.
            """
            pairwise_distances = torch.pdist(x, p=2)
            return pairwise_distances.pow(2).mul(-t).exp().mean()

        def align_gaussian(x, y, t):
            """
            Compute the alignment between anchor points and their positives based on the Gaussian kernel.

            Args:
                x (Tensor): Tensor of shape (M, d) containing anchor embeddings.
                y (Tensor): Tensor of shape (M, d) containing the corresponding positive embeddings.
                t (float): Scaling parameter.
            """
            pairwise_distances = (x - y).norm(p=2, dim=1)
            return pairwise_distances.pow(2).mul(-t).exp().mean()

        energy = (gaussian_kernel(z_1, self.T).mean() +
                  gaussian_kernel(z_2, self.T).mean())
        alignment = 2 * align_gaussian(z_1, z_2, self.T)


        loss = -alignment + 128 * energy

        return loss, 0.0

    def conver_target(self, z, k=3):
        total_batch, dim = z.shape
       
        assert total_batch % k == 0, f"Input z must be divisible by {k} along dim 0"

        z = F.normalize(z, p=2, dim=1)

        logits = z @ z.t()

        matrix = logits.reshape(k, total_batch//k, k, total_batch//k)

        matrix = matrix.mean(dim=3).mean(dim=1)

        return matrix



def generate_random_P1(num_states):
    P1 = np.random.rand(num_states)
    P1 /= np.sum(P1)
    return P1

def generate_random_A(num_states):
    A = np.random.rand(num_states, num_states)
    A /= A.sum(axis=1, keepdims=True)
    return A


def split_graphs(dataset, train_ratio=0.5, val_ratio=0.0, test_ratio=0.1):
    """
    Randomly split nodes into training, validation and test sets (boolean masks).
    data: Graph data object containing .x, .y, .edge_index etc. (e.g. PyG Data object)
    train_ratio, val_ratio, test_ratio: Split ratios for train/val/test
    """
    from torch.utils.data import random_split

    total = len(dataset)
    train_len = int(train_ratio * total)
    val_len = int(val_ratio * total)
    test_len = total - train_len - val_len

    return random_split(dataset, [train_len, val_len, test_len])

def get_c1_c2(P1, A, state_num):

    P1 = torch.tensor(P1)
    A = torch.tensor(A)

    P1A = P1 @ A

    c1 = torch.zeros(state_num, state_num)
    for i in range(state_num):
        for j in range(state_num):
            for k in range(state_num):
                c1[i][j] += P1[k] * (A[k][i] * A[k][j])

    c2 = torch.zeros(state_num, state_num)

    for i in range(state_num):
        for j in range(state_num):
            for k in range(state_num):
                c2[i][j] += P1[k] * (A[k][i] * P1A[j])

    return c1, c2

def metric_fn(z: torch.Tensor, labels: torch.Tensor, T: float) -> torch.Tensor:

    n = z.size(0)
    device = z.device
    labels = labels.to(device)

    # Normalization + temperature scaling
    z = F.normalize(z, p=2, dim=1)
    logits = z @ z.T / T  # (n, n)

    # Mask diagonal (can be commented out if not needed)
    # logits.fill_diagonal_(-1e9)

    # softmax → Pij probability matrix
    pij = F.softmax(logits, dim=1)  # (n, n)

    # Get number of classes
    num_classes = int(labels.max().item()) + 1

    # Construct mask for each class: bool Tensor (num_classes, n)
    class_mask = torch.stack([(labels == c) for c in range(num_classes)])  # (C, n)

    # Initialize result: average probability of each sample for each class
    probs_by_class = torch.zeros(n, num_classes, device=device)

    for c in range(num_classes):
        idx_c = class_mask[c]  # shape: (n,)
        # For each row i → average probability for class c
        # i.e., pij[i, j] where j ∈ class c
        probs_by_class[:, c] = pij[:, idx_c].mean(dim=1)

    return probs_by_class  # (n, C)

def aggregate_probs_by_class(
    probs_by_class: torch.Tensor,  # (n, C)
    labels: torch.Tensor,          # (n,)
    reduction: str = "mean",       # only support "mean"
) -> torch.Tensor:


    if reduction != "mean":
        raise NotImplementedError("Only support reduction='mean'")

    device = probs_by_class.device
    labels = labels.to(device)

    n, num_classes = probs_by_class.shape

    class_sum = torch.zeros(num_classes, num_classes, device=device)
    class_sum.index_add_(0, labels, probs_by_class)

    counts = torch.bincount(labels, minlength=num_classes).unsqueeze(1)  # (C, 1)

    class_matrix = class_sum / counts.clamp_min(1)
    class_matrix[counts.squeeze(1) == 0] = float("nan")

    return class_matrix

def get_model_emb(data, model, device):
    out = []
    for b0 in data:
        b0 = b0.to(device)
        out.append(model(b0.x, b0.edge_index, b0.batch))
    return out


def symmetric_normal(A, max_iter=10):  # c10

    A = torch.exp(A)
    for _ in range(max_iter):
        row_sums = torch.sum(A, dim=1, keepdim=True)
        A = A / row_sums
        A = (A + A.T) / 2

    return A


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='synthetic toy example')
    parser.add_argument('--sample_num', default=4000, type=int, help='dataset size')
    parser.add_argument('--state_num', default=3, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--feat_dim', default=8, type=int)
    parser.add_argument('--hidden_dim', default=16, type=int)
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--loss_type', default=0, type=int)
    parser.add_argument('--aug_k', default=2, type=int)
    parser.add_argument('--tau', default=1.0, type=float)
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--beta', default= -0.0, type=float)
    parser.add_argument('--outlog', default=0, type=int)

    args = parser.parse_args()

    out_num = 0

    start_out = f'out_{out_num}.log'

    if args.outlog==0:
        out_log = f'out_{out_num}.log'
    else:
        out_log = f'out_{out_num}{args.outlog}.log'


    # get synthetic dataset

    # P1 = generate_random_P1(args.state_num)
    # A = generate_random_A(args.state_num)
    #
    P1 = np.array([1/3, 1/3, 1/3])
    A = np.array([[0.5, 0.3, 0.2],
                 [0.2, 0.5, 0.3],
                 [0.2, 0.3, 0.5]])

    if args.loss_type==2:
        args.aug_k = 3


    dataset = get_synthetic_graph_dataset(class_num=args.state_num, graph_num=args.sample_num, feat_dim=args.feat_dim, P=P1, transition_matrix=A, k=args.aug_k)

    train_set, val_set, test_set = split_graphs(dataset)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    # test_loader = DataLoader(test_set, batch_size=32, shuffle=False)


    device = f'cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    model = GNN(args.feat_dim, args.hidden_dim, args.hidden_dim, tau=args.tau, alpha=args.alpha, beta=args.beta).to(device)

    print(f"loss_type: {args.loss_type}")
    if args.loss_type==0:
        criterion = model.InfoNCE
    elif args.loss_type==1:
        criterion = model.SC_InfoNCE
    elif args.loss_type==2:
        criterion = model.IS_InfoNCE
    elif args.loss_type==3:
        criterion = model.DCL
    elif args.loss_type == 4:
        criterion = model.DHEL
    elif args.loss_type == 5:
        criterion = model.GKCL
    elif args.loss_type == 6:
        criterion = model.simple_cl
    elif args.loss_type == 7:
        criterion = get_f_micl_loss(div_type='js')#"kl" "chi2" "js" "sh" "tsallis" "vlc"
    else:
        raise NotImplementedError(f"loss type '{args.loss_type}' is not yet implemented")

    num_warmup_steps = 10
    total_steps = args.epochs+1

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=total_steps 
    )

    init_similarity_log(args.state_num, out_log)

    for epoch in range(0, total_steps):
        model.train()
        total_loss = 0.0

        for data_list in train_loader:
            z_list = get_model_emb(data_list, model, device)

            loss, pij = criterion(z_list)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)


        if epoch % args.log_step == 0:

            model.eval()
            if len(test_set)<args.batch_size:
                bs = len(test_set)
            else:
                bs = args.batch_size
            loader = DataLoader(
                test_set,
                batch_size=bs,
                shuffle=True,
                drop_last=True,
            )

            met_chunks = []
            emb_chunks = []
            lab_chunks = []

            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    data = batch[0]
                else:
                    data = batch

                data = data.to(device)

                embeddings = model(data.x, data.edge_index, data.batch)  # (B, d)
                labels = data.class_idx  # (B,)
                unique_labels = torch.unique(labels)
                if unique_labels.size(0)!= args.state_num:
                    break

                metric = metric_fn(embeddings, labels, args.tau)

                met_chunks.append(metric.cpu())
                emb_chunks.append(embeddings.cpu())
                lab_chunks.append(labels.cpu())


            test_met = torch.cat(met_chunks, dim=0)  # (N, d)
            test_lab = torch.cat(lab_chunks, dim=0)  # (N, )
            test_emb = torch.cat(emb_chunks, dim=0)

            Pij = aggregate_probs_by_class(test_met, test_lab)
            write_similarity_row(epoch, Pij, out_log)
            print(Pij)

            unique_labels = torch.unique(test_lab)
            num_samples_per_class = args.sample_num//args.state_num//40

            get_idx = []
            for lbl in unique_labels:
                idx = (test_lab == lbl).nonzero().view(-1)
                get_idx.extend(idx[:num_samples_per_class])

            get_idx = torch.tensor(get_idx)

            sim_matrix = model.conver_target(test_emb[get_idx], args.state_num)
            print(sim_matrix)
            print("!!!!!!!!!!!!!!!!!!")
            write_similarity_row(epoch, sim_matrix, out_log)

    c1, c2 = get_c1_c2(P1, A, args.state_num)

    E_p = c1/(c1+c2* (args.batch_size-1))
    E_p = E_p.numpy()

    print("InfoNCE：")
    print(E_p)

    E_p = c1/((args.batch_size-1)*c2) *args.alpha - args.beta/(args.batch_size-1)
    E_p = E_p.numpy()

    print("SC_InfoNCE：")
    print(E_p)

