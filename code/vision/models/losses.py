import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import diffdist
import torch.distributed as dist
from .utils import *


def gather(z):
    gather_z = [torch.zeros_like(z) for _ in range(torch.distributed.get_world_size())]
    gather_z = diffdist.functional.all_gather(gather_z, z)
    gather_z = torch.cat(gather_z)

    return gather_z


def accuracy(logits, labels, k):
    topk = torch.sort(logits.topk(k, dim=1)[1], 1)[0]
    labels = torch.sort(labels, 1)[0]
    acc = (topk == labels).all(1).float()
    return acc


def mean_cumulative_gain(logits, labels, k):
    topk = torch.sort(logits.topk(k, dim=1)[1], 1)[0]
    labels = torch.sort(labels, 1)[0]
    mcg = (topk == labels).float().mean(1)
    return mcg


def mean_average_precision(logits, labels, k):
    # TODO: not the fastest solution but looks fine
    argsort = torch.argsort(logits, dim=1, descending=True)
    labels_to_sorted_idx = torch.sort(torch.gather(torch.argsort(argsort, dim=1), 1, labels), dim=1)[0] + 1
    precision = (1 + torch.arange(k, device=logits.device).float()) / labels_to_sorted_idx
    return precision.sum(1) / k


class NTXent(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """
    LARGE_NUMBER = 1e9

    def __init__(self, tau=1., gpu=None, multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed
        self.norm = 1.

    def forward(self, z, get_map=False):

        n = z.shape[0]
        assert n % self.multiplier == 0

        z = F.normalize(z, p=2, dim=1) / np.sqrt(self.tau) 

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
            z = torch.cat(z_sorted, dim=0)
            n = z.shape[0]

        logits = z @ z.t()
        logits[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER

        logprob = F.log_softmax(logits, dim=1)

        # logprob = symmetric_normal(logits)
        # logprob = torch.log(logprob)

        # choose all positive objects for an example, for i it would be (i + k * n/m), where k=0...(m-1)
        m = self.multiplier
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n // m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)

        # TODO: maybe different terms for each process should only be computed here...
        loss = -logprob[np.repeat(np.arange(n), m - 1), labels].sum() / n / (m - 1) / self.norm

        # zero the probability of identical pairs
        pred = logprob.data.clone()
        pred[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER
        acc = accuracy(pred, torch.LongTensor(labels.reshape(n, m - 1)).to(logprob.device), m - 1)

        if get_map:
            _map = mean_average_precision(pred, torch.LongTensor(labels.reshape(n, m - 1)).to(logprob.device), m - 1)
            return loss, acc, _map

        return loss, acc


class gcl_IS(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """

    def __init__(self, tau=1., multiplier=2, alpha = 1., beta = 0., distributed=False):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed
        self.norm = 1.
        self.SMALL_NUM = np.log(1e-45)
        self.alpha = alpha
        self.beta = beta
        self.include_augs = True
        self.gamma = 16

    def forward(self, z, get_map=False):#1

        n = z.shape[0]
        assert n % self.multiplier == 0
        assert self.multiplier == 2

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...]
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
            z = torch.cat(z_sorted, dim=0)

        x1, x2 = z.chunk(self.multiplier, dim=0)
        T = self.tau

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)/ T
        pos_sim = torch.diag(matrix)
        neg_sim = (matrix.sum(dim=1) - pos_sim) / (x1.size(0)-1)

        sim_matrix = torch.exp(matrix )
        pos = torch.diag(sim_matrix)
        p_ij = pos / (sim_matrix.sum(dim=1) - pos)

        with torch.no_grad():
            alpha = p_ij - 1 + self.alpha  # v5

        loss = - torch.log(p_ij) - alpha * pos_sim + neg_sim * (self.beta)  # [-1,],好像是线性增加的

        return loss.mean(), p_ij.mean()

    def forward_IS(self, z, get_map=False):#2,c100,57.9

        n = z.shape[0]
        assert n % self.multiplier == 0
        assert self.multiplier == 3

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...]
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
            z = torch.cat(z_sorted, dim=0)

        x1, x2, x3 = z.chunk(self.multiplier, dim=0)
        T = self.tau

        c1 = torch.cosine_similarity(x1, x2)
        pos_sim = torch.exp(c1 / T)

        x3_abs = x3.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x3, x3) / torch.einsum('i,j->ij', x3_abs, x3_abs)
        sim_matrix = torch.exp(sim_matrix / T)

        p_ij = pos_sim / (sim_matrix.sum(dim=1) - torch.diag(sim_matrix))

        loss = - torch.log(p_ij).mean()

        return loss, p_ij.mean()

    def forward_DCL(self, z, get_map=False):#3

        n = z.shape[0]
        assert n % self.multiplier == 0
        assert self.multiplier == 2
        z = F.normalize(z, p=2, dim=1)

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...]
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
            z = torch.cat(z_sorted, dim=0)
            n = z.shape[0]

        z1, z2 = z.chunk(self.multiplier, dim=0)
        T = self.tau

        cross_view_distance = torch.mm(z1, z2.t())
        positive_loss = -torch.diag(cross_view_distance) / T
        # if self.weight_fn is not None:
        #     positive_loss = positive_loss * self.weight_fn(z1, z2)
        neg_similarity = torch.cat((torch.mm(z1, z1.t()), cross_view_distance), dim=1) / T
        neg_mask = torch.eye(z1.size(0), device=z1.device).repeat(1, 2)
        negative_loss = torch.logsumexp(neg_similarity + neg_mask * self.SMALL_NUM, dim=1, keepdim=False)
        return (positive_loss + negative_loss).mean(), 0.0


    def forward_DHEL(self, z, get_map=False):#4
        """
        Decoupled Hyperspherical Energy Loss (DHEL) from https://arxiv.org/abs/2405.18045.
        """

        n = z.shape[0]
        assert n % self.multiplier == 0
        assert self.multiplier == 2
        batch_size = n // 2

        z = F.normalize(z, p=2, dim=1)

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...]
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
            z = torch.cat(z_sorted, dim=0)
            batch_size = z.shape[0]//2

        z_1, z_2 = z.chunk(self.multiplier, dim=0)

        sim_matrix_anchor = torch.exp(torch.mm(z_1, z_1.t()) / self.tau)

        # Create a mask to exclude self-similarity
        mask = torch.eye(batch_size, device=z.device).bool()
        sim_matrix_anchor = sim_matrix_anchor.masked_fill(mask, 0)

        # Compute the positive similarities between anchor and positive samples
        pos_sim = torch.exp(torch.sum(z_1 * z_2, dim=-1) / self.tau)

        if self.include_augs:
            # Compute the similarity matrix for the positive samples
            sim_matrix_pos = torch.exp(torch.mm(z_2, z_2.t()) / self.tau)
            sim_matrix_pos = sim_matrix_pos.masked_fill(mask, 0)

            # Compute the contrastive loss including augmentations
            loss = -torch.log(pos_sim / (sim_matrix_anchor.sum(dim=-1) * sim_matrix_pos.sum(dim=-1))).mean()
        else:
            # Compute the contrastive loss without including augmentations
            loss = -torch.log(pos_sim / sim_matrix_anchor.sum(dim=-1)).mean()

        with torch.no_grad():
            p_ij = pos_sim / sim_matrix_anchor.sum(dim=-1)


        return loss, p_ij.mean()

    def forward_GKCL(self, z, get_map=False):#5, Results not available
        """
        Gaussian-Kernel Contrastive Loss (KCL) from https://arxiv.org/abs/2405.18045.
        """

        n = z.shape[0]
        assert n % self.multiplier == 0
        assert self.multiplier == 2
        batch_size = n // 2

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...]
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
            z = torch.cat(z_sorted, dim=0)
            batch_size = z.shape[0]//2


        x1, x2 = z.chunk(self.multiplier, dim=0)

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)


        matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        pos_sim = torch.diag(matrix)
        neg_sim = (matrix.sum(dim=1) - pos_sim) / (x1.size(0) - 1)

        loss = - pos_sim + neg_sim

        return loss.mean(), 0.0


    def resort_data(self, z):

        z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
        # all_gather fills the list as [<proc0>, <proc1>, ...]
        # TODO: try to rewrite it with pytorch official tools
        z_list = diffdist.functional.all_gather(z_list, z)
        # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
        z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
        # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
        z_sorted = []
        for m in range(self.multiplier):
            for i in range(dist.get_world_size()):
                z_sorted.append(z_list[i * self.multiplier + m])
        z = torch.cat(z_sorted, dim=0)
        n = z.shape[0]

        return z, n



def symmetric_normal(A, max_iter=10):  # c10 90.68

    A = torch.exp(A)
    for _ in range(max_iter):
        row_sums = torch.sum(A, dim=1, keepdim=True)
        A = A / row_sums
        A = (A + A.T) / 2

    return A


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
#c10 32， 100; c100 32, 47.81 128, 47.60
    def __init__(self, div_type: str = "kl", *, sigma: float = 1.0, alpha: float = 64.0,
                 tsallis_alpha: float = 1.5, eps: float = 1e-8,  tau: float =1.,
                 multiplier: int=2, distributed: bool=False):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed

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
    def forward(self,  z, get_map=False):
        n = z.shape[0]
        assert n % self.multiplier == 0
        assert self.multiplier == 2

        z = F.normalize(z, p=2, dim=1)

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...]
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
            z = torch.cat(z_sorted, dim=0)

        z1, z2 = z.chunk(self.multiplier, dim=0)

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
