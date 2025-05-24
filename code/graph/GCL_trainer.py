import torch
import torch.optim as optim
import numpy as np
from torch_geometric.data import DataLoader
import os
import torch.nn.functional as F
import torch_geometric.utils as tg_utils
import torch.nn as nn

import random
from torch.utils.data import Sampler
from torch.utils.data import Dataset

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

        return loss



class GraphContrastiveLearningTrainer:
    def __init__(self, config):
        self.config = config
        self.batch_size = config["pre_params"]["batch_size"]
        self.epochs = config["pre_params"]["epochs"]
        self.lr = config["pre_params"]["lr"]
        self.use_pre = config["pre_params"]["use_pre"]
        self.T = config["pre_params"]["T"]
        self.loss_type = config["pre_params"]["loss_type"]
        self.weights_dir = config["pre_params"]["weights_dir"]
        self.aug_ratio = config["pre_params"]["aug_ratio"]
        self.alpha = config["pre_params"]["alpha"]
        self.beta = config["pre_params"]["beta"]
        self.aug_K = config["pre_params"]["aug_K"]

        self.LARGE_NUMBER = 1e9
        self.need_load_pre_model = config["pre_params"]["load_pre_model"]

        if config["pre_params"]["loss_type"] == 2:
            self.aug_num = 3
        else:
            self.aug_num = 2

        self.fmicl = FMICLLoss(div_type='kl', sigma= self.alpha, alpha=self.beta)

    def save_sij_grad(self, grad):
        self.sij_grad = grad.clone()

    def train(self, dataset, model, ref_model, device):
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(1, self.epochs + 1):
            model.train()
            ref_model.eval()
            total_loss = 0.0

            for idx, data_list in enumerate(data_loader):
                # if data_list[0].y.size(0) < self.batch_size:
                #     continue

                out_list = self.get_model_emb(data_list, model, device)

                if self.loss_type==0:
                    loss = self.InfoNCE(out_list)
                elif self.loss_type==1:
                    loss = self.SC_InfoNCE(out_list)
                elif self.loss_type==2:
                    loss = self.IS_InfoNCE(out_list)
                elif self.loss_type == 3:
                    loss = self.DCL(out_list)
                elif self.loss_type == 4:
                    loss = self.DHEL(out_list)
                elif self.loss_type == 5:
                    loss = self.GKCL(out_list)
                elif self.loss_type ==6:
                    loss = self.simple_cl(out_list)
                elif self.loss_type == 7:
                    loss = self.fmicl(out_list)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print(torch.diag(self.sij_grad).mean())
                # print((self.sij_grad.sum(dim=-1)-torch.diag(self.sij_grad)).mean())
                # print("!!!!!!!!!!!!!!!!!!!")

                total_loss += loss.item()


            avg_loss = total_loss / len(data_loader)

            print(f"Epoch [{epoch}/{self.epochs}], Loss: {avg_loss:.4f}")
            # if epoch% 1 ==0:
            #     print(f"P_ij: {p_ij:.4f}")

    def InfoNCE(self, z_list):

        x0, x1, x2 = z_list[0], z_list[1], z_list[2]

        z = torch.cat([x1, x2], dim=0)

        n = z.shape[0]
        z = F.normalize(z, p=2, dim=1) / np.sqrt(self.T)

        logits = z @ z.t()
        # logits[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER

        logprob = F.softmax(logits, dim=1)
        # logprob = symmetric_normal(logits)
        # exponential_kernel_normalization

        logprob = torch.log(logprob)

        # choose all positive objects for an example, for i it would be (i + k * n/m), where k=0...(m-1)
        m = 2
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n // m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)

        # TODO: maybe different terms for each process should only be computed here...
        loss = -logprob[np.repeat(np.arange(n), m - 1), labels].sum() / n / (m - 1)

        return loss

    def DCL(self, z_list):

        x0, x1, x2 = z_list[0], z_list[1], z_list[2]
        T = self.T

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)


        matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs) / T

        sim_matrix = torch.exp(matrix)
        pos = torch.diag(sim_matrix)
        p_ij = pos / (sim_matrix.sum(dim=1) - pos)

        loss = - torch.log(p_ij)

        return loss.mean()

    def simple_cl(self, z_list):

        x0, x1, x2 = z_list[0], z_list[1], z_list[2]

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)


        matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        pos_sim = torch.diag(matrix)
        neg_sim = (matrix.sum(dim=1) - pos_sim) / (x1.size(0) - 1)

        loss = - pos_sim + neg_sim

        return loss.mean()

    def SC_InfoNCE(self, z_list):

        x0, x1, x2 = z_list[0], z_list[1], z_list[2]

        T = self.T

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)


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

        return loss.mean()

    def IS_InfoNCE(self, z_list):

        x1, x2, x3 = z_list[1], z_list[2], z_list[3]

        T = self.T

        c1 = torch.cosine_similarity(x1, x2)
        pos_sim = torch.exp(c1 / T)

        x3_abs = x3.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x3, x3) / torch.einsum('i,j->ij', x3_abs, x3_abs)
        sim_matrix = torch.exp(sim_matrix / T)

        p_ij = pos_sim / (sim_matrix.sum(dim=1) - torch.diag(sim_matrix))

        loss = - torch.log(p_ij).mean()

        return loss

    def DHEL(self, z_list, include_augs=True):  # 4
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

        return loss

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

        return loss


    def run_contrastive_learning(self, dataset, model, ref_model, device):
        if not self.use_pre:
            return 0

        # self.load_model(model, device)

        if not self.need_load_pre_model:
            # if self.check_param_exists():
            #     self.load_ref_model(ref_model, device)

            self.train(dataset, model, ref_model, device)
            self.save_model(model)
            self.need_load_pre_model = True
        else:
            self.load_model(model, device)


    def save_model(self, model):
        """Save model parameters"""
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)
        dataset_name = self.config['dataset']
        model_name = self.config['net_params']['gnn_type']
        diff = self.config['diff']
        filename = f"{dataset_name}_{model_name}_{diff}.pth"
        checkpoint_path = os.path.join(self.weights_dir, filename)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model parameters saved to {checkpoint_path}")

    def check_param_exists(self):
        """Check if pre-trained model parameters exist"""
        if not os.path.exists(self.weights_dir):
            return False
        files = os.listdir(self.weights_dir)

        dataset_name = self.config['dataset']
        model_name = self.config['net_params']['gnn_type']
        # diff = self.config['diff']

        matching_files = [f for f in files if f.startswith(f"{dataset_name}_{model_name}_")]

        return len(matching_files) > 0

    def load_model(self, model, device):
        """Load existing model parameters"""
        dataset_name = self.config['dataset']
        model_name = self.config['net_params']['gnn_type']
        diff = self.config['diff']
        filename = f"{dataset_name}_{model_name}_{diff}.pth"
        checkpoint_path = os.path.join(self.weights_dir, filename)

        pretrained_dict = torch.load(checkpoint_path, map_location=device, weights_only=True) # Only load parameters that exist in pre-training
        model_dict = model.state_dict()


        keys = []
        for k, v in pretrained_dict.items():  # Only load parameters that exist in pre-training
            keys.append(k)

        for i, t in enumerate(model_dict.items()):
            k, v = t
            if v.size() == pretrained_dict[keys[i]].size():
                model_dict[k] = pretrained_dict[keys[i]]
                # i = i + 1
        model.load_state_dict(model_dict)

        print("Successfully loaded pre-trained model parameters.")

    def load_ref_model(self, model, device):
        """Load existing model parameters"""
        dataset_name = self.config['dataset']
        model_name = self.config['net_params']['gnn_type']
        diff = self.config['ref_diff']
        filename = f"{dataset_name}_{model_name}_{diff}.pth"
        checkpoint_path = os.path.join(self.weights_dir, filename)

        pretrained_dict = torch.load(checkpoint_path, map_location=device, weights_only=True) # Only load parameters that exist in pre-training
        model_dict = model.state_dict()


        keys = []
        for k, v in pretrained_dict.items():  # Only load parameters that exist in pre-training
            keys.append(k)

        for i, t in enumerate(model_dict.items()):
            k, v = t
            if v.size() == pretrained_dict[keys[i]].size():
                model_dict[k] = pretrained_dict[keys[i]]
                # i = i + 1
        model.load_state_dict(model_dict)

        print("Successfully loaded reference model parameters.")

    def get_model_emb(self, data, model, device):
        out = []
        for i in data:
            out.append(model.forward_pre(i.to(device)))
        return out

    def get_sim_matrix(self, z1):

        n = z1.shape[0]

        z1 = F.normalize(z1, p=2, dim=1) / np.sqrt(self.T)  # 标准化

        s1 = z1 @ z1.T

        s1[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER

        s1 = F.softmax(s1, dim=1)

        return s1



