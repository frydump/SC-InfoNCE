import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()

def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    # Separate representation
    z1, z2 = pooler_output[:,0], pooler_output[:,1]

    # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))


    # Separate representation
    z1, z2 = pooler_output[:,0], pooler_output[:,1]
    z3 = pooler_output[:,2] if num_sent == 3 else pooler_output[:,0]  # 若无z3，直接用z2占位

        
    if cls.model_args.loss_type == 0:
        loss = simcse_loss(z2, z3, T=cls.model_args.temp)
    elif cls.model_args.loss_type == 1:
        loss = sc_infonce_loss(z2, z3, 
                             T=cls.model_args.temp,
                             alpha=cls.model_args.sc_alpha,
                             beta=cls.model_args.sc_beta)
    elif cls.model_args.loss_type == 2:
        loss = simclr_loss(z2, z3, T=cls.model_args.temp)
    elif cls.model_args.loss_type == 3:
        loss = dcl_loss(z2, z3, T=cls.model_args.temp)
    elif cls.model_args.loss_type == 4:
        loss = dhel_loss(z2, z3, T=cls.model_args.temp)
    elif cls.model_args.loss_type == 5:
        loss = gkcl_loss(z2, z3, T=cls.model_args.temp)
    elif cls.model_args.loss_type == 6:
        loss = simple_cl_loss(z2, z3, T=cls.model_args.temp)
    elif cls.model_args.loss_type == 7:
        f_micl_loss = FMICLLoss(div_type="kl", sigma=cls.model_args.sc_beta, alpha=cls.model_args.sc_alpha)

        loss = f_micl_loss(z2, z3)
    else:
        raise NotImplementedError(f"Loss type {cls.model_args.loss_type} is not implemented")

    loss_fct = nn.CrossEntropyLoss()

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )



class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )

def simclr_loss(x1, x2, T=0.05, LARGE_NUMBER=1e9):

    z = torch.cat([x1, x2], dim=0)  # (2N, hidden)
    n = z.shape[0]
    z = F.normalize(z, p=2, dim=1) / np.sqrt(T)

    logits = z @ z.t()  # (2N, 2N)
    # logits[torch.arange(n), torch.arange(n)] = -LARGE_NUMBER 

    logprob = F.softmax(logits, dim=1)
    logprob = torch.log(logprob)

    m = 2
    labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n // m, n)) % n
    labels = labels.reshape(n, m)[:, 1:].reshape(-1)

    loss = -logprob[np.repeat(np.arange(n), m - 1), labels].sum() / n / (m - 1)

    return loss

def info_nce(z1, z2, temperature=0.05):
    """
    z1, z2: (N, hidden_dim)
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sim_matrix = torch.matmul(z1, z2.T) / temperature  # (N, N)
    labels = torch.arange(z1.size(0)).to(z1.device)
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

def sc_infonce_loss(x1, x2, T=0.05, alpha=1.0, beta=0.0):
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)
    
    matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs) / T
    pos_sim = torch.diag(matrix)
    neg_sim = (matrix.sum(dim=1) - pos_sim) / (x1.size(0) - 1)
    
    sim_matrix = torch.exp(matrix)
    pos = torch.diag(sim_matrix)
    p_ij = pos / (sim_matrix.sum(dim=1)- pos_sim)
    
    with torch.no_grad():
        alpha = - 1 + alpha
    
    loss = - torch.log(p_ij) - alpha * pos_sim + neg_sim * beta
    
    return loss.mean()

def simcse_loss(z1, z2, T=0.05):
    # Compute cosine similarity matrix
    sim_matrix = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1) / T

    # Labels are 0, 1, ..., batch_size - 1
    labels = torch.arange(sim_matrix.size(0), device=z1.device)

    # Cross entropy loss
    loss = nn.CrossEntropyLoss()(sim_matrix, labels)
    return loss

def dcl_loss(x1, x2, T=0.05):

    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)
    
    matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs) / T
    
    sim_matrix = torch.exp(matrix)
    pos = torch.diag(sim_matrix)
    p_ij = pos / (sim_matrix.sum(dim=1)-pos)
    
    loss = - torch.log(p_ij)
    
    return loss.mean()

def simple_cl_loss(x1, x2, T=0.05):

    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)
    
    matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    pos_sim = torch.diag(matrix)
    neg_sim = (matrix.sum(dim=1))/(x1.size(0)-1)
    
    loss = - pos_sim + neg_sim
    
    return loss.mean()

def dhel_loss(x1, x2, T=0.05):
    """
    Decoupled Hyperspherical Energy Loss (DHEL) implementation
    """
    z_1 = F.normalize(x1, p=2, dim=1)
    z_2 = F.normalize(x2, p=2, dim=1)
    
    sim_matrix_anchor = torch.exp(torch.mm(z_1, z_1.t()) / T)
    mask = torch.eye(z_1.size(0), device=z_1.device).bool()
    sim_matrix_anchor = sim_matrix_anchor.masked_fill(mask, 0)
    
    pos_sim = torch.exp(torch.sum(z_1 * z_2, dim=-1) / T)
    
    sim_matrix_pos = torch.exp(torch.mm(z_2, z_2.t()) / T)
    sim_matrix_pos = sim_matrix_pos.masked_fill(mask, 0)
    
    loss = -torch.log(pos_sim / (sim_matrix_anchor.sum(dim=-1) * sim_matrix_pos.sum(dim=-1))).mean()
    
    return loss

def gkcl_loss(x1, x2, T=0.05):
    """
    Gaussian-Kernel Contrastive Loss (GKCL) implementation
    """
    def gaussian_kernel(x, t):
        pairwise_distances = torch.pdist(x, p=2)
        return pairwise_distances.pow(2).mul(-t).exp().mean()
    
    def align_gaussian(x, y, t):
        pairwise_distances = (x - y).norm(p=2, dim=1)
        return pairwise_distances.pow(2).mul(-t).exp().mean()
    
    energy = (gaussian_kernel(x1, T).mean() + gaussian_kernel(x2, T).mean())
    alignment = 2 * align_gaussian(x1, x2, T)
    
    loss = -alignment + 128 * energy
    
    return loss

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
    def forward(self, z1, z2):

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
    
