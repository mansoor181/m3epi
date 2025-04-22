"""
Losses for M3EPI.
Includes:
  • supervised BCE losses
  • classic InfoNCE (intra / inter)
  • ReGCL‑style Gradient‑Weighted InfoNCE
The GW‑NCE part is identical to the original ReGCL implementation,
with three modifications so it works for bipartite graphs (|Ag| ≠ |Ab|):
  (1)  _build_adjacency handles rectangular adjacency
  (2)  _semi_loss falls back to edge‑based positives when |z1| ≠ |z2|
  (3)  gradient_weighted_nce averages the two semi‑losses safely when
       their lengths differ.
Everything else is a line‑for‑line port; **no in‑place ops** on tensors
that require gradients are performed.
"""
from __future__ import annotations
import torch, math
import torch.nn.functional as F
from torch_geometric.utils import to_scipy_sparse_matrix


# # Supervised losses
# def binary_cross_entropy(pred, target):
#     return F.binary_cross_entropy(pred, target)

# ------------------------------------------------------------------
# Supervised BCE with class balancing (no logits required)
# ------------------------------------------------------------------
def binary_cross_entropy(pred: torch.Tensor,
                         target: torch.Tensor,
                         pos_weight=None) -> torch.Tensor:
    """
    BCE on sigmoid probabilities **with optional positive‑class weight**.

    Args
    ----
    pred : torch.Tensor
        Probabilities in [0,1], shape [N] or [N,1].
    target : torch.Tensor
        Ground‑truth 0/1 labels, same shape as `pred`.
    pos_weight : int | float | torch.Tensor | None
        Ratio (#neg / #pos) or any scalar weight > 1.  If a Python
        scalar is passed (as in the YAML), it is promoted to a tensor
        on the right device.  If `None`, ordinary BCE is used.

    Returns
    -------
    torch.Tensor  –  scalar mean loss.
    """
    if pos_weight is None:
        # Standard BCE
        return F.binary_cross_entropy(pred, target)

    # Ensure tensor & device
    if not torch.is_tensor(pos_weight):
        pos_weight = torch.tensor(float(pos_weight),
                                  dtype=pred.dtype,
                                  device=pred.device)

    # Build element‑wise weight mask: w_pos for label 1, 1.0 for label 0
    weight = torch.ones_like(target, dtype=pred.dtype, device=pred.device)
    weight = torch.where(target == 1, pos_weight, weight)

    return F.binary_cross_entropy(pred, target, weight=weight)

# ==========================================
# Contrastive losses
# ==========================================

def ntxent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Classic SimCLR NT-Xent: treats each i in z1 as anchor, i in z2 as positive,
    all other j≠i in z2 as negatives.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.mm(z1, z2.t()) / temperature      # [B,B]
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)

def intra_nce_loss_graph(h: torch.Tensor, y: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """
    Intra‐graph InfoNCE: for each binding residue i (y[i]==1) as anchor,
    positives are other binding residues, negatives are all non‐binding.
    """
    pos = (y == 1).nonzero(as_tuple=True)[0]
    neg = (y == 0).nonzero(as_tuple=True)[0]
    if pos.numel() == 0:
        return torch.tensor(0., device=h.device)
    h_norm = F.normalize(h, dim=1)
    loss = 0.0
    for i in pos:
        anchor = h_norm[i].unsqueeze(0)                        # [1,D]
        pos_feats = h_norm[pos[pos != i]]                      # [P-1, D]
        neg_feats = h_norm[neg]                                # [Q,   D]
        if pos_feats.numel() == 0:                             # no other pos
            continue
        sim_pos = torch.exp((anchor @ pos_feats.t()) / tau)    # [1, P-1]
        sim_neg = torch.exp((anchor @ neg_feats.t()) / tau)    # [1, Q]
        num = sim_pos.sum()
        den = num + sim_neg.sum()
        loss += -torch.log(num / (den + 1e-8))
    return loss / pos.numel()

def intra_nce_loss(
    ag_h: torch.Tensor, ab_h: torch.Tensor,
    y_ag: torch.Tensor, y_ab: torch.Tensor,
    tau: float = 0.1
) -> torch.Tensor:
    """Sum of intra‐graph losses on antigen and antibody."""
    return intra_nce_loss_graph(ag_h, y_ag, tau) + intra_nce_loss_graph(ab_h, y_ab, tau)

def inter_nce_loss(
    ag_h: torch.Tensor, ab_h: torch.Tensor,
    y_ag: torch.Tensor, y_ab: torch.Tensor,
    tau: float = 0.1
) -> torch.Tensor:
    """
    Inter‐graph InfoNCE:
      – A→B: each binding node in Ag as anchor, positives are binding nodes in Ab,
        negatives are all non‐binding in both.
      – B→A: symmetrically.
    """
    pos_ag = (y_ag == 1).nonzero(as_tuple=True)[0]
    neg_ag = (y_ag == 0).nonzero(as_tuple=True)[0]
    pos_ab = (y_ab == 1).nonzero(as_tuple=True)[0]
    neg_ab = (y_ab == 0).nonzero(as_tuple=True)[0]

    ag_norm = F.normalize(ag_h, dim=1)
    ab_norm = F.normalize(ab_h, dim=1)

    def _one_direction(anchor_feats, pos_feats, neg_feats):
        if pos_feats.numel() == 0 or anchor_feats.numel() == 0:
            return torch.tensor(0., device=anchor_feats.device)
        loss = 0.0
        for i in range(anchor_feats.size(0)):
            a = anchor_feats[i].unsqueeze(0)                   # [1,D]
            sims_pos = torch.exp((a @ pos_feats.t()) / tau)    # [1, P]
            sims_neg = torch.exp((a @ neg_feats.t()) / tau)    # [1, Q]
            num = sims_pos.sum()
            den = num + sims_neg.sum()
            loss += -torch.log(num / (den + 1e-8))
        return loss / anchor_feats.size(0)

    # A→B
    A2B = _one_direction(
        ag_norm[pos_ag],             # anchors
        ab_norm[pos_ab],             # positives
        torch.cat([ag_norm[neg_ag], ab_norm[neg_ab]], dim=0)  # negatives
    )
    # B→A
    B2A = _one_direction(
        ab_norm[pos_ab],
        ag_norm[pos_ag],
        torch.cat([ag_norm[neg_ag], ab_norm[neg_ab]], dim=0)
    )

    return A2B + B2A



# ---------------------------------------------------------------------
# -------- ReGCL : Gradient‑weighted InfoNCE (view‑agnostic) ----------
# ---------------------------------------------------------------------

# ----------------------------------------------------------------------
# helpers – GPU helper & cosine‑sim utility
# ----------------------------------------------------------------------
def _device(t: torch.Tensor) -> torch.device:              # shorthand
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _to_dev(x: torch.Tensor) -> torch.Tensor:
    return x.to(_device(x))

def _sim(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    return z1 @ z2.t()                                     # [N,M]


# ----------------------------------------------------------------------
# adjacency builder that copes with rectangular bipartite edges
# ----------------------------------------------------------------------
@torch.no_grad()
def _build_adj(edge_index: torch.LongTensor,
               rows: int, cols: int) -> torch.Tensor:
    if rows == cols:                                       # square
        A = to_scipy_sparse_matrix(edge_index, num_nodes=rows).toarray()
        return torch.as_tensor(A, dtype=torch.float32, device=edge_index.device)
    else:                                                  # rectangular
        A = torch.zeros(rows, cols, dtype=torch.float32, device=edge_index.device)
        r, c = edge_index
        mask = (r < rows) & (c < cols)                     # safety
        A[r[mask], c[mask]] = 1.
        return A

# ----------------------------------------------------------------------
# gradient‑guided weight matrix (identical logic to ReGCL)
# ----------------------------------------------------------------------
def _get_W(z: torch.Tensor, edge_index: torch.LongTensor,
           mode: str, Pm: torch.Tensor,
           tau: float, cutrate: float, cutway: int) -> torch.Tensor:

    n, m = Pm.shape                                          # n rows, m cols
    dev  = z.device
    A    = _build_adj(edge_index, n, m)                      # dense adjacency

    deg  = (A.sum(1) + 1).sqrt().view(n, 1)                  # [n,1]
    P    = (Pm / (Pm.sum(1, keepdim=True) + 1e-12)).detach() # row‑norm
    diag = (torch.diag(P) if n == m else (P*A).sum(1))       # [n]
    P_bf = diag.view(n, 1).expand(n, n).detach()

    P = P / deg                                              # scale
    if n == m:
        P = P / deg.t()

    W = torch.zeros_like(P)

    if mode == 'between' and n == m:                         # original inter‑view
        P12 = torch.sigmoid(-(P - P.mean())/(P.std()+1e-6)) * A

        sum2 = (A*P).sum(0).view(n,1)
        P3   = (diag.view(n,1)-1)/(deg**2)
        P23  = torch.sigmoid((sum2+P3 - (sum2+P3).mean())/((sum2+P3).std()+1e-6))

        W5k  = 1/deg / deg.t()
        P5k  = torch.sigmoid((torch.abs((P_bf-1)*W5k) -
                              torch.abs((P_bf-1)*W5k).mean())/
                             (torch.abs((P_bf-1)*W5k).std()+1e-6)) * A
        P5   = P5k.sum(0).view(n,1)

        mask = A @ A
        Pi   = P * deg
        Pi   = Pi * mask
        sum4 = Pi.sum(0).view(n,1)

        W += 0.5*(P12 + P5k)
        W += 0.5*(torch.diag((sum2+P3).squeeze()) +
                  torch.diag((sum4+P5).squeeze()))
    else:                                                    # reflexive OR rectangular
        P12 = torch.sigmoid(-(P - P.mean())/(P.std()+1e-6)) * A
        W  += P12

    W = torch.where(W==0, torch.ones_like(W), W)             # keep grads
    return W


# ----------------------------------------------------------------------
# semi‑loss: one direction
# ----------------------------------------------------------------------
def _semi_loss(z1, z2, ei1, ei2, tau, cutrate, cutway):
    f          = lambda s: torch.exp(s / tau)
    R_sim      = f(_sim(z1, z1))                              # [N,N]
    B_sim      = f(_sim(z1, z2))                              # [N,M]
    R_weighted = R_sim * _to_dev(_get_W(z1, ei1, 'refl',
                                        R_sim, tau, cutrate, cutway))
    B_weighted = B_sim * _to_dev(_get_W(z1, ei2, 'between',
                                        B_sim, tau, cutrate, cutway))

    N, M = B_sim.shape
    if N == M:                                               # square (original case)
        pos = B_weighted.diag()
    else:                                                    # bipartite: edge‑based positives
        A   = _build_adj(ei2, N, M)
        pos = (B_weighted * A).sum(1)                        # [N]

    denom = R_weighted.sum(1) + B_weighted.sum(1) - \
            (R_weighted.diag() if N==M else 0.)
    ratio = torch.clamp(pos / (denom + 1e-12), min=1e-8)
    return -torch.log(ratio)                                 # [N]



def _batched_semi_loss(z1: torch.Tensor, z2: torch.Tensor,
                       tau: float, cutrate: float, cutway: int,
                       batch_size: int) -> torch.Tensor:
    """
    Batched variant for very large graphs (always rectangular logic).
    """
    losses = []
    exp = lambda m: torch.exp(m / tau)
    for st in range(0, z1.size(0), batch_size):
        ed   = min(st + batch_size, z1.size(0))
        R    = exp(_sim(z1[st:ed], z1))
        B    = exp(_sim(z1[st:ed], z2))
        pos  = B.diag()
        den  = R.sum(1) + B.sum(1) - B.diag()
        losses.append(-torch.log(pos / (den + 1e-12)))
    return torch.cat(losses)


# ----------------------------------------------------------------------
# public wrapper – now size‑aware
# ----------------------------------------------------------------------
def _transpose_bipartite(ei: torch.LongTensor) -> torch.LongTensor:
    """
    Swap row/col indices of a bipartite edge list.
    Works even when ei is empty.
    """
    if ei.numel() == 0:
        return ei.clone()
    return ei.flip(0)        # [[r],[c]] -> [[c],[r]]

def gradient_weighted_nce(z1: torch.Tensor, z2: torch.Tensor,
                          edge_index1: torch.LongTensor,
                          edge_index2: torch.LongTensor,
                          temperature: float,
                          cutrate: float, cutway: int,
                          mean: bool = True, batch_size: int = 0
                          ) -> torch.Tensor:
    """
    Full ReGCL loss (Eq 17) with a safeguard for bipartite graphs.

    • If |z1| == |z2|  ➜ use both directions   (identical to the paper)
    • If |z1| != |z2|  ➜ use only the forward direction
      (reverse direction would need the *other* graph’s intra‑edges,
       which we do not have; trying to reuse the same edges overflows).
    """

    same_size = z1.size(0) == z2.size(0)

    # ----- forward direction -------------------------------------------------
    if batch_size == 0:
        l1 = _semi_loss(z1, z2, edge_index1, edge_index2,
                        temperature, cutrate, cutway)
    else:
        l1 = _batched_semi_loss(z1, z2, temperature,
                                cutrate, cutway, batch_size)

    # ----- reverse direction (only if square) --------------------------------
    if same_size:
        if batch_size == 0:
            l2 = _semi_loss(z2, z1,
                            edge_index2 if edge_index2.size(0)==2 else
                            _transpose_bipartite(edge_index2),
                            edge_index1 if edge_index1.size(0)==2 else
                            _transpose_bipartite(edge_index1),
                            temperature, cutrate, cutway)
        else:
            l2 = _batched_semi_loss(z2, z1, temperature,
                                    cutrate, cutway, batch_size)
        loss = 0.5 * (l1 + l2)
    else:
        # bipartite → only forward direction available
        loss = l1

    return loss.mean() if mean else loss.sum()



# def gradient_weighted_nce(z1, z2, edge_index=None, tau=0.1):
#     """
#     Placeholder for Gradient-weighted NCE loss. Currently wraps into ntxent_loss;
#     replace body with full gradient-weighted implementation.
#     """
#     # normalize
#     z1_norm = F.normalize(z1, dim=1)
#     z2_norm = F.normalize(z2, dim=1)
#     # fallback to standard InfoNCE
#     return ntxent_loss(z1_norm, z2_norm, temperature=tau)


# ######### from WALLE ##############

# from typing import Optional, Union

# import torch
# import torch.nn.functional as F
# from torch import Tensor

# # --------------------
# # loss factory function
# # --------------------
# '''
# Loss function configuration format:
# {
#     "loss": [
#         {
#             "name": "badj_rec_loss",  # F.binary_cross_entropy
#             "w": 1.0,
#             "kwargs": {}
#         },
#         {
#             "name": "bipartite_edge_loss",
#             "w": 0.0003942821556421417,
#             "kwargs": {"thr": 40}
#         },
#         {
#             "name": "l2",
#             "w": 1e-2,
#             "kwargs": {}
#         }
#     ]
# }
# each loss term is a dict, with keys:
# - name: (str) loss function name
# - w: (float) weight for this loss term
# - kwargs: (dict) kwargs for this loss function
# '''


# def edge_index_bg_sum_loss(
#     edge_index_bg_pred: Tensor,
#     thr: Optional[float] = None,
# ) -> Tensor:
#     """ Calculate the bipartite-edge loss
#     Minimize this loss encourages the model to reconstruct the bipartite graph
#     with the same number of average edges as in the ground-truth graphs

#     Input is a single AbAg graph pair
#     Args:
#         edge_index_bg_pred (Tensor): predicted bipartite adjacency, shape (Nb, Ng)
#         thr: (float) threshold for the number of edges in the reconstructed bipartite graph
#     Returns:
#         loss (Tensor): sum of the difference between the number of edges in the reconstructed bipartite graph
#             and the average over the ground-truth bipartite graphs
#     """
#     thr = 40 if thr is None else thr
#     return torch.abs(torch.sum(edge_index_bg_pred) - thr)


# def edge_index_bg_rec_loss(
#     edge_index_bg_pred: Tensor,
#     edge_index_bg_true: Tensor,
#     weight_tensor: Union[Tensor, float],
#     reduction: str = 'none'
# ) -> Tensor:
#     """ Calculate interface edge reconstruction loss

#     Input is a single AbAg graph pair
#     Args:
#         edge_index_bg_pred: (Tensor) reconstructed bipartite adjacency matrix, shape (Nb, Ng), float  between 0, 1
#         edge_index_bg_true: (Tensor) ground-truth  bipartite adjacency matrix, shape (Nb, Ng), binary 0/1
#         weight_tensor: (Tensor) for balancing pos/neg samples, shape (Nb*Ng,)
#     """
#     device = edge_index_bg_pred.device

#     # if weight_tensor is a float or a scalar Tensor, i.e. positive edge weight,
#     # convert it to a tensor of the same shape as edge_index_bg_true
#     if isinstance(weight_tensor, (int, float)):  # Use a tuple of types for isinstance
#         weight_tensor = torch.tensor(weight_tensor, device=device)
#     if isinstance(weight_tensor, Tensor) and weight_tensor.ndim == 0:
#         weight_tensor = edge_index_bg_true * weight_tensor
#         weight_tensor[edge_index_bg_true == 0] = 1
#         weight_tensor = weight_tensor.float().to(device)

#     try:
#         assert edge_index_bg_pred.reshape(-1).shape == edge_index_bg_true.reshape(-1).shape == weight_tensor.reshape(-1).shape
#     except AssertionError as e:
#         raise ValueError(
#             "The following shapes should be the same but received:\n"
#             f"{edge_index_bg_pred.reshape(-1).shape=}\n"
#             f"{edge_index_bg_true.reshape(-1).shape=}\n"
#             f"{weight_tensor.reshape(-1).shape=}\n"
#         ) from e

#     return F.binary_cross_entropy(
#         edge_index_bg_pred.reshape(-1),  # shape (b*g, )
#         edge_index_bg_true.reshape(-1),  # shape (b*g, )
#         weight=weight_tensor.reshape(-1),
#         reduction=reduction,
#     )


# ######### from MIPE ##############


# import math
# import torch
# import torch.nn as nn
# from torch.nn.modules.loss import _Loss

# class NTXentLoss(_Loss):
#     '''
#         Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
#         Args:
#             z1, z2: Tensor of shape [batch_size, z_dim]
#             tau: Float. Usually in (0,1].
#             norm: Boolean. Whether to apply normlization.
#         '''

#     def __init__(self, norm: bool = True, tau: float = 0.5, uniformity_reg=0, variance_reg=0,
#                  covariance_reg=0) -> None:
#         super(NTXentLoss, self).__init__()
#         self.norm = norm
#         self.tau = tau
#         self.uniformity_reg = uniformity_reg
#         self.variance_reg = variance_reg
#         self.covariance_reg = covariance_reg

#     def std_loss(self,x):
#         std = torch.sqrt(x.var(dim=0) + 1e-04)
#         return torch.mean(torch.relu(1 - std))

#     def cov_loss(self,x):
#         batch_size, metric_dim = x.size()
#         x = x - x.mean(dim=0)
#         cov = (x.T @ x) / (batch_size - 1)
#         off_diag_cov = cov.flatten()[:-1].view(metric_dim - 1, metric_dim + 1)[:, 1:].flatten()
#         return off_diag_cov.pow_(2).sum() / metric_dim + 1e-8

#     def uniformity_loss(self,x1,x2,t=2):
#         sq_pdist_x1 = torch.pdist(x1, p=2).pow(2)
#         uniformity_x1 = sq_pdist_x1.mul(-t).exp().mean().log()
#         sq_pdist_x2 = torch.pdist(x2, p=2).pow(2)
#         uniformity_x2 = sq_pdist_x2.mul(-t).exp().mean().log()
#         return (uniformity_x1 + uniformity_x2) / 2

#     def forward(self, z1, z2, **kwargs):
#         batch_size, _ = z1.size()
#         sim_matrix = torch.einsum('ik,jk->ij', z1, z2)

#         if self.norm:
#             z1_abs = z1.norm(dim=1)
#             z2_abs = z2.norm(dim=1)
#             sim_matrix = sim_matrix / (torch.einsum('i,j->ij', z1_abs, z2_abs) + 1e-8)

#         sim_matrix = torch.exp(sim_matrix / self.tau)
#         pos_sim = torch.diagonal(sim_matrix)
#         loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim+ 1e-8)
#         loss=loss.float()
#         loss = - torch.log(loss+ 1e-8).mean()

#         if self.variance_reg > 0:
#             loss += self.variance_reg * (self.std_loss(z1) + self.std_loss(z2))
#         if self.covariance_reg > 0:
#             loss += self.covariance_reg * (self.cov_loss(z1) + self.cov_loss(z2))
#         if self.uniformity_reg > 0:
#             loss += self.uniformity_reg * self.uniformity_loss(z1, z2)
#         return loss

