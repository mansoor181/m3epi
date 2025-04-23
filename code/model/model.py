import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv

from model.CrossAttention import CrossAttentionBlock

class GraphLayer(nn.Module):
    """
    Single GNN layer: GCN/GAT/GIN + optional residual.
    """
    def __init__(self, in_dim, out_dim, model_type="GCN", use_residual=False):
        super().__init__()
        self.use_residual = use_residual
        # select convolution type
        if model_type == "GCN":
            self.conv = GCNConv(in_dim, out_dim)
        elif model_type == "GAT":
            self.conv = GATConv(in_dim, out_dim)
        elif model_type == "GIN":
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
            self.conv = GINConv(mlp)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        # residual projection if dims differ
        if use_residual and in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim)
        else:
            self.res_proj = None

    def forward(self, x, edge_index):
        out = self.conv(x, edge_index)
        if self.use_residual:
            res = x if self.res_proj is None else self.res_proj(x)
            out = out + res
        return F.relu(out)

class GraphEncoder(nn.Module):
    """
    Dynamic GNN encoder: uses `hidden_dims` list for layering.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dims: list,
                 output_dim: int,
                 model_type: str = "GCN",
                 use_residual: bool = False,
                 dropout: float = 0.5):
        super().__init__()
        # build dims from input -> hidden_dims -> output
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList([
            GraphLayer(dims[i], dims[i+1], model_type, use_residual)
            for i in range(len(dims)-1)
        ])
        self.dropout = dropout

    def forward(self, x, edge_index):
        # apply all but last layer with dropout
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # last layer, no dropout
        return self.layers[-1](x, edge_index)

"""
### Runtime impact
* **Dot** fastest, O(NM) mat‑mul.  
* **MLP** same memory as dot; a single extra FC pass.  
* **Attention** adds multi‑head projection, slight memory overhead but still O(NM).
"""

# ─────────────────────────────────────────────────────────────
# 1) Dot‑product decoder (formerly BipartiteDecoder)
# ─────────────────────────────────────────────────────────────
class DotDecoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.interaction = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.interaction)

    def forward(self, ag_embed, ab_embed):
        logits = ag_embed @ self.interaction @ ab_embed.t()
        return torch.sigmoid(logits)

# ─────────────────────────────────────────────────────────────
# 2) MLP‑based pairwise decoder
# ─────────────────────────────────────────────────────────────
class MLPDecoder(nn.Module):
    """Score each Ag–Ab pair by concatenating their embeddings into a tiny MLP."""
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, ag_embed, ab_embed):
        N, D = ag_embed.shape
        M    = ab_embed.shape[0]
        # expand to pairwise
        ag_exp = ag_embed.unsqueeze(1).expand(N, M, D)
        ab_exp = ab_embed.unsqueeze(0).expand(N, M, D)
        pair   = torch.cat([ag_exp, ab_exp], dim=-1)      # [N,M,2D]
        logits = self.net(pair).squeeze(-1)               # [N,M]
        return torch.sigmoid(logits)

# ─────────────────────────────────────────────────────────────
# 3) Cross‑Attention decoder
# ─────────────────────────────────────────────────────────────

class AttentionDecoder(nn.Module):
    """Cross‑attend each antigen node over all antibody nodes (multi‑head)."""
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = CrossAttentionBlock(dim, num_heads)

    def forward(self, ag_embed, ab_embed):
        # returns [N,M] attention‑score matrix
        return self.attn(ag_embed, ab_embed)
    

class M3EPI(nn.Module):
    def __init__(self, config):
        super().__init__()
        mt = config.model.name
        ur = config.model.use_residual
        dr = config.model.dropout
        print(f"[INFO] Model={mt} | residual={ur} | dropout={dr}")

        # two separate encoders
        self.ag_encoder = GraphEncoder(
            input_dim=config.model.encoder.antigen.input_dim,
            hidden_dims=config.model.encoder.antigen.hidden_dims,
            output_dim=config.model.encoder.antigen.output_dim,
            model_type=mt, use_residual=ur, dropout=dr
        )
        self.ab_encoder = GraphEncoder(
            input_dim=config.model.encoder.antibody.input_dim,
            hidden_dims=config.model.encoder.antibody.hidden_dims,
            output_dim=config.model.encoder.antibody.output_dim,
            model_type=mt, use_residual=ur, dropout=dr
        )

        # ─────────────────────────────────────────────────────
        # decoder dispatch
        dec_cfg = config.model.decoder
        dec_type = dec_cfg.type.lower()
        dim      = dec_cfg.interaction_dim

        if dec_type == "dot":
            self.decoder = DotDecoder(dim)
        elif dec_type == "mlp":
            self.decoder = MLPDecoder(dim, dec_cfg.mlp_hidden)
        elif dec_type == "attention":
            self.decoder = AttentionDecoder(dim, dec_cfg.heads)
        else:
            raise ValueError(f"Unknown decoder type: {dec_type}")

        self.threshold = dec_cfg.threshold
    # ─────────────────────────────────────────────────────────────
    def forward(self, batch):
        ag_x, ag_e = batch['x_g'], batch['edge_index_g']
        ab_x, ab_e = batch['x_b'], batch['edge_index_b']

        ag_emb = self.ag_encoder(ag_x, ag_e)
        ab_emb = self.ab_encoder(ab_x, ab_e)
        ip     = self.decoder(ag_emb, ab_emb)       # [N,M]
        # print(ip)
        epi_prob = ip.max(dim=1).values             # as before

        return {
            'ag_embed': ag_emb,
            'ab_embed': ab_emb,
            'interaction_probs': ip,
            'epitope_prob': epi_prob
        }


# class BipartiteDecoder(nn.Module):
#     """
#     Dot-product decoder for Ab-Ag bipartite edges.
#     """
#     def __init__(self, hidden_dim: int):
#         super().__init__()
#         self.interaction = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
#         nn.init.xavier_uniform_(self.interaction)

#     def forward(self, ag_embed, ab_embed):
#         logits = ag_embed @ self.interaction @ ab_embed.t()
#         return torch.sigmoid(logits)

# class M3EPI(nn.Module):
#     """
#     Integrates two GraphEncoders (antigen/antibody) + BipartiteDecoder.
#     """
#     def __init__(self, config):
#         super().__init__()
#         mt = config.model.name
#         ur = config.model.use_residual
#         dr = config.model.dropout
#         print(f"[INFO] Model={mt} | residual={ur} | dropout={dr}")

#         # antigen encoder
#         self.ag_encoder = GraphEncoder(
#             input_dim=config.model.encoder.antigen.input_dim,
#             hidden_dims=config.model.encoder.antigen.hidden_dims,
#             output_dim=config.model.encoder.antigen.output_dim,
#             model_type=mt,
#             use_residual=ur,
#             dropout=dr
#         )
#         # antibody encoder
#         self.ab_encoder = GraphEncoder(
#             input_dim=config.model.encoder.antibody.input_dim,
#             hidden_dims=config.model.encoder.antibody.hidden_dims,
#             output_dim=config.model.encoder.antibody.output_dim,
#             model_type=mt,
#             use_residual=ur,
#             dropout=dr
#         )
#         # decoder
#         self.decoder = BipartiteDecoder(config.model.decoder.interaction_dim)
#         self.threshold = config.model.decoder.threshold

#     def forward(self, batch):
#         ag_x, ag_e = batch['x_g'], batch['edge_index_g']
#         ab_x, ab_e = batch['x_b'], batch['edge_index_b']

#         ag_emb = self.ag_encoder(ag_x, ag_e)
#         ab_emb = self.ab_encoder(ab_x, ab_e)
#         ip = self.decoder(ag_emb, ab_emb)
#         epi_prob = ip.max(dim=1).values
#         return {
#             'ag_embed': ag_emb,
#             'ab_embed': ab_emb,
#             'interaction_probs': ip,
#             'epitope_prob': epi_prob
#         }




# """
# TODO: [mansoor]
# - use pytorch geometric instead of torchdrug
# - define and create two base graph models: GCN and GAT
# - define encoder and decoder:
#     - encoder has two base graph modules (either GCN or GAT) that take as input 
#     antigen (m nodes) and antibody (n nodes) graphs respectively (GCN-1 and GCN2 or GAT1 or GAT2)
#     - the processed/encoded embeddings are then passed to decoder
#     where dot product between embeddings is calculated returning m x n probability matrix
#     - the predicted probabilities for each pair of nodes in the antigen and antibody graph
#      - These probabilities are then thresholded to 0 or 1, representing an edge or non-edge. 
#      Then, the corresponding nodes from the antigen graph with at least an edge (bipartite link)
#        with the antibody graph is assigned as epitope (binding site residue).
#     # - make number of layers an hyperparameter and create

# - use hydra for config.yaml which uses loss.yaml, callbacks.yaml, wandb.yaml, model.yaml and hparams.yaml
# """



# import torch
# import torch.nn as nn
# from torch_geometric.nn import GCNConv, GATConv
# import torch.nn.functional as F

# class GraphEncoder(nn.Module):
#     def __init__(self, input_dim, hidden_dims, output_dim, model_type="GCN"):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         dims = [input_dim] + hidden_dims + [output_dim]
        
#         for i in range(len(dims)-1):
#             if model_type == "GCN":
#                 self.layers.append(GCNConv(dims[i], dims[i+1]))
#             else:
#                 self.layers.append(GATConv(dims[i], dims[i+1]))
                
#     def forward(self, x, edge_index):
#         for layer in self.layers[:-1]:
#             x = F.relu(layer(x, edge_index))
#             x = F.dropout(x, p=0.5, training=self.training)
#         return self.layers[-1](x, edge_index)

# class BipartiteDecoder(nn.Module):
#     """
#     - return a list of [Ng, Nb] matrices for each graph in the batch
#     """
#     def __init__(self, hidden_dim):
#         super().__init__()
#         self.interaction = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
#         nn.init.xavier_uniform_(self.interaction)
        
#     def forward(self, ag_embed, ab_embed):
#         # Return interaction probability matrix
#         # Ab-Ag bipartite edges        => edge_index_bg
#         return torch.sigmoid(torch.mm(torch.mm(ag_embed, self.interaction), ab_embed.t()))

# class M3EPI(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         # Encoders
#         self.ag_encoder = GraphEncoder(
#             input_dim=config.model.encoder.antigen.input_dim,
#             hidden_dims=config.model.encoder.antigen.hidden_dims,
#             output_dim=config.model.encoder.antigen.output_dim,
#             model_type=config.model.name
#         )
        
#         self.ab_encoder = GraphEncoder(
#             input_dim=config.model.encoder.antibody.input_dim, 
#             hidden_dims=config.model.encoder.antibody.hidden_dims,
#             output_dim=config.model.encoder.antibody.output_dim,
#             model_type=config.model.name
#         )
        
#         # Decoder
#         self.decoder = BipartiteDecoder(config.model.decoder.interaction_dim)
#         self.threshold = config.model.decoder.threshold
        
#     def forward(self, batch):
#         # Process antigen
#         ag_embed = self.ag_encoder(
#             batch["x_g"],
#             batch["edge_index_g"]
#         )
        
#         # Process antibody  
#         ab_embed = self.ab_encoder(
#             batch["x_b"],
#             batch["edge_index_b"]
#         )
        
#         # Calculate interaction probabilities
#         interaction_probs = self.decoder(ag_embed, ab_embed) 
        
#         # Get epitope predictions
#         # epitope_pred = (interaction_probs > self.threshold).any(dim=1)
#         epitope_prob = interaction_probs.max(dim=1).values  # [m], probability each AG node binds any AB node
        
#         return {
#             'ag_embed': ag_embed,
#             'ab_embed': ab_embed,
#             'interaction_probs': interaction_probs,
#             'epitope_prob': epitope_prob
#         }



# ################## from WALLE ##################

# # logging
# import logging
# # basic
# import os
# import os.path as osp
# import re
# import sys
# from pathlib import Path
# from pprint import pprint
# from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional,
#                     Set, Tuple, Union)

# import numpy as np
# import pandas as pd
# # torch tools
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# # pyg tools
# import torch_geometric as tg
# import torch_geometric.transforms as T
# import torch_scatter as ts
# from omegaconf import DictConfig, OmegaConf
# from torch import Tensor
# from torch_geometric.data import Batch as PygBatch
# from torch_geometric.data import Data as PygData
# from torch_geometric.data import Dataset as PygDataset
# from torch_geometric.datasets import TUDataset
# from torch_geometric.loader import DataLoader as PygDataLoader
# from torch_geometric.nn import GATConv, GCNConv
# from torch_geometric.utils import to_dense_adj, to_dense_batch, to_undirected

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s {%(pathname)s:%(lineno)d} [%(threadName)s] [%(levelname)s] %(name)s - %(message)s",
#     datefmt="%H:%M:%S",
# )
# # custom
# from asep.data.asepv1_dataset import AsEPv1Dataset


# class PyGAbAgIntGAE(nn.Module):
#     def __init__(
#         self,
#         input_ab_dim: int,  # input dims
#         input_ag_dim: int,  # input dims
#         dim_list: List[int],  # dims (length = len(act_list) + 1)
#         act_list: List[str],  # acts
#         decoder: Optional[Dict] = None,  # layer type
#         try_gpu: bool = True,  # use gpu
#         input_ab_act: str = "relu",  # input activation
#         input_ag_act: str = "relu",  # input activation
#     ):
#         super().__init__()
#         decoder = (
#             {
#                 "name": "inner_prod",
#             }
#             if decoder is None
#             else decoder
#         )
#         self.device = torch.device(
#             "cuda" if try_gpu and torch.cuda.is_available() else "cpu"
#         )

#         # add to hparams
#         self.hparams = {
#             "input_ab_dim": input_ab_dim,
#             "input_ag_dim": input_ag_dim,
#             "dim_list": dim_list,
#             "act_list": act_list,
#             "decoder": decoder,
#         }
#         # self._args_sanity_check()

#         # encoder
#         _default_conv_kwargs = {"normalize": True}  # DO NOT set cache to True
#         self.B_encoder_block = self._create_a_encoder_block(
#             node_feat_name="x_b",
#             edge_index_name="edge_index_b",
#             input_dim=input_ab_dim,
#             input_act=input_ab_act,
#             dim_list=dim_list,
#             act_list=act_list,
#             gcn_kwargs=_default_conv_kwargs,
#         ).to(self.device)
#         self.G_encoder_block = self._create_a_encoder_block(
#             node_feat_name="x_g",
#             edge_index_name="edge_index_g",
#             input_dim=input_ag_dim,
#             input_act=input_ag_act,
#             dim_list=dim_list,
#             act_list=act_list,
#             gcn_kwargs=_default_conv_kwargs,
#         ).to(self.device)

#         # decoder attr placeholder
#         self.decoder = self.decoder_factory(self.hparams["decoder"])
#         self._dc_func: Callable = self.decoder_func_factory(self.hparams["decoder"])

#     def _args_sanity_check(self):
#         # 1. if dim_list or act_list is provided, assert dim_list length is equal to act_list length + 1
#         if self.hparams["dim_list"] is not None or self.hparams["act_list"] is not None:
#             try:
#                 assert (
#                     len(self.hparams["dim_list"]) == len(self.hparams["act_list"]) + 1
#                 ), (
#                     f"dim_list length must be equal to act_list length + 1, "
#                     f"got dim_list {self.hparams['dim_list']} and act_list {self.hparams['act_list']}"
#                 )
#             except AssertionError as e:
#                 raise ValueError(
#                     "dim_list length must be equal to act_list length + 1, "
#                 ) from e
#         # 2. if decoder is provided, assert decoder name is in ['inner_prod', 'fc', 'bilinear']
#         if self.hparams["decoder"] is not None:
#             try:
#                 assert isinstance(self.hparams["decoder"], dict)
#             except AssertionError as e:
#                 raise TypeError(
#                     f"decoder must be a dict, got {self.hparams['decoder']}"
#                 ) from e
#             try:
#                 assert self.hparams["decoder"]["name"] in (
#                     "inner_prod",
#                     "fc",
#                     "bilinear",
#                 )
#             except AssertionError as e:
#                 raise ValueError(
#                     f"decoder {self.hparams['decoder']['name']} not supported, "
#                     "please choose from ['inner_prod', 'fc', 'bilinear']"
#                 ) from e

#     def _create_a_encoder_block(
#         self,
#         node_feat_name: str,
#         edge_index_name: str,
#         input_dim: int,
#         input_act: str,
#         dim_list: List[int],
#         act_list: List[str],
#         gcn_kwargs: Dict[str, Any],
#     ):
#         def _create_linear_layer(i: int, in_channels: int, out_channels: int) -> tuple:
#             if i == 0:
#                 mapping = f"{node_feat_name} -> {node_feat_name}_{i+1}"
#             else:
#                 mapping = f"{node_feat_name}_{i} -> {node_feat_name}_{i+1}"
#             # print(mapping)

#             return (
#                 nn.Linear(in_channels, out_channels),
#                 mapping,
#             )

#         def _create_act_layer(act_name: Optional[str]) -> nn.Module:
#             # assert act_name is either None or str
#             assert act_name is None or isinstance(
#                 act_name, str
#             ), f"act_name must be None or str, got {act_name}"

#             if act_name is None:
#                 # return identity
#                 return (nn.Identity(),)
#             elif act_name.lower() == "relu":
#                 return (nn.ReLU(inplace=True),)
#             elif act_name.lower() == "leakyrelu":
#                 return (nn.LeakyReLU(inplace=True),)
#             else:
#                 raise ValueError(
#                     f"activation {act_name} not supported, please choose from ['relu', 'leakyrelu', None]"
#                 )

#         modules = [
#             _create_linear_layer(0, input_dim, dim_list[0]),
#             _create_act_layer(input_act),
#         ]

#         for i in range(len(dim_list) - 1):
#             modules.extend(
#                 [
#                     _create_linear_layer(
#                         i + 1, dim_list[i], dim_list[i + 1]
#                     ),  # i+1 increment due to the input layer
#                     _create_act_layer(act_list[i]),
#                 ]
#             )

#         return tg.nn.Sequential(
#             input_args=f"{node_feat_name}, {edge_index_name}", modules=modules
#         )

#     def _init_fc_decoder(self, decoder) -> nn.Sequential:
#         bias: bool = decoder["bias"]
#         dp: Optional[float] = decoder["dropout"]

#         dc = nn.ModuleList()

#         # dropout
#         if dp is not None:
#             dc.append(nn.Dropout(dp))
#         # fc linear
#         dc.append(
#             nn.Linear(
#                 in_features=self.hparams["dim_list"][-1] * 2, out_features=1, bias=bias
#             )
#         )
#         # make it a sequential
#         dc = nn.Sequential(*dc)

#         return dc

#     def encode(self, batch: PygBatch) -> Tuple[Tensor, Tensor]:
#         """
#         Args:
#             batch: (PygBatch) batched data returned by PyG DataLoader
#         Returns:
#             B_z: (Tensor) shape (Nb, C)
#             G_z: (Tensor) shape (Ng, C)
#         """
#         batch = batch.to(self.device)
#         B_z = self.B_encoder_block(batch.x_b, batch.edge_index_b)  # (Nb, C)
#         G_z = self.G_encoder_block(batch.x_g, batch.edge_index_g)  # (Ng, C)

#         return B_z, G_z

#     def decoder_factory(
#         self, decoder_dict: Dict[str, str]
#     ) -> Union[nn.Module, nn.Parameter, None]:
#         name = decoder_dict["name"]
#         if name == "bilinear":
#             init_method = decoder_dict.get("init_method", "kaiming_normal_")
#             decoder = nn.Parameter(
#                 data=torch.empty(
#                     self.hparams["dim_list"][-1], self.hparams["dim_list"][-1]
#                 ),
#                 requires_grad=True,
#             )
#             torch.nn.init.__dict__[init_method](decoder)
#             return decoder
#         elif name == "fc":
#             return self._init_fc_decoder(decoder_dict)
#         elif name == "inner_prod":
#             return

#     def decoder_func_factory(self, decoder_dict: Dict[str, str]) -> Callable:
#         name = decoder_dict["name"]
#         if name == "bilinear":
#             return lambda b_z, g_z: b_z @ self.decoder @ g_z.t()
#         elif name == "fc":
#             def _fc_runner(b_z: Tensor, g_z: Tensor) -> Tensor:
#                 """
#                 # (Nb, Ng, C*2)) -> (Nb, Ng, 1)
#                 # h = torch.cat([
#                 #         z_ab.unsqueeze(1).tile((1, z_ag.size(0), 1)),  # (Nb, 1, C) -> (Nb, Ng, C)
#                 #         z_ag.unsqueeze(0).tile((z_ab.size(0), 1, 1)),  # (1, Ng, C) -> (Nb, Ng, C)
#                 #     ], dim=-1)
#                 """
#                 h = torch.cat(
#                     [
#                         b_z.unsqueeze(1).expand(
#                             -1, g_z.size(0), -1
#                         ),  # (Nb, 1, C) -> (Nb, Ng, C)
#                         g_z.unsqueeze(0).expand(
#                             b_z.size(0), -1, -1
#                         ),  # (1, Ng, C) -> (Nb, Ng, C)
#                     ],
#                     dim=-1,
#                 )
#                 # (Nb, Ng, C*2) -> (Nb, Ng, 1)
#                 h = self.decoder(h)
#                 return h.squeeze(-1)  # (Nb, Ng, 1) -> (Nb, Ng)

#             return _fc_runner
#         elif name == "inner_prod":
#             return lambda b_z, g_z: b_z @ g_z.t()

#     def decode(
#         self, B_z: Tensor, G_z: Tensor, batch: PygBatch
#     ) -> Tuple[Tensor, Tensor]:
#         """
#         Inner Product Decoder

#         Args:
#             B_z: (Tensor)  shape (Nb, dim_latent)
#             G_z: (Tensor)  shape (Ng, dim_latent)

#         Returns:
#             A_reconstruct: (Tensor) shape (B, G)
#                 reconstructed bipartite adjacency matrix
#         """
#         # move batch to device
#         batch = batch.to(self.device)

#         edge_index_bg_pred = []
#         edge_index_bg_true = []

#         # dense bipartite edge index
#         edge_index_bg_dense = torch.zeros(batch.x_b.shape[0], batch.x_g.shape[0]).to(
#             self.device
#         )
#         edge_index_bg_dense[batch.edge_index_bg[0], batch.edge_index_bg[1]] = 1

#         # get graph sizes (number of nodes) in the batch, used to slice the dense bipartite edge index
#         node2graph_idx = torch.stack(
#             [
#                 torch.cumsum(
#                     torch.cat(
#                         [
#                             torch.zeros(1).long().to(self.device),
#                             batch.x_b_batch.bincount(),
#                         ]
#                     ),
#                     dim=0,
#                 ),  # (Nb+1, ) CDR     nodes
#                 torch.cumsum(
#                     torch.cat(
#                         [
#                             torch.zeros(1).long().to(self.device),
#                             batch.x_g_batch.bincount(),
#                         ]
#                     ),
#                     dim=0,
#                 ),  # (Ng+1, ) antigen nodes
#             ],
#             dim=0,
#         )

#         for i in range(batch.num_graphs):
#             edge_index_bg_pred.append(
#                 F.sigmoid(
#                     self._dc_func(
#                         b_z=B_z[batch.x_b_batch == i], g_z=G_z[batch.x_g_batch == i]
#                     )
#                 )
#             )  # Tensor (Nb, Ng)
#             edge_index_bg_true.append(
#                 edge_index_bg_dense[
#                     node2graph_idx[0, i] : node2graph_idx[0, i + 1],
#                     node2graph_idx[1, i] : node2graph_idx[1, i + 1],
#                 ]
#             )  # Tensor (Nb, Ng)

#         return edge_index_bg_pred, edge_index_bg_true

#     def forward(self, batch: PygBatch) -> Dict[str, Union[int, Tensor]]:
#         # device
#         batch = batch.to(self.device)
#         # encode
#         z_ab, z_ag = self.encode(batch)  # (Nb, C), (Ng, C)
#         # decode
#         edge_index_bg_pred, edge_index_bg_true = self.decode(z_ab, z_ag, batch)

#         return {
#             "abdbid": batch.abdbid,  # List[str]
#             "edge_index_bg_pred": edge_index_bg_pred,  # List[Tensor (Nb, Ng)]
#             "edge_index_bg_true": edge_index_bg_true,  # List[Tensor (Nb, Ng)]
#         }


# # a linear version of the model
# class LinearAbAgIntGAE(nn.Module):
#     def __init__(
#         self,
#         input_ab_dim: int,  # input dims
#         input_ag_dim: int,  # input dims
#         dim_list: List[int],  # dims (length = len(act_list) + 1)
#         act_list: List[str],  # acts
#         decoder: Optional[Dict] = None,  # layer type
#         try_gpu: bool = True,  # use gpu
#         input_ab_act: str = "relu",  # input activation
#         input_ag_act: str = "relu",  # input activation
#     ):
#         super().__init__()
#         decoder = (
#             {
#                 "name": "inner_prod",
#             }
#             if decoder is None
#             else decoder
#         )
#         self.device = torch.device(
#             "cuda" if try_gpu and torch.cuda.is_available() else "cpu"
#         )

#         # add to hparams
#         self.hparams = {
#             "input_ab_dim": input_ab_dim,
#             "input_ag_dim": input_ag_dim,
#             "dim_list": dim_list,
#             "act_list": act_list,
#             "decoder": decoder,
#         }
#         # self._args_sanity_check()

#         # encoder
#         self.B_encoder_block = self._create_a_encoder_block(
#             node_feat_name="x_b",
#             input_dim=input_ab_dim,
#             input_act=input_ab_act,
#             dim_list=dim_list,
#             act_list=act_list,
#         ).to(self.device)
#         self.G_encoder_block = self._create_a_encoder_block(
#             node_feat_name="x_g",
#             input_dim=input_ag_dim,
#             input_act=input_ag_act,
#             dim_list=dim_list,
#             act_list=act_list,
#         ).to(self.device)

#         # decoder attr placeholder
#         self.decoder = self.decoder_factory(self.hparams["decoder"])
#         self._dc_func: Callable = self.decoder_func_factory(self.hparams["decoder"])

#     def _args_sanity_check(self):
#         # 1. if dim_list or act_list is provided, assert dim_list length is equal to act_list length + 1
#         if self.hparams["dim_list"] is not None or self.hparams["act_list"] is not None:
#             try:
#                 assert (
#                     len(self.hparams["dim_list"]) == len(self.hparams["act_list"]) + 1
#                 ), (
#                     f"dim_list length must be equal to act_list length + 1, "
#                     f"got dim_list {self.hparams['dim_list']} and act_list {self.hparams['act_list']}"
#                 )
#             except AssertionError as e:
#                 raise ValueError(
#                     "dim_list length must be equal to act_list length + 1, "
#                 ) from e
#         # 2. if decoder is provided, assert decoder name is in ['inner_prod', 'fc', 'bilinear']
#         if self.hparams["decoder"] is not None:
#             try:
#                 assert isinstance(self.hparams["decoder"], Union[dict, DictConfig])
#             except AssertionError as e:
#                 raise TypeError(
#                     f"decoder must be a dict, got {self.hparams['decoder']}"
#                 ) from e
#             try:
#                 assert self.hparams["decoder"]["name"] in (
#                     "inner_prod",
#                     "fc",
#                     "bilinear",
#                 )
#             except AssertionError as e:
#                 raise ValueError(
#                     f"decoder {self.hparams['decoder']['name']} not supported, "
#                     "please choose from ['inner_prod', 'fc', 'bilinear']"
#                 ) from e

#     def _create_a_encoder_block(
#         self,
#         node_feat_name: str,
#         input_dim: int,
#         input_act: str,
#         dim_list: List[int],
#         act_list: List[str],
#     ):
#         def _create_linear_layer(i: int, in_channels: int, out_channels: int) -> tuple:
#             if i == 0:
#                 mapping = f"{node_feat_name} -> {node_feat_name}_{i+1}"
#             else:
#                 mapping = f"{node_feat_name}_{i} -> {node_feat_name}_{i+1}"
#             # print(mapping)

#             return (
#                 nn.Linear(in_channels, out_channels),
#                 mapping,
#             )

#         def _create_act_layer(act_name: Optional[str]) -> nn.Module:
#             # assert act_name is either None or str
#             assert act_name is None or isinstance(
#                 act_name, str
#             ), f"act_name must be None or str, got {act_name}"

#             if act_name is None:
#                 return (nn.Identity(),)
#             elif act_name.lower() == "relu":
#                 return (nn.ReLU(inplace=True),)
#             elif act_name.lower() == "leakyrelu":
#                 return (nn.LeakyReLU(inplace=True),)
#             else:
#                 raise ValueError(
#                     f"activation {act_name} not supported, please choose from ['relu', 'leakyrelu', None]"
#                 )

#         modules = [
#             _create_linear_layer(0, input_dim, dim_list[0]),  # First layer
#             _create_act_layer(input_act),
#         ]

#         for i in range(len(dim_list) - 1):  # Additional layers
#             modules.extend(
#                 [
#                     _create_linear_layer(
#                         i + 1, dim_list[i], dim_list[i + 1]
#                     ),  # i+1 increment due to the input layer
#                     _create_act_layer(act_list[i]),
#                 ]
#             )

#         return tg.nn.Sequential(input_args=f"{node_feat_name}", modules=modules)

#     def _init_fc_decoder(self, decoder) -> nn.Sequential:
#         bias: bool = decoder["bias"]
#         dp: Optional[float] = decoder["dropout"]

#         dc = nn.ModuleList()

#         # dropout
#         if dp is not None:
#             dc.append(nn.Dropout(dp))
#         # fc linear
#         dc.append(
#             nn.Linear(
#                 in_features=self.hparams["dim_list"][-1] * 2, out_features=1, bias=bias
#             )
#         )
#         # make it a sequential
#         dc = nn.Sequential(*dc)

#         return dc

#     def encode(self, batch: PygBatch) -> Tuple[Tensor, Tensor]:
#         """
#         Args:
#             batch: (PygBatch) batched data returned by PyG DataLoader
#         Returns:
#             B_z: (Tensor) shape (Nb, C)
#             G_z: (Tensor) shape (Ng, C)
#         """
#         batch = batch.to(self.device)
#         B_z = self.B_encoder_block(batch.x_b)  # , batch.edge_index_b)  # (Nb, C)
#         G_z = self.G_encoder_block(batch.x_g)  # , batch.edge_index_g)  # (Ng, C)

#         return B_z, G_z

#     def decoder_factory(
#         self, decoder_dict: Dict[str, str]
#     ) -> Union[nn.Module, nn.Parameter, None]:
#         name = decoder_dict["name"]

#         if name == "bilinear":
#             init_method = decoder_dict.get("init_method", "kaiming_normal_")
#             decoder = nn.Parameter(
#                 data=torch.empty(
#                     self.hparams["dim_list"][-1], self.hparams["dim_list"][-1]
#                 ),
#                 requires_grad=True,
#             )
#             torch.nn.init.__dict__[init_method](decoder)
#             return decoder

#         elif name == "fc":
#             return self._init_fc_decoder(decoder_dict)

#         elif name == "inner_prod":
#             return

#     def decoder_func_factory(self, decoder_dict: Dict[str, str]) -> Callable:
#         name = decoder_dict["name"]

#         if name == "bilinear":
#             return lambda b_z, g_z: b_z @ self.decoder @ g_z.t()

#         elif name == "fc":

#             def _fc_runner(b_z: Tensor, g_z: Tensor) -> Tensor:
#                 """
#                 # (Nb, Ng, C*2)) -> (Nb, Ng, 1)
#                 # h = torch.cat([
#                 #         z_ab.unsqueeze(1).tile((1, z_ag.size(0), 1)),  # (Nb, 1, C) -> (Nb, Ng, C)
#                 #         z_ag.unsqueeze(0).tile((z_ab.size(0), 1, 1)),  # (1, Ng, C) -> (Nb, Ng, C)
#                 #     ], dim=-1)
#                 """
#                 h = torch.cat(
#                     [
#                         b_z.unsqueeze(1).expand(
#                             -1, g_z.size(0), -1
#                         ),  # (Nb, 1, C) -> (Nb, Ng, C)
#                         g_z.unsqueeze(0).expand(
#                             b_z.size(0), -1, -1
#                         ),  # (1, Ng, C) -> (Nb, Ng, C)
#                     ],
#                     dim=-1,
#                 )
#                 # (Nb, Ng, C*2) -> (Nb, Ng, 1)
#                 h = self.decoder(h)
#                 return h.squeeze(-1)  # (Nb, Ng, 1) -> (Nb, Ng)

#             return _fc_runner

#         elif name == "inner_prod":
#             return lambda b_z, g_z: b_z @ g_z.t()

#     def decode(
#         self, B_z: Tensor, G_z: Tensor, batch: PygBatch
#     ) -> Tuple[Tensor, Tensor]:
#         """
#         Inner Product Decoder

#         Args:
#             z_ab: (Tensor)  shape (Nb, dim_latent)
#             z_ag: (Tensor)  shape (Ng, dim_latent)

#         Returns:
#             A_reconstruct: (Tensor) shape (B, G)
#                 reconstructed bipartite adjacency matrix
#         """
#         # move batch to device
#         batch = batch.to(self.device)

#         edge_index_bg_pred = []
#         edge_index_bg_true = []

#         # dense bipartite edge index
#         edge_index_bg_dense = torch.zeros(batch.x_b.shape[0], batch.x_g.shape[0]).to(
#             self.device
#         )
#         edge_index_bg_dense[batch.edge_index_bg[0], batch.edge_index_bg[1]] = 1

#         # get graph sizes (number of nodes) in the batch, used to slice the dense bipartite edge index
#         node2graph_idx = torch.stack(
#             [
#                 torch.cumsum(
#                     torch.cat(
#                         [
#                             torch.zeros(1).long().to(self.device),
#                             batch.x_b_batch.bincount(),
#                         ]
#                     ),
#                     dim=0,
#                 ),  # (Nb+1, ) CDR     nodes
#                 torch.cumsum(
#                     torch.cat(
#                         [
#                             torch.zeros(1).long().to(self.device),
#                             batch.x_g_batch.bincount(),
#                         ]
#                     ),
#                     dim=0,
#                 ),  # (Ng+1, ) antigen nodes
#             ],
#             dim=0,
#         )

#         for i in range(batch.num_graphs):
#             edge_index_bg_pred.append(
#                 F.sigmoid(
#                     self._dc_func(
#                         b_z=B_z[batch.x_b_batch == i], g_z=G_z[batch.x_g_batch == i]
#                     )
#                 )
#             )  # Tensor (Nb, Ng)
#             edge_index_bg_true.append(
#                 edge_index_bg_dense[
#                     node2graph_idx[0, i] : node2graph_idx[0, i + 1],
#                     node2graph_idx[1, i] : node2graph_idx[1, i + 1],
#                 ]
#             )  # Tensor (Nb, Ng)

#         return edge_index_bg_pred, edge_index_bg_true

#     def forward(self, batch: PygBatch) -> Dict[str, Union[int, Tensor]]:
#         # device
#         batch = batch.to(self.device)
#         # encode
#         z_ab, z_ag = self.encode(batch)  # (Nb, C), (Ng, C)
#         # decode
#         edge_index_bg_pred, edge_index_bg_true = self.decode(z_ab, z_ag, batch)

#         return {
#             "abdbid": batch.abdbid,  # List[str]
#             "edge_index_bg_pred": edge_index_bg_pred,  # List[Tensor (Nb, Ng)]
#             "edge_index_bg_true": edge_index_bg_true,  # List[Tensor (Nb, Ng)]
#         }



# ################## from MIPE ##################


# import os
# import torch
# from torch_geometric.nn import Linear
# from egnn_pytorch import EGNN
# from torch.nn import Sequential, BatchNorm1d, Dropout, Sigmoid, Conv1d, LSTM, LayerNorm, ReLU
# from torchdrug import models
# from model.CrossAttention import CrossAttention

# class MIPE(torch.nn.Module):
#     def __init__(self, share_weight=False, dropout=0.5, heads=4):
#         """
#         MIPE (Multi-scale Interaction Prediction Engine) model for protein-protein interaction prediction.
        
#         Args:
#             share_weight (bool): Whether to share weights between antigen and antibody processing branches
#             dropout (float): Dropout rate for regularization
#             heads (int): Number of attention heads (unused in current implementation)
#         """
#         super(MIPE, self).__init__()
        
#         # Dimension configurations
#         self.node_attr_dim = 62        # Dimension of node features
#         self.hidden_dim = 64           # Base hidden dimension
#         self.esm_dim = 1280            # ESM (Antigen) embedding dimension
#         self.prott5_dim = 1024         # ProtT5 embedding dimension (unused)
#         self.ablang_dim = 768          # AbLang (Antibody) embedding dimension
#         self.hidden_dim_cnn = 64       # CNN first layer hidden dimension
#         self.hidden_dim_cnn2 = 128     # CNN second layer hidden dimension
#         self.hidden_dim_cnn3 = 256     # CNN third layer hidden dimension
#         self.h1_dim = 64               # LSTM output dimension
#         self.h2_dim = 64               # Secondary feature dimension
#         self.share_weight = share_weight  # Weight sharing flag
#         self.dropout = dropout         # Dropout rate
#         self.heads = 1                 # Number of heads (unused)
#         self.multiheads = 16           # Number of attention heads for CrossAttention

#         # --------------------------
#         # 1. Graph Neural Network (Structure Processing)
#         # --------------------------
#         # GearNet for processing protein structures
#         self.gearnet = models.GearNet(
#             input_dim=62, 
#             hidden_dims=[64, 64, 64],  # 3 layers with 64 hidden units each
#             num_relation=3             # Number of edge types
#         )

#         # --------------------------
#         # 2. Sequence Processing Pipeline
#         # --------------------------
#         # CNN layers for antigen (ESM embeddings)
#         self.ag_cnn1 = Conv1d(
#             in_channels=self.esm_dim, 
#             out_channels=self.hidden_dim_cnn, 
#             kernel_size=5, 
#             padding='same'  # Maintains sequence length
#         )
        
#         # CNN layers for antibody (AbLang embeddings)
#         if self.share_weight:
#             self.ab_cnn1 = self.ag_cnn1  # Weight sharing
#         else:
#             self.ab_cnn1 = Conv1d(
#                 in_channels=self.ablang_dim, 
#                 out_channels=self.hidden_dim_cnn, 
#                 kernel_size=5,
#                 padding='same'
#             )

#         # Dilated CNN layers (capture longer-range dependencies)
#         self.ag_cnn2 = Conv1d(
#             self.hidden_dim_cnn, 
#             self.hidden_dim_cnn2, 
#             3, 
#             dilation=2,  # Increased receptive field
#             padding='same'
#         )
#         self.ag_cnn3 = Conv1d(
#             self.hidden_dim_cnn2, 
#             self.hidden_dim_cnn3, 
#             3, 
#             dilation=4,  # Even larger receptive field
#             padding='same'
#         )
        
#         # Antibody CNN branches (with optional weight sharing)
#         if self.share_weight:
#             self.ab_cnn2 = self.ag_cnn2
#             self.ab_cnn3 = self.ag_cnn3
#         else:
#             self.ab_cnn2 = Conv1d(
#                 self.hidden_dim_cnn, 
#                 self.hidden_dim_cnn2, 
#                 3, 
#                 dilation=2, 
#                 padding='same'
#             )
#             self.ab_cnn3 = Conv1d(
#                 self.hidden_dim_cnn2, 
#                 self.hidden_dim_cnn3, 
#                 3, 
#                 dilation=4, 
#                 padding='same'
#             )

#         # --------------------------
#         # 3. Bidirectional LSTM
#         # --------------------------
#         # Processes CNN outputs to capture sequential patterns
#         self.ag_lstm = LSTM(
#             input_size=self.hidden_dim_cnn3,
#             hidden_size=self.h1_dim // 2,  # Half dimension for bidirectional
#             num_layers=1,
#             batch_first=True,
#             bidirectional=True  # Bidirectional processing
#         )
        
#         # Antibody LSTM (with optional weight sharing)
#         if self.share_weight:
#             self.ab_lstm = self.ag_lstm
#         else:
#             self.ab_lstm = LSTM(
#                 input_size=self.hidden_dim_cnn3,
#                 hidden_size=self.h1_dim // 2,
#                 num_layers=1,
#                 batch_first=True,
#                 bidirectional=True
#             )

#         # --------------------------
#         # 4. Output Prediction Heads
#         # --------------------------
#         # Sequence-based prediction heads
#         self.linear_ag_seq = Sequential(
#             Linear(self.h1_dim, 1),  # Binary classification
#             Sigmoid()               # Probability output
#         )
#         self.linear_ag_strc = Sequential(
#             Linear(self.h1_dim, 1),
#             Sigmoid()
#         )
#         self.linear_ab_seq = Sequential(
#             Linear(self.h1_dim, 1),
#             Sigmoid()
#         )
#         self.linear_ab_strc = Sequential(
#             Linear(self.h1_dim, 1),
#             Sigmoid()
#         )

#         # --------------------------
#         # 5. Feature Fusion Layers
#         # --------------------------
#         # Combine structural and sequential features
#         self.linear_ag = Linear(self.h1_dim + self.h1_dim, self.h1_dim)
#         self.linear_ab = Linear(self.h1_dim + self.h1_dim, self.h1_dim)

#         # --------------------------
#         # 6. Normalization Layers
#         # --------------------------
#         # Batch normalization with dropout for CNN outputs
#         self.ag_bnorm1 = Sequential(
#             BatchNorm1d(self.hidden_dim_cnn, track_running_stats=False),
#             Dropout(0.5)
#         )
#         self.ag_bnorm2 = Sequential(
#             BatchNorm1d(self.hidden_dim_cnn2, track_running_stats=False),
#             Dropout(0.5)
#         )
#         self.ag_bnorm3 = Sequential(
#             BatchNorm1d(self.hidden_dim_cnn3, track_running_stats=False),
#             Dropout(0.5)
#         )
#         self.ag_bnorm4 = Sequential(
#             BatchNorm1d(self.hidden_dim, track_running_stats=False),
#             Dropout(0.5)
#         )
        
#         # Antibody normalization branches
#         self.ab_bnorm1 = Sequential(
#             BatchNorm1d(self.hidden_dim_cnn, track_running_stats=False),
#             Dropout(0.5)
#         )
#         self.ab_bnorm2 = Sequential(
#             BatchNorm1d(self.hidden_dim_cnn2, track_running_stats=False),
#             Dropout(0.5)
#         )
#         self.ab_bnorm3 = Sequential(
#             BatchNorm1d(self.hidden_dim_cnn3, track_running_stats=False),
#             Dropout(0.5)
#         )
#         self.ab_bnorm4 = Sequential(
#             BatchNorm1d(self.hidden_dim, track_running_stats=False),
#             Dropout(0.5)
#         )

#         # --------------------------
#         # 7. Cross-Attention Mechanism
#         # --------------------------
#         # Mutual attention between antigen and antibody features
#         self.ag_crossattention = CrossAttention(self.h1_dim, self.multiheads)
#         self.ab_crossattention = CrossAttention(self.h1_dim, self.multiheads)
        
#         # Feed-forward networks for attention outputs
#         self.ag_feed_forward = Sequential(
#             Linear(self.h1_dim, 4 * self.h1_dim),  # Expand dimension
#             ReLU(),
#             Linear(4 * self.h1_dim, self.h1_dim)   # Compress back
#         )
#         self.ab_feed_forward = Sequential(
#             Linear(self.h1_dim, 4 * self.h1_dim),
#             ReLU(),
#             Linear(4 * self.h1_dim, self.h1_dim)
#         )
        
#         # Layer normalization for attention outputs
#         self.ag_norm1 = LayerNorm(self.h1_dim)
#         self.ag_norm2 = LayerNorm(self.h1_dim)
#         self.ab_norm1 = LayerNorm(self.h1_dim)
#         self.ab_norm2 = LayerNorm(self.h1_dim)

#         # --------------------------
#         # 8. Final Prediction Heads
#         # --------------------------
#         self.ag_linearsigmoid = Sequential(
#             Linear(self.h1_dim, 1),  # Final binary prediction
#             Sigmoid()
#         )
#         self.ab_linearsigmoid = Sequential(
#             Linear(self.h1_dim, 1),
#             Sigmoid()
#         )

#     def forward(self, *agab):
#         """
#         Forward pass for the MIPE model.
        
#         Args:
#             agab: Tuple containing:
#                 - ag_x: Antigen node features
#                 - ag_edge_index: Antigen graph structure
#                 - ab_x: Antibody node features
#                 - ab_edge_index: Antibody graph structure
#                 - ag_esm: Antigen ESM embeddings
#                 - ab_esm: Antibody AbLang embeddings
#                 - ... (other optional args)
        
#         Returns:
#             Tuple of 14 elements containing predictions and intermediate features
#         """
#         # Unpack inputs
#         ag_x = agab[0]            # Antigen node features
#         ag_edge_index = agab[1]   # Antigen graph edges
#         ab_x = agab[2]            # Antibody node features
#         ab_edge_index = agab[3]   # Antibody graph edges
#         ag_esm = agab[4]          # Antigen sequence embeddings (ESM)
#         ab_esm = agab[5]          # Antibody sequence embeddings (AbLang)

#         # --------------------------
#         # Antigen Processing Branch
#         # --------------------------
        
#         # 1. Structure Processing (GearNet)
#         ag_edge_index[1] = torch.nn.functional.normalize(ag_edge_index[1])
#         ag_h1 = self.gearnet(ag_edge_index[0], ag_edge_index[1])["node_feature"]
#         ag_out_strc = self.linear_ag_strc(ag_h1)  # Structure-based prediction
        
#         # 2. Sequence Processing
#         # a) Normalize and process through CNN
#         ag_esm = torch.nn.functional.normalize(ag_esm)
#         ag_esm = self.ag_bnorm1((self.ag_cnn1(ag_esm)))
#         ag_esm = self.ag_bnorm2((self.ag_cnn2(ag_esm)))
#         ag_esm = self.ag_bnorm3((self.ag_cnn3(ag_esm)))
        
#         # b) LSTM processing
#         ag_esm = ag_esm.transpose(1, 2)  # Prepare for LSTM (batch, seq, features)
#         output_tensor, _ = self.ag_lstm(ag_esm)
        
#         # Combine bidirectional outputs
#         ag_h2 = torch.cat(
#             (output_tensor[:, :, :64 // 2],  # Forward pass
#              output_tensor[:, :, 64 // 2:]), # Backward pass
#             dim=2
#         )
#         ag_h2 = torch.squeeze(ag_h2, dim=0)  # Remove batch dimension
#         ag_h2 = self.ag_bnorm4(ag_h2)
#         ag_out_seq = self.linear_ag_seq(ag_h2)  # Sequence-based prediction
        
#         # 3. Feature Fusion
#         ag_h1 = self.linear_ag(torch.cat((ag_h1, ag_h2), dim=1))

#         # --------------------------
#         # Antibody Processing Branch (same structure as antigen)
#         # --------------------------
#         ab_edge_index[1] = torch.nn.functional.normalize(ab_edge_index[1])
#         ab_h1 = self.gearnet(ab_edge_index[0], ab_edge_index[1])["node_feature"]
#         ab_out_strc = self.linear_ab_strc(ab_h1)
        
#         ab_esm = torch.nn.functional.normalize(ab_esm)
#         ab_esm = self.ab_bnorm1((self.ab_cnn1(ab_esm)))
#         ab_esm = self.ab_bnorm2((self.ab_cnn2(ab_esm)))
#         ab_esm = self.ab_bnorm3((self.ab_cnn3(ab_esm)))
#         ab_esm = ab_esm.transpose(1, 2)
        
#         output_tensor, _ = self.ab_lstm(ab_esm)
#         ab_h2 = torch.cat(
#             (output_tensor[:, :, :64 // 2],
#              output_tensor[:, :, 64 // 2:]),
#             dim=2
#         )
#         ab_h2 = torch.squeeze(ab_h2, dim=0)
#         ab_h2 = self.ab_bnorm4(ab_h2)
#         ab_out_seq = self.linear_ab_seq(ab_h2)
#         ab_h1 = self.linear_ab(torch.cat((ab_h1, ab_h2), dim=1))

#         # --------------------------
#         # Cross-Attention Mechanism
#         # --------------------------
#         # Antigen attends to antibody features
#         ag_attention, ag_attention_weights = self.ag_crossattention(ag_h1, ab_h1)
#         # Antibody attends to antigen features
#         ab_attention, ab_attention_weights = self.ab_crossattention(ab_h1, ag_h1)
        
#         # Residual connections with normalization
#         ag_res1 = self.ag_norm1(ag_h1 + ag_attention)
#         ab_res1 = self.ab_norm1(ab_h1 + ab_attention)
        
#         # Feed-forward with residual
#         ag_res2 = self.ag_norm2(ag_res1 + self.ag_feed_forward(ag_res1))
#         ab_res2 = self.ab_norm2(ab_res1 + self.ab_feed_forward(ab_res1))

#         # --------------------------
#         # Final Predictions
#         # --------------------------
#         ag_out = self.ag_linearsigmoid(ag_res2)
#         ab_out = self.ab_linearsigmoid(ab_res2)

#         # Return all relevant outputs
#         return (
#             ag_out, ab_out,                   # Final predictions
#             ag_attention_weights,             # Antigen attention weights
#             ab_attention_weights,             # Antibody attention weights
#             ag_h1, ag_attention,              # Antigen features and attention
#             ab_h1, ab_attention,              # Antibody features and attention
#             ag_h2, ab_h2,                     # Sequence features
#             ag_out_seq, ag_out_strc,          # Intermediate predictions
#             ab_out_seq, ab_out_strc           # Intermediate predictions
#         )

