"""
TODO: [mansoor]
- refactor train_model.py from walle and main.py from mipe
- perform train, val, test in this file with wandb logging
- perform k-fold cross validation
- define modes of working: train, dev, tunning, sweep
"""

import os, logging

# ─────────────────────────────────────────────────────────────────────────────
# Silence the W&B malloc garbage on macOS and only log errors
# ─────────────────────────────────────────────────────────────────────────────
os.environ["WANDB_SILENT"] = "true"
logging.getLogger("wandb").setLevel(logging.ERROR)

import hydra
from omegaconf import DictConfig
import torch
from torch_geometric.data import Data
from torch.optim import Adam
import wandb

from wandb.errors.errors import CommError
from wandb.sdk.lib.service_connection import WandbServiceNotOwnedError

import numpy as np
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader as PygDataLoader
from tqdm import tqdm
import time, csv
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

from model.model import M3EPI
from model.loss import (
    binary_cross_entropy,
    ntxent_loss,
    intra_nce_loss,
    inter_nce_loss,
    gradient_weighted_nce
)
from model.metric import EpitopeMetrics
from model.callbacks import EarlyStopping, ModelCheckpoint
from utils import seed_everything, get_device, load_data, initialize_wandb, train_test_split

torch.set_float32_matmul_precision("high")


class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_b":
            return self.x_b.size(0)
        if key == "edge_index_g":
            return self.x_g.size(0)
        if key == "edge_index_ag_ab":
            # ensure the right increment for batching
            return torch.tensor([[self.x_b.size(0)], [self.x_g.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)


def create_dataloader(dataset, cfg):
    data_list = []
    for item in dataset:
        pair = PairData(
            x_b=torch.tensor(item["vertex_AB"], dtype=torch.float),
            edge_index_b=torch.tensor(item["edge_AB"], dtype=torch.long),
            y_b=torch.tensor(item["label_AB"], dtype=torch.float),
            x_g=torch.tensor(item["vertex_AG"], dtype=torch.float),
            edge_index_g=torch.tensor(item["edge_AG"], dtype=torch.long),
            y_g=torch.tensor(item["label_AG"], dtype=torch.float),
            edge_index_ag_ab=torch.tensor(item["edge_AGAB"], dtype=torch.long),
        )
        data_list.append(pair)
    return PygDataLoader(
        data_list,
        batch_size=cfg.hparams.train.batch_size,
        shuffle=True,
        follow_batch=["x_g", "x_b"]
    )


def edge_index_to_adj(edge_index, m, n):
    adj = torch.zeros((m, n), device=edge_index.device)
    adj[edge_index[0], edge_index[1]] = 1.0
    return adj


def compute_edge_loss(outputs, batch, device):
    losses = []
    # reconstruct per-complex pointers
    ptr_g = torch.cat([torch.tensor([0], device=device),
                       torch.cumsum(batch.x_g_batch.bincount(), dim=0)])
    ptr_b = torch.cat([torch.tensor([0], device=device),
                       torch.cumsum(batch.x_b_batch.bincount(), dim=0)])
    for i in range(ptr_g.size(0)-1):
        g0, g1 = ptr_g[i].item(), ptr_g[i+1].item()
        b0, b1 = ptr_b[i].item(), ptr_b[i+1].item()
        mask = ((batch.edge_index_ag_ab[0] >= g0) &
                (batch.edge_index_ag_ab[0] <  g1) &
                (batch.edge_index_ag_ab[1] >= b0) &
                (batch.edge_index_ag_ab[1] <  b1))
        ei = batch.edge_index_ag_ab[:, mask].clone()
        ei[0] -= g0; ei[1] -= b0
        adj = edge_index_to_adj(ei, g1-g0, b1-b0)
        pred = outputs["interaction_probs"][g0:g1, b0:b1]
        losses.append(binary_cross_entropy(pred, adj))
    return torch.stack(losses).mean()

def compute_contrastive_loss(out, batch, cfg, device):
    """
    Switch between 'infonce' (classic intra+inter) and 'gwnce' (ReGCL).
    """
    ptr_g = torch.cat([torch.tensor([0], device=device),
                       torch.cumsum(batch.x_g_batch.bincount(), 0)])
    ptr_b = torch.cat([torch.tensor([0], device=device),
                       torch.cumsum(batch.x_b_batch.bincount(), 0)])

    losses = []
    name, τ = cfg.loss.contrastive.name, cfg.loss.contrastive.temperature

    for i in range(ptr_g.size(0) - 1):
        g0, g1 = ptr_g[i].item(), ptr_g[i + 1].item()
        b0, b1 = ptr_b[i].item(), ptr_b[i + 1].item()

        ag_h = out["ag_embed"][g0:g1]
        ab_h = out["ab_embed"][b0:b1]
        y_ag, y_ab = batch.y_g[g0:g1].long(), batch.y_b[b0:b1].long()

        if name == "infonce":
            intra = intra_nce_loss(ag_h, ab_h, y_ag, y_ab, τ)
            inter = inter_nce_loss(ag_h, ab_h, y_ag, y_ab, τ)
            loss_i = (cfg.loss.contrastive.intra_weight * intra +
                      cfg.loss.contrastive.inter_weight * inter)

        elif name == 'gwnce':
            # local antigen graph
            ei_g = batch.edge_index_g[:, (batch.edge_index_g[0] >= g0) &
                                           (batch.edge_index_g[0] <  g1) &
                                           (batch.edge_index_g[1] >= g0) &
                                           (batch.edge_index_g[1] <  g1)].clone()
            ei_g -= torch.tensor([[g0], [g0]], device=device)

            # local bipartite graph
            ei_bg = batch.edge_index_ag_ab[:, (batch.edge_index_ag_ab[0] >= g0) &
                                                (batch.edge_index_ag_ab[0] <  g1) &
                                                (batch.edge_index_ag_ab[1] >= b0) &
                                                (batch.edge_index_ag_ab[1] <  b1)].clone()
            ei_bg[0] -= g0; ei_bg[1] -= b0

            loss_i = gradient_weighted_nce(
                ag_h, ab_h, ei_g, ei_bg,
                temperature=τ,
                cutrate=cfg.loss.gwnce.cut_rate,
                cutway=cfg.loss.gwnce.cut_way,
                mean=True, batch_size=0) * cfg.loss.gwnce.weight

        losses.append(loss_i)

    return torch.stack(losses).mean()



def train_epoch(model, loader, optimizer, device, metrics, cfg):
    model.train()
    total_loss = 0
    metrics.reset()

    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch)

        # 1) Node loss
        # node_loss = binary_cross_entropy(
        #     outputs["epitope_prob"].float(),
        #     batch.y_g.float()
        # )
        """
        NOTE: 
        - pos_weight assigns weights to the positive class due to class imbalance
        """
        node_loss = binary_cross_entropy(
            outputs["epitope_prob"].float(),
            batch.y_g.float(),
            pos_weight=cfg.loss.node_prediction.pos_weight
        )

        # 2) Edge loss
        edge_loss = compute_edge_loss(outputs, batch, device)

        # 3) Contrastive loss
        if cfg.loss.contrastive.name in ["gwnce", "infonce"]:
            cl_loss = compute_contrastive_loss(outputs, batch, cfg, device)
            loss = node_loss + edge_loss + cl_loss
        else:
            loss = node_loss + edge_loss
            # print("bce only")
        
        # print(loss, cl_loss, node_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        ep_pred = (outputs["epitope_prob"] > model.threshold).float()
        metrics.update(ep_pred, batch.y_g.long())

    return total_loss / len(loader), metrics.compute()


def validate_epoch(model, loader, device, metrics, cfg):
    model.eval()
    total_loss = 0
    metrics.reset()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)

            # node_loss = binary_cross_entropy(
            #     outputs["epitope_prob"].float(),
            #     batch.y_g.float()
            # )
            node_loss = binary_cross_entropy(
                outputs["epitope_prob"].float(),
                batch.y_g.float(),
                pos_weight=cfg.loss.node_prediction.pos_weight
            )
            edge_loss = compute_edge_loss(outputs, batch, device)
            
            # 3) Contrastive loss
            if cfg.loss.contrastive.name in ["gwnce", "infonce"]:
                cl_loss = compute_contrastive_loss(outputs, batch, cfg, device)
                loss = node_loss + edge_loss + cl_loss
            else:
                loss = node_loss + edge_loss
                # print("bce only")

            total_loss += loss.item()

            """
            TESTME: 
            - try different node prediction thresholds to monitor precision, recall, f1
            """
            # for thr in [0.5, 0.3, 0.2, 0.1]:
            #     ep_pred = (outputs["epitope_prob"] > thr).float()
            #     metrics.update(ep_pred, batch.y_g.long())

            ep_pred = (outputs["epitope_prob"] > model.threshold).float()
            metrics.update(ep_pred, batch.y_g.long())

    return total_loss / len(loader), metrics.compute()

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    start_time = time.time()
    seed_everything(cfg.seed)
    device = get_device()
    print("Using device: ", device)
    # print("sweep config", cfg)

    # 1) Load data asep_m3epi_transformed_test.pkl
    full_data = load_data(os.path.join(cfg.data_dir, "asep_m3epi_transformed.pkl"))
    # full_data = load_data(os.path.join(cfg.data_dir, "asep_m3epi_transformed_test.pkl"))


    # 2) MODE‑SPECIFIC HANDLING
    if cfg.mode.mode == "dev":
        # only take first N examples
        full_data = full_data[: cfg.mode.data.dev_subset]

    # 2) Train/test split if in test mode
    if cfg.mode.mode == "test":
        train_data, test_data = train_test_split(full_data, cfg.seed)
    else:
        train_data, test_data = full_data, None


    # 3) initialize W&B
    if cfg.logging_method == "wandb":
        initialize_wandb(cfg)

    # 4) build CV or single-fold iterator
    kf = KFold(n_splits=cfg.hparams.train.kfolds,
               shuffle=True, random_state=cfg.seed)
    if cfg.mode.mode == "test":
        splits = [(list(range(len(train_data))), [])]
    else:
        splits = list(kf.split(train_data))

    all_best = []
    met = EpitopeMetrics().to(device)

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n▶︎ Fold {fold+1}")
        train_subset = [train_data[i] for i in train_idx]
        train_dl = create_dataloader(train_subset, cfg)

        if cfg.mode.mode == "test":
            val_dl = None
        else:
            val_subset = [train_data[i] for i in val_idx]
            val_dl = create_dataloader(val_subset, cfg)

        model = M3EPI(cfg).to(device)
        opt   = Adam(model.parameters(),
                     lr=cfg.hparams.train.learning_rate,
                     weight_decay=cfg.hparams.train.weight_decay)

        if cfg.mode.mode in ("train", "dev"):
            es = EarlyStopping(**cfg.callbacks.early_stopping)
            ck = ModelCheckpoint(**cfg.callbacks.model_checkpoint, config=cfg)
        else:
            es = ck = None

        for epoch in range(cfg.hparams.train.num_epochs):
            tr_loss, _ = train_epoch(model, train_dl, opt, device, met, cfg)
            if val_dl is not None:
                vl_loss, vl_met = validate_epoch(model, val_dl, device, met, cfg)
            else:
                vl_loss, vl_met = tr_loss, met.compute()

            # ─────────────────────────────────────────────────────────────
            # ✦ new: log metrics to W&B each epoch
            # append val to the metrics: val_mcc, val_auprc, val_f1, val_precision, val_recall, val_auroc
            # logs these metrics to wandb and maximize val_mcc
            if cfg.logging_method == "wandb":
                wandb.log({
                    "fold": fold,
                    "epoch": epoch,
                    "train_loss": tr_loss,
                    "val_loss": vl_loss,
                    **{f"val_{k}": v for k,v in vl_met.items()} 
                })
            # ─────────────────────────────────────────────────────────────

            # log & checkpoint
            if ck is not None:
                ck(model, vl_loss, epoch)
            if es is not None and es(vl_loss):
                break

            # convert to floats before storing
            final_metrics = {
                k: (v.cpu().item() if isinstance(v, torch.Tensor) else v)
                for k, v in vl_met.items()
            }
        all_best.append(final_metrics) # save best results from each fold exp

        # all_best.append(vl_met)

    # 5) report CV or test
    avg = {k: np.mean([m[k] for m in all_best]) for k in all_best[0]}
    std = {k: np.std([m[k] for m in all_best])  for k in all_best[0]}
    print("\n=== Final ===")
    for k in avg:
        print(f"{k}: {avg[k]:.4f} ± {std[k]:.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # ✦ new: log k-fold aggregate to W&B
    if cfg.logging_method == "wandb":
        wandb.log({f"cv_{k}": avg[k] for k in avg})
        wandb.log({f"cv_{k}_std": std[k] for k in std})
    # ─────────────────────────────────────────────────────────────────────────

    # 6) if in TEST mode, evaluate hold-out
    if cfg.mode.mode == "test":
        test_dl = create_dataloader(test_data, cfg)
        _, test_met = validate_epoch(model, test_dl, device, met, cfg)
        print("\n=== Test ===")
        for k, v in test_met.items():
            print(f"{k}: {v:.4f}")
        # ─────────────────────────────────────────────────────────────────────────
        # ✦ new: log test metrics to W&B
        if cfg.logging_method == "wandb":
            wandb.log({f"test_{k}": v for k,v in test_met.items()})
        # ─────────────────────────────────────────────────────────────────────────

    # 7) save per-fold summary 
    ####################### saving results summary #############################
    # measure training duration
    elapsed = time.time() - start_time

    # where to save
    base = Path(cfg.callbacks.model_checkpoint.dirpath) \
           / cfg.model.name \
           / cfg.model.decoder.type \
           / (cfg.loss.contrastive.name or "ce")
    summary_dir = base / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # 1) per-fold CSV
    per_fold_file = summary_dir / f"{cfg.model.name}_{cfg.model.decoder.type}_{cfg.loss.contrastive.name}_"
    per_fold_file = per_fold_file.with_name(per_fold_file.name + "cv_folds.csv")
    with open(per_fold_file, "w", newline="") as f:
        writer = csv.writer(f)
        # header
        header = ["fold"] + list(all_best[0].keys())
        writer.writerow(header + ["train_time_s"])
        # one row per fold
        for i, met in enumerate(all_best,1):
            row = [i] + [met[k] for k in header[1:]] + [""]
            writer.writerow(row)
        # final row: means ± std
        means = {k: np.mean([m[k] for m in all_best]) for k in all_best[0]}
        stds  = {k: np.std ([m[k] for m in all_best]) for k in all_best[0]}
        mean_row = ["mean"] + [f"{means[k]:.4f}±{stds[k]:.4f}" for k in header[1:]] + [f"{elapsed:.1f}"]
        writer.writerow(mean_row)
    print(f"→ per‐fold summary saved to {per_fold_file}")

    if cfg.logging_method=="wandb" and fold == cfg.hparams.train.kfolds-1:
        wandb.log({"train_time_s": elapsed})
    ####################### end saving results summary #############################

    # ─────────────────────────────────────────────────────────────────────────
    # ✦ new: close W&B run once everything is done
    if cfg.logging_method == "wandb":
        """
        FIXME: during multi_gpu runs, wandb.finish() throws an error on run completion.
        """
        # try:
        #     wandb.finish()
        # except WandbServiceNotOwnedError:
        #     # If the service actually belonged to the agent process, ignore
        #     pass
        wandb.finish()
    # ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()






"""
Example usage:
python main.py \
  mode=test \
  wandb.notes="test run" \
  wandb.tags=["test"] \
  model.name=GIN \
  model.decoder.type=attention \
  loss.contrastive.name=gwnce \
  hparams.train.batch_size=64 \
  hparams.train.kfolds=5 \
  hparams.train.learning_rate=1e-3 \
  num_threads=1
"""


# @hydra.main(config_path="conf", config_name="config")
# def main(cfg: DictConfig):
#     start_time = time.time()
#     seed_everything(cfg.seed)
#     device = get_device()

#     # 1) Load data
#     full_data = load_data(os.path.join(cfg.data_dir,
#                                       "asep_m3epi_transformed.pkl"))
    
#     # print(cfg.mode.mode)

#     # 2) Train/test split if in test mode
#     if cfg.mode.mode == "test":
#         train_data, test_data = train_test_split(full_data, cfg.seed)
#     else:
#         train_data, test_data = full_data, None

#     # 3) initialize W&B
#     if cfg.logging_method == "wandb":
#         initialize_wandb(cfg)

#     # 4) build CV or single-fold iterator
#     kf = KFold(n_splits=cfg.hparams.train.kfolds,
#                shuffle=True, random_state=cfg.seed)
#     if cfg.mode.mode == "test":
#         splits = [(list(range(len(train_data))), [])]
#     else:
#         splits = list(kf.split(train_data))

#     all_best = []
#     met = EpitopeMetrics()

#     for fold, (train_idx, val_idx) in enumerate(splits):
#         print(f"\n▶︎ Fold {fold+1}")
#         train_subset = [train_data[i] for i in train_idx]
#         train_dl = create_dataloader(train_subset, cfg)

#         if cfg.mode.mode == "test":
#             val_dl = None
#         else:
#             val_subset = [train_data[i] for i in val_idx]
#             val_dl = create_dataloader(val_subset, cfg)

#         model = M3EPI(cfg).to(device)
#         opt   = Adam(model.parameters(),
#                      lr=cfg.hparams.train.learning_rate,
#                      weight_decay=cfg.hparams.train.weight_decay)

#         if cfg.mode.mode in ("train", "dev"):
#             es = EarlyStopping(**cfg.callbacks.early_stopping)
#             ck = ModelCheckpoint(**cfg.callbacks.model_checkpoint, config=cfg)
#         else:
#             es = ck = None

#         for epoch in range(cfg.hparams.train.num_epochs):
#             tr_loss, _ = train_epoch(model, train_dl, opt, device, met, cfg)
#             if val_dl is not None:
#                 vl_loss, vl_met = validate_epoch(model, val_dl, device, met, cfg)
#             else:
#                 vl_loss, vl_met = tr_loss, met.compute()

#             # log & checkpoint
#             if ck is not None:
#                 ck(model, vl_loss, epoch)
#             if es is not None and es(vl_loss):
#                 break

#         all_best.append(vl_met)

#     # 5) report CV or test
#     avg = {k: np.mean([m[k] for m in all_best]) for k in all_best[0]}
#     std = {k: np.std([m[k] for m in all_best])  for k in all_best[0]}
#     print("\n=== Final ===")
#     for k in avg:
#         print(f"{k}: {avg[k]:.4f} ± {std[k]:.4f}")

#     # 6) if in TEST mode, evaluate hold-out
#     if cfg.mode.mode == "test":
#         test_dl = create_dataloader(test_data, cfg)
#         _, test_met = validate_epoch(model, test_dl, device, met, cfg)
#         print("\n=== Test ===")
#         for k, v in test_met.items():
#             print(f"{k}: {v:.4f}")

#     # 7) save per‐fold summary (unchanged) …
#     ####################### saving results summary #############################
#     # measure training duration
#     elapsed = time.time() - start_time

#     # where to save
#     base = Path(cfg.callbacks.model_checkpoint.dirpath) \
#            / cfg.model.name \
#            / cfg.model.decoder.type \
#            / (cfg.loss.contrastive.name or "ce")
#     summary_dir = base / "summary"
#     summary_dir.mkdir(parents=True, exist_ok=True)

#     # 1) per‑fold CSV
#     per_fold_file = summary_dir / f"{cfg.model.name}_{cfg.model.decoder.type}_{cfg.loss.contrastive.name}_"
#     per_fold_file = per_fold_file.with_name(per_fold_file.name + "cv_folds.csv")
#     with open(per_fold_file, "w", newline="") as f:
#         writer = csv.writer(f)
#         # header
#         header = ["fold"] + list(all_best[0].keys())
#         writer.writerow(header + ["train_time_s"])
#         # one row per fold
#         for i, met in enumerate(all_best,1):
#             row = [i] + [met[k].item() for k in header[1:]] + [""]
#             writer.writerow(row)
#         # final row: means ± std
#         means = {k: np.mean([m[k] for m in all_best]) for k in all_best[0]}
#         stds  = {k: np.std ([m[k] for m in all_best]) for k in all_best[0]}
#         mean_row = ["mean"] + [f"{means[k]:.4f}±{stds[k]:.4f}" for k in header[1:]] + [f"{elapsed:.1f}"]
#         writer.writerow(mean_row)
#     print(f"→ per‐fold summary saved to {per_fold_file}")

#     if cfg.logging_method=="wandb" and fold == cfg.hparams.train.kfolds-1:
#         wandb.log({"train_time_s": elapsed})
#     ####################### end saving results summary #############################

#     if cfg.logging_method == "wandb":
#         wandb.finish()

# if __name__ == "__main__":
#     main()


# @hydra.main(config_path="conf", config_name="config")
# def main(cfg: DictConfig):
#     start_time = time.time()

#     seed_everything(cfg.seed)
#     device = get_device()
#     print("Using device: ", device)

#     # 1) load full data
#     data = load_data(os.path.join(cfg.data_dir,
#                                   "asep_mipe_transformed_100_examples.pkl"))

#     # 2) MODE‑SPECIFIC HANDLING
#     if cfg.mode.mode == "dev":
#         # only take first N examples
#         data = data[: cfg.mode.data.dev_subset]

#     # 3) initialize wandb once for train & sweep (& offline for dev)
#     if cfg.logging_method == "wandb":
#         initialize_wandb(cfg)

#     # 4) set up cross‑validation folds
#     kf = KFold(n_splits=cfg.hparams.train.kfolds,
#                shuffle=True, random_state=cfg.seed)
#     all_best = []

#     for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
#         print(f"\n▶︎ Fold {fold+1}")
#         train_dl = create_dataloader([data[i] for i in train_idx], cfg)
#         val_dl   = create_dataloader([data[i] for i in val_idx],   cfg)

#         model = M3EPI(cfg).to(device)
#         opt   = Adam(model.parameters(),
#                      lr=cfg.hparams.train.learning_rate,
#                      weight_decay=cfg.hparams.train.weight_decay)

#         # only use early‑stopping & checkpoint in 'train' or 'dev'
#         # print(cfg.mode)
#         if cfg.mode.mode in ("train", "dev"):
#             es = EarlyStopping(**cfg.callbacks.early_stopping)
#             # ck = ModelCheckpoint(**cfg.callbacks.model_checkpoint)
#             # print("Saving model checkpoint...")

#             ck = ModelCheckpoint(**cfg.callbacks.model_checkpoint, config=cfg)
#         else:
#             # print("Don't save model checkpoint for sweep...")
#             es = ck = None

#         met = EpitopeMetrics()

#         for epoch in range(cfg.hparams.train.num_epochs):
#             tr_loss, tr_met = train_epoch(model, train_dl, opt,
#                                           device, met, cfg)
#             vl_loss, vl_met = validate_epoch(model, val_dl,
#                                              device, met, cfg)

#             # log metrics
#             if cfg.logging_method == "wandb":
#                 wandb.log({
#                     f"fold{fold+1}/train_loss": tr_loss,
#                     f"fold{fold+1}/val_loss":   vl_loss,
#                     **{f"fold{fold+1}/train_{k}": v for k,v in tr_met.items()},
#                     **{f"fold{fold+1}/val_{k}":   v for k,v in vl_met.items()},
#                     "epoch": epoch
#                 })

#             # checkpoint & early stop
#             if ck is not None:
#                 ck(model, vl_loss, epoch)
#             if es is not None and es(vl_loss):
#                 print(f"⏹ Early stopping @ epoch {epoch}")
#                 break

#         all_best.append(vl_met)

#     # aggregate & print
#     avg = {k: np.mean([m[k] for m in all_best]) for k in all_best[0]}
#     std = {k: np.std([m[k] for m in all_best])  for k in all_best[0]}
#     print("\n=== Final ===")
#     for k in avg:
#         print(f"{k}: {avg[k]:.4f} ± {std[k]:.4f}")

#     ####################### saving results summary #############################
#     # measure training duration
#     elapsed = time.time() - start_time

#     # where to save
#     base = Path(cfg.callbacks.model_checkpoint.dirpath) \
#            / cfg.model.name \
#            / cfg.model.decoder.type \
#            / (cfg.loss.contrastive.name or "ce")
#     summary_dir = base / "summary"
#     summary_dir.mkdir(parents=True, exist_ok=True)

#     # 1) per‑fold CSV
#     per_fold_file = summary_dir / f"{cfg.model.name}_{cfg.model.decoder.type}_{cfg.loss.contrastive.name}_"
#     per_fold_file = per_fold_file.with_name(per_fold_file.name + "cv_folds.csv")
#     with open(per_fold_file, "w", newline="") as f:
#         writer = csv.writer(f)
#         # header
#         header = ["fold"] + list(all_best[0].keys())
#         writer.writerow(header + ["train_time_s"])
#         # one row per fold
#         for i, met in enumerate(all_best,1):
#             row = [i] + [met[k].item() for k in header[1:]] + [""]
#             writer.writerow(row)
#         # final row: means ± std
#         means = {k: np.mean([m[k] for m in all_best]) for k in all_best[0]}
#         stds  = {k: np.std ([m[k] for m in all_best]) for k in all_best[0]}
#         mean_row = ["mean"] + [f"{means[k]:.4f}±{stds[k]:.4f}" for k in header[1:]] + [f"{elapsed:.1f}"]
#         writer.writerow(mean_row)
#     print(f"→ per‐fold summary saved to {per_fold_file}")

#     if cfg.logging_method=="wandb" and fold == cfg.hparams.train.kfolds-1:
#         wandb.log({"train_time_s": elapsed})
#     ####################### end saving results summary #############################

#     if cfg.logging_method == "wandb":
#         wandb.finish()


# if __name__ == "__main__":
#     main()

    
    





# import os
# import hydra
# from omegaconf import DictConfig
# import torch
# # from torch.utils.data import DataLoader, Dataset
# from torch_geometric.data import Data, Dataset, Batch
# from torch.optim import Adam
# import wandb
# from pathlib import Path
# import numpy as np
# from sklearn.model_selection import KFold

# from model.model import M3EPI
# from model.loss import binary_cross_entropy, ntxent_loss
# from model.metric import EpitopeMetrics
# from model.callbacks import EarlyStopping, ModelCheckpoint
# from utils import seed_everything, get_device, load_data, initialize_wandb

# from typing import Any, Callable, Dict, List, Optional, Tuple

# import yaml
# from loguru import logger
# from torch import Tensor
# from torch.optim import lr_scheduler
# from torch.utils.tensorboard import SummaryWriter
# from torch_geometric.data import Batch as PygBatch
# from torch_geometric.loader import DataLoader as PygDataLoader
# import warnings
# warnings.filterwarnings("ignore")

# from tqdm import tqdm
# # set precision
# torch.set_float32_matmul_precision("high")


# class PairData(Data):
#     def __inc__(self, key, value, *args, **kwargs):
#         if key == "edge_index_b":
#             return self.x_b.size(0)
#         if key == "edge_index_g":
#             return self.x_g.size(0)
#         if key == "edge_index_ag_ab":
#             return torch.tensor([[self.x_b.size(0)], [self.x_g.size(0)]])
#         return super().__inc__(key, value, *args, **kwargs)


# def create_dataloader(dataset, cfg):
#     # Example: create a list of PairData objects
#     data_list = []
#     for item in dataset:
#         pair = PairData(
#             x_b=torch.tensor(item["vertex_AB"], dtype=torch.float),
#             edge_index_b=torch.tensor(item["edge_AB"], dtype=torch.long),
#             y_b=torch.tensor(item["label_AB"], dtype=torch.float),
#             x_g=torch.tensor(item["vertex_AG"], dtype=torch.float),
#             edge_index_g=torch.tensor(item["edge_AG"], dtype=torch.long),
#             y_g=torch.tensor(item["label_AG"], dtype=torch.float),
#             edge_index_ag_ab=torch.tensor(item["edge_AGAB"], dtype=torch.long),
#         )
#         data_list.append(pair)

#     # Use PyG DataLoader
#     data_loader = PygDataLoader(data_list, batch_size=cfg.hparams.train.batch_size, shuffle=True)
#     return data_loader


# # python

# def edge_index_to_adj(edge_index, num_ag_nodes, num_ab_nodes):
#     # edge_index: shape [2, num_edges]
#     adj = torch.zeros((num_ag_nodes, num_ab_nodes), device=edge_index.device)
#     print(adj.shape )
#     print(edge_index)
#     adj[edge_index[0], edge_index[1]] = 1.0
#     return adj

# def train_epoch(model, train_loader, optimizer, device, metrics):
#     model.train()
#     total_loss = 0
#     metrics.reset()
#     for batch in train_loader:
#         batch = batch.to(device)
#         outputs = model(batch)



#         # Node (epitope) loss
#         node_loss = binary_cross_entropy(
#             outputs['epitope_pred'].float(), batch["y_g"].float()
#         )

#         # Edge (link) loss
#         # Construct ground-truth adjacency matrix from edge_index_ag_ab
#         m = len(batch["x_g"])
#         n = len(batch["x_b"])
#         adj_true = edge_index_to_adj(batch["edge_index_ag_ab"], m, n)
#         edge_loss = binary_cross_entropy(
#             outputs['interaction_probs'], adj_true
#         )

#         # Total loss (weight as needed)
#         loss = node_loss + edge_loss

#         optimizer.zero_grad()

#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         # print(len(outputs['epitope_pred']), len(batch['y_g']))

#         metrics.update(outputs['epitope_pred'].float(), batch['y_g'].long())
#         # Optionally: metrics.update_edge(outputs['interaction_probs'], adj_true)

#     return total_loss / len(train_loader) , metrics.compute()

# def validate_epoch(model, val_loader, device, metrics):
#     model.eval()
#     total_loss = 0
#     metrics.reset()
#     with torch.no_grad():
#         for batch in val_loader:
#             batch = batch.to(device)
#             outputs = model(batch)

#             node_loss = binary_cross_entropy(
#                 outputs['epitope_pred'].float(), batch["y_g"]
#             )
#             m = len(batch["x_g"])
#             n = len(batch["x_b"])
#             adj_true = edge_index_to_adj(batch["edge_index_ag_ab"], m, n)
#             edge_loss = binary_cross_entropy(
#                 outputs['interaction_probs'], adj_true
#             )

#             # contrast_loss = ntxent_loss(
#             # outputs['ag_embed'], 
#             # outputs['ab_embed']
#             # )
        
#             # loss = edge_loss + node_loss + 0.1 * contrast_loss

#             loss = node_loss  + edge_loss

#             total_loss += loss.item()
#             print(len(outputs['epitope_pred']), len(batch['y_g']))
#             metrics.update(outputs['epitope_pred'].float(), batch['y_g'].long())
#             # Optionally: metrics.update_edge(outputs['interaction_probs'], adj_true)

#     return total_loss / len(val_loader) , metrics.compute()



# @hydra.main(config_path="conf", config_name="config")
# def main(cfg: DictConfig):
#     # Set random seed
#     seed_everything(cfg.seed)

#     # Setup device
#     device = get_device()

#     # Load data
#     data = load_data(cfg.data_dir + "/asep_mipe_transformed_100_examples.pkl")
#     # data = load_data(cfg.data_dir + "/dict_pre_cal_esm2_esm2.pt")

#     # Initialize wandb
#     if cfg.logging_method == 'wandb':
#         initialize_wandb(cfg)

#     # Setup k-fold cross validation
#     kfold = KFold(n_splits=5, shuffle=True, random_state=cfg.seed)

#     # Track best metrics across folds
#     best_metrics = []

#     for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
#         print(f"\nTraining Fold {fold+1}")

#         # Create datasets
#         train_data = [data[i] for i in train_idx]
#         val_data = [data[i] for i in val_idx]
        

#         # Create data loaders using PyTorch Geometric DataLoader
#         train_loader = create_dataloader(train_data, cfg)

#         # train_loader = DataLoader(
#         #     train_data,
#         #     batch_size=cfg.hparams.train.batch_size,
#         #     shuffle=True
#         # )
        
#         val_loader = create_dataloader(val_data, cfg)

#         # val_loader = DataLoader(
#         #     val_data,
#         #     batch_size=cfg.hparams.train.batch_size
#         # )
        

#         # Initialize model
#         model = M3EPI(cfg).to(device)

#         # Setup optimizer
#         optimizer = Adam(
#             model.parameters(),
#             lr=cfg.hparams.train.learning_rate,
#             weight_decay=cfg.hparams.train.weight_decay
#         )

#         # Setup callbacks
#         early_stopping = EarlyStopping(**cfg.callbacks.early_stopping)
#         model_checkpoint = ModelCheckpoint(**cfg.callbacks.model_checkpoint)
#         metrics = EpitopeMetrics()

#         # Training loop
#         for epoch in range(cfg.hparams.train.num_epochs):
#             # train_loss, train_metrics = train_epoch(
#             #     model, train_loader, optimizer, device, metrics
#             # )
#             # val_loss, val_metrics = validate_epoch(
#             #     model, val_loader, device, metrics
#             # )
#             train_loss, train_metrics = train_epoch(
#                 model, train_loader, optimizer, device, metrics
#             )
#             val_loss, val_metrics = validate_epoch(
#                 model, val_loader, device, metrics
#             )

#             # Log metrics
#             if cfg.logging_method == 'wandb':
#                 wandb.log({
#                     f'fold_{fold+1}/train_loss': train_loss,
#                     f'fold_{fold+1}/val_loss': val_loss,
#                     **{f'fold_{fold+1}/train_{k}': v for k, v in train_metrics.items()},
#                     **{f'fold_{fold+1}/val_{k}': v for k, v in val_metrics.items()},
#                     'epoch': epoch
#                 })

#             # Save checkpoint
#             model_checkpoint(model, val_loss, epoch)

#             # Early stopping
#             if early_stopping(val_loss):
#                 print(f"Early stopping triggered at epoch {epoch}")
#                 break

#         best_metrics.append(val_metrics)

#     # Print final results
#     metrics_avg = {
#         k: np.mean([m[k] for m in best_metrics])
#         for k in best_metrics[0].keys()
#     }
#     metrics_std = {
#         k: np.std([m[k] for m in best_metrics])
#         for k in best_metrics[0].keys()
#     }

#     print("\nFinal Results:")
#     for k in metrics_avg.keys():
#         print(f"{k}: {metrics_avg[k]:.4f} ± {metrics_std[k]:.4f}")

#     if cfg.logging_method == 'wandb':
#         wandb.finish()

# if __name__ == "__main__":
#     main()




# ################## from MIPE ##########################

# import os, time, csv
# import torch
# from tqdm import tqdm
# import numpy as np
# import pandas as pd
# from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryMatthewsCorrCoef, BinaryAveragePrecision, BinaryAUROC
# from model.loss import NTXentLoss  # Custom NT-Xent loss implementation
# from model import MIPE  # Main model implementation
# from utils import *  # Utility functions

# import warnings
# warnings.filterwarnings("ignore")

# # Add current directory and code directory to Python path
# import sys, os
# sys.path.append(os.getcwd())
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'code')))

# if __name__ == '__main__':
#     # Set device to GPU if available, otherwise CPU
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     """
#     Data loading section
#     TODO: [mansoor]
#     - add data file paths
#     - change the pkl file loading to torch.load() due to cuda-only original pkl file
#         - re-saved the original pkl file on cuda machine using torch.save()
#     """
#     # Define data and results directories
#     data_dir = os.path.join(os.getcwd(), "../../../data/asep/trans_baselines/mipe")
#     results_dir = os.path.join(os.getcwd(), "../../../results/hgraphepi/baselines/mipe")

#     # Load preprocessed data
#     mipe_cvdata_pkl_path = os.path.join(data_dir, "mipe_cvdata_cpu.pkl")
#     data_PECAN = torch.load(mipe_cvdata_pkl_path, map_location=torch.device('cpu'))

#     epoch = 2  # Original value was 800

#     # 5-fold cross-validation setup
#     K = 5
#     for kfold in range(K):

#         print(f"\n{'='*50}")
#         print(f"Fold {kfold+1}/{K}")
#         print(f"{'='*50}")

#         # Split data into train/val/test sets for current fold
#         train_data_PECAN, val_data_PECAN, test_data_PECAN = get_k_fold_data(K, kfold, data_PECAN)

#         # print(train_data_PECAN)
        
#         # Initialize model, loss functions and optimizer
#         model = MIPE().to(device)
#         loss_NTXent = NTXentLoss()  # Contrastive loss
#         loss_BCE = torch.nn.BCELoss()  # Binary cross-entropy loss
#         optim = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
        
#         # Training parameters
#         # epoch = 2  # Original value was 800
#         model.train()
#         Loss_val = []  # Track validation loss
#         AUROC_val = []  # Track validation AUROC
#         torch.cuda.empty_cache()  # Clear GPU cache

#         best_val_auroc = 0
#         best_epoch = 0
        
#         # Training loop
#         # for e in range(epoch):
        
#         # Training loop with tqdm
#         for e in tqdm(range(epoch), desc="Training Progress", unit="epoch"):
            
#             ############################# training phase ############################# 
                        
#             train_loss = 0
#             start_time = time.time()

#             # Initialize metrics for training epoch
#             result_loss_train_all = 0.0
#             train_auprc = 0.0
#             train_auroc = 0.0
#             train_precision = 0.0
#             train_recall = 0.0
#             train_mcc = 0.0
#             ab_train_auprc = 0.0
#             ab_train_auroc = 0.0
#             ab_train_precision = 0.0
#             ab_train_recall = 0.0
#             ab_train_mcc = 0.0
            
#             model.train()  # Set model to training mode
            
#             # Batch training loop
            
#             # Batch processing with progress bar
#             batch_iter = tqdm(range(len(train_data_PECAN)), desc=f"Epoch {e+1}", leave=False)

#             # for i in range(len(train_data_PECAN)):
#             for i in batch_iter:
#                 # Load antigen (AG) data
#                 ag_node_attr = torch.tensor(train_data_PECAN[i]["vertex_AG"], dtype=torch.float).to(device)
#                 ag_edge_ind = torch.tensor(train_data_PECAN[i]["edge_AG"], dtype=torch.long).to(device)
#                 ag_targets = torch.tensor(train_data_PECAN[i]["label_AG"], dtype=torch.float).to(device)
                
#                 # Load antibody (AB) data
#                 ab_node_attr = torch.tensor(train_data_PECAN[i]["vertex_AB"], dtype=torch.float).to(device)
#                 ab_edge_ind = torch.tensor(train_data_PECAN[i]["edge_AB"], dtype=torch.long).to(device)
#                 ab_targets = torch.tensor(train_data_PECAN[i]["label_AB"], dtype=torch.float).to(device)
                
#                 # Load protein language model embeddings
#                 ag_esm = torch.unsqueeze(torch.tensor(train_data_PECAN[i]["ESM1b_AG"]).to(device).t(), dim=0)
#                 ab_esm = torch.unsqueeze(torch.tensor(train_data_PECAN[i]["AbLang_AB"]).to(device).t(), dim=0)
                
#                 # Prepare input tensors
#                 ag_node_attr = torch.unsqueeze(ag_node_attr, dim=0)
#                 ab_node_attr = torch.unsqueeze(ab_node_attr, dim=0)
                
#                 # Create graph structures
#                 ag_edge_ind, ab_edge_ind = CreateGearnetGraph(train_data_PECAN[i])

#                 # print(ag_edge_ind)
                
#                 # Prepare model input tuple of size 14 (6 mandatory arguments, 8 optional)
#                 agab = [ag_node_attr, ag_edge_ind, ab_node_attr, ab_edge_ind, ag_esm, ab_esm, False, ag_targets, ab_targets, i]
                
#                 # Forward pass
#                 outputs = model(*agab)
                
#                 # Create edge labels for interaction prediction
#                 edge_label = np.zeros((ag_targets.shape[0], ab_targets.shape[0]))
#                 inter_edge_index = (train_data_PECAN[i]["edge_AGAB"])
#                 for edge_ind in range(len(inter_edge_index[0])):
#                     edge_label[inter_edge_index[0][edge_ind], inter_edge_index[1][edge_ind]] = 1.0
#                 edge_label = torch.tensor(edge_label, dtype=torch.float32).to(device)
                
#                 # Multimodal loss calculation
#                 # Get sequence and structure outputs
#                 ag_out_seq = outputs[10]
#                 ag_out_strc = outputs[11]
#                 ab_out_seq = outputs[12]
#                 ab_out_strc = outputs[13]
                
#                 # Intra-modal contrastive losses
#                 multimodal_loss_ag_intra_seq = loss_NTXent(outputs[4], outputs[4])
#                 multimodal_loss_ag_intra_struc = loss_NTXent(outputs[8], outputs[8])
#                 multimodal_loss_ab_intra_seq = loss_NTXent(outputs[6], outputs[6])
#                 multimodal_loss_ab_intra_strc = loss_NTXent(outputs[9], outputs[9])
                
#                 # Create tags for positive examples
#                 tag_ag = []
#                 tag_ab = []
#                 for tag_idx in range(len(train_data_PECAN[i]["label_AG"])):
#                     if ((train_data_PECAN[i]["label_AG"][tag_idx] and ag_out_seq[tag_idx]) or (
#                             train_data_PECAN[i]["label_AG"][tag_idx] and ag_out_strc[tag_idx])):
#                         tag_ag.append(1.)
#                     else:
#                         tag_ag.append(0.)
#                 for tag_idx in range(len(train_data_PECAN[i]["label_AB"])):
#                     if ((train_data_PECAN[i]["label_AB"][tag_idx] and ab_out_seq[tag_idx]) or (
#                             train_data_PECAN[i]["label_AB"][tag_idx] and ab_out_strc[tag_idx])):
#                         tag_ab.append(1.)
#                     else:
#                         tag_ab.append(0.)
                
#                 # Calculate cosine similarities
#                 cos_ag = consine_inter(outputs[4], outputs[8])
#                 cos_ab = consine_inter(outputs[6], outputs[9])
                
#                 # Weight positive examples
#                 cos_ag_pos = torch.mul(torch.tensor(tag_ag).to(device), cos_ag.to(device))
#                 cos_ab_pos = torch.mul(torch.tensor(tag_ab).to(device), cos_ab.to(device))
                
#                 # Inter-modal alignment losses
#                 multimodal_loss_ag_inter = torch.div(torch.sum(cos_ag_pos), torch.sum(cos_ag))
#                 multimodal_loss_ab_inter = torch.div(torch.sum(cos_ab_pos), torch.sum(cos_ab))
                
#                 # Combined multimodal loss with weighting
#                 multimodal_loss = 0.06 * (multimodal_loss_ag_inter + multimodal_loss_ab_inter + 
#                                         multimodal_loss_ag_intra_seq + multimodal_loss_ag_intra_struc + 
#                                         multimodal_loss_ab_intra_seq + multimodal_loss_ab_intra_strc + 
#                                         loss_NTXent(outputs[4], outputs[8]) + loss_NTXent(outputs[6], outputs[9]))
                
#                 # Total loss calculation with different component weights
#                 result_loss = 6 * loss_BCE((outputs[0]).squeeze(dim=1), ag_targets) + \
#                              6 * loss_BCE((outputs[1]).squeeze(dim=1), ab_targets) + \
#                              1 * multimodal_loss + \
#                              10 * (loss_BCE(outputs[2], edge_label)) + \
#                              10 * (loss_BCE(outputs[3].t(), edge_label))
                
#                 # Batch accumulation and update
#                 if ((i) % 16 == 0):
#                     result_loss_batch = result_loss
#                 else:
#                     result_loss_batch = result_loss + result_loss_batch
                
#                 # Update model parameters every 16 samples or at end of dataset
#                 if ((i + 1) % 16 == 0 or i == len(train_data_PECAN) - 1):
#                     optim.zero_grad()
#                     result_loss_batch.backward()
#                     optim.step()
                
#                 # Track training metrics
#                 result_loss_i = float(result_loss.item())
#                 result_loss_train_all = result_loss_train_all + result_loss_i
                
#                 # Aggregate outputs and targets for epoch evaluation
#                 output_ag = torch.flatten(outputs[0]) if i==0 else torch.cat((output_ag, torch.flatten(outputs[0])), dim=0)
#                 target_ag = ag_targets.long() if i==0 else torch.cat((target_ag, ag_targets.long()), dim=0)
#                 output_ab = torch.flatten(outputs[1]) if i==0 else torch.cat((output_ab, torch.flatten(outputs[1])), dim=0)
#                 target_ab = ab_targets.long() if i==0 else torch.cat((target_ab, ab_targets.long()), dim=0)

#                 # Update progress bar
#                 batch_iter.set_postfix({
#                     'batch_loss': f"{result_loss_i:.4f}",
#                     'avg_loss': f"{train_loss/(i+1):.4f}"
#                 })

#             ############################# Validation phase ############################# 
#             model.eval()  # Set model to evaluation mode
#             val_total_loss = 0
#             val_total_loss_all = 0
#             right_number = 0

#             val_loss = 0
#             val_outputs = []
#             val_targets = []

#             # Initialize validation metrics
#             val_auprc = 0.0
#             val_auroc = 0.0
#             val_precision = 0.0
#             val_recall = 0.0
#             val_f1 = 0.0
#             val_bacc = 0.0
#             val_mcc = 0.0
#             ab_val_auprc = 0.0
#             ab_val_auroc = 0.0
#             ab_val_precision = 0.0
#             ab_val_recall = 0.0
#             ab_val_f1 = 0.0
#             ab_val_bacc = 0.0
#             ab_val_mcc = 0.0

#             with torch.no_grad():  # Disable gradient calculation
#                 # for j in range(len(val_data_PECAN)):

#                 for j in tqdm(range(len(val_data_PECAN)), desc="Validating", leave=False):

#                     # Load validation data (similar to training data loading)
#                     ag_node_attr = torch.tensor(val_data_PECAN[j]["vertex_AG"], dtype=torch.float)
#                     ag_edge_ind = torch.tensor(val_data_PECAN[j]["edge_AG"], dtype=torch.long)
#                     ag_targets = torch.tensor(val_data_PECAN[j]["label_AG"], dtype=torch.float)
                    
#                     ab_node_attr = torch.tensor(val_data_PECAN[j]["vertex_AB"], dtype=torch.float)
#                     ab_edge_ind = torch.tensor(val_data_PECAN[j]["edge_AB"], dtype=torch.long)
#                     ab_targets = torch.tensor(val_data_PECAN[j]["label_AB"], dtype=torch.float)
                    
#                     # Move data to device
#                     ag_node_attr = ag_node_attr.to(device)
#                     ag_edge_ind = ag_edge_ind.to(device)
#                     ag_targets = ag_targets.to(device)
#                     ab_node_attr = ab_node_attr.to(device)
#                     ab_edge_ind = ab_edge_ind.to(device)
#                     ab_targets = ab_targets.to(device)
                    
#                     # Load protein language model embeddings
#                     ag_esm = torch.tensor(val_data_PECAN[j]["ESM1b_AG"]).to(device)
#                     ag_esm = ag_esm.t()
#                     ag_esm = torch.unsqueeze(ag_esm, dim=0)
#                     ab_esm = torch.tensor(val_data_PECAN[j]["AbLang_AB"]).to(device)
#                     ab_esm = ab_esm.t()
#                     ab_esm = torch.unsqueeze(ab_esm, dim=0)

#                     # Prepare input tensors
#                     ag_node_attr = torch.unsqueeze(ag_node_attr, dim=0)
#                     ab_node_attr = torch.unsqueeze(ab_node_attr, dim=0)
                    
#                     # Create graph structures
#                     ag_edge_ind, ab_edge_ind = CreateGearnetGraph(val_data_PECAN[j])
                    
#                     # Prepare model input
#                     agab = [ag_node_attr, ag_edge_ind, ab_node_attr, ab_edge_ind, ag_esm, ab_esm, False, ag_targets, ab_targets, i]
                    
#                     # Forward pass
#                     outputs = model(*agab)
                    
#                     # Create edge labels
#                     edge_label = np.zeros((ag_targets.shape[0], ab_targets.shape[0]))
#                     inter_edge_index = (val_data_PECAN[j]["edge_AGAB"])
#                     for edge_ind in range(len(inter_edge_index[0])):
#                         edge_label[inter_edge_index[0][edge_ind], inter_edge_index[1][edge_ind]] = 1.0
#                     edge_label = torch.tensor(edge_label, dtype=torch.float32).to(device)
                    
#                     # Multimodal loss calculation (similar to training)
#                     ag_out_seq = outputs[10]
#                     ag_out_strc = outputs[11]
#                     ab_out_seq = outputs[12]
#                     ab_out_strc = outputs[13]
                    
#                     multimodal_loss_ag_intra_seq = loss_NTXent(outputs[4], outputs[4])
#                     multimodal_loss_ag_intra_struc = loss_NTXent(outputs[8], outputs[8])
#                     multimodal_loss_ab_intra_seq = loss_NTXent(outputs[6], outputs[6])
#                     multimodal_loss_ab_intra_strc = loss_NTXent(outputs[9], outputs[9])
                    
#                     tag_ag = []
#                     tag_ab = []
#                     for tag_idx in range(len(val_data_PECAN[j]["label_AG"])):
#                         if ((val_data_PECAN[j]["label_AG"][tag_idx] and ag_out_seq[tag_idx]) or (
#                                 val_data_PECAN[j]["label_AG"][tag_idx] and ag_out_strc[tag_idx])):
#                             tag_ag.append(1.)
#                         else:
#                             tag_ag.append(0.)
#                     for tag_idx in range(len(val_data_PECAN[j]["label_AB"])):
#                         if ((val_data_PECAN[j]["label_AB"][tag_idx] and ab_out_seq[tag_idx]) or (
#                                 val_data_PECAN[j]["label_AB"][tag_idx] and ab_out_strc[tag_idx])):
#                             tag_ab.append(1.)
#                         else:
#                             tag_ab.append(0.)
                    
#                     cos_ag = consine_inter(outputs[4], outputs[8])
#                     cos_ab = consine_inter(outputs[6], outputs[9])
                    
#                     cos_ag_pos = torch.mul(torch.tensor(tag_ag).to(device), cos_ag.to(device))
#                     cos_ab_pos = torch.mul(torch.tensor(tag_ab).to(device), cos_ab.to(device))
                    
#                     multimodal_loss_ag_inter = torch.div(torch.sum(cos_ag_pos), torch.sum(cos_ag))
#                     multimodal_loss_ab_inter = torch.div(torch.sum(cos_ab_pos), torch.sum(cos_ab))
                    
#                     multimodal_loss = 0.06 * (multimodal_loss_ag_inter + multimodal_loss_ab_inter + 
#                                             multimodal_loss_ag_intra_seq + multimodal_loss_ag_intra_struc + 
#                                             multimodal_loss_ab_intra_seq + multimodal_loss_ab_intra_strc + 
#                                             loss_NTXent(outputs[4], outputs[8]) + loss_NTXent(outputs[6], outputs[9]))
                    
#                     # Validation loss calculation
#                     val_result_loss = 6 * loss_BCE((outputs[0]).squeeze(dim=1), ag_targets) + \
#                                      6 * loss_BCE((outputs[1]).squeeze(dim=1), ab_targets) + \
#                                      1 * multimodal_loss + \
#                                      10 * (loss_BCE(outputs[2], edge_label)) + \
#                                      10 * (loss_BCE(outputs[3].t(), edge_label))
                    
#                     # Track validation metrics
#                     val_result_loss_j = float(val_result_loss.item())
#                     val_total_loss_all = val_total_loss_all + val_result_loss_j

#                     # Aggregate validation outputs and targets
#                     output_ag_val = torch.flatten(outputs[0]) if j == 0 else torch.cat((output_ag_val, torch.flatten(outputs[0])), dim=0)
#                     target_ag_val = ag_targets.long() if j == 0 else torch.cat((target_ag_val, ag_targets.long()), dim=0)
#                     output_ab_val = torch.flatten(outputs[1]) if j == 0 else torch.cat((output_ab_val, torch.flatten(outputs[1])), dim=0)
#                     target_ab_val = ab_targets.long() if j == 0 else torch.cat((target_ab_val, ab_targets.long()), dim=0)

#                     val_loss += val_result_loss_j

#             # Calculate metrics
#             val_loss /= len(val_data_PECAN)
#             auprc_ag, auroc_ag, precision_ag, recall_ag, f1_ag, bacc_ag, mcc_ag = evalution_prot(
#                 output_ag_val, target_ag_val)
#             auprc_ab, auroc_ab, precision_ab, recall_ab, f1_ab, bacc_ab, mcc_ab = evalution_prot(
#                 output_ab_val, target_ab_val)

#             # Periodic evaluation and model saving
#             if ((e + 1) % 10 == 0):
#                 # Evaluate on training and validation sets
#                 (auprc_ag, auroc_ag, precision_ag, recall_ag, f1_ag, bacc_ag, mcc_ag) = evalution_prot(output_ag, target_ag)
#                 (auprc_ab, auroc_ab, precision_ab, recall_ab, f1_ab, bacc_ab, mcc_ab) = evalution_prot(output_ab, target_ab)
#                 (val_auprc_ag, val_auroc_ag, val_precision_ag, val_recall_ag, val_f1_ag, val_bacc_ag, val_mcc_ag) = evalution_prot(output_ag_val, target_ag_val)
#                 (val_auprc_ab, val_auroc_ab, val_precision_ab, val_recall_ab, val_f1_ab, val_bacc_ab, val_mcc_ab) = evalution_prot(output_ab_val, target_ab_val)                

#                 print("=============="+str((e + 1))+"=================")
#                 # Save model checkpoint
#                 # torch.save(model, "output_files/modelsave/model_k{}_{}".format(kfold, (e + 1)))

#                 torch.save(model, f"{results_dir}/model_k{kfold}_{e + 1}")

#                 Loss_val.append(val_total_loss_all / len(val_data_PECAN))
#                 AUROC_val.append(val_auroc / len(val_data_PECAN))

            
#             # Update best model
#             if auroc_ag > best_val_auroc:
#                 best_val_auroc = auroc_ag
#                 best_epoch = e
#                 torch.save(model.state_dict(), f"{results_dir}/best_model_k{kfold}.pt")
            
#             # Print epoch summary
#             epoch_time = time.time() - start_time
#             print(f"\nEpoch {e+1}/{epoch} | Time: {epoch_time:.2f}s")
#             print(f"Train Loss: {train_loss/len(train_data_PECAN):.4f} | Val Loss: {val_loss:.4f}")
#             print(f"AG Metrics - AUPRC: {auprc_ag:.4f} | AUROC: {auroc_ag:.4f} | Prec: {precision_ag:.4f} | Rec: {recall_ag:.4f} | F1: {f1_ag:.4f} | BAcc: {bacc_ag:.4f} | MCC: {mcc_ag:.4f}")
#             print(f"AB Metrics - AUPRC: {auprc_ab:.4f} | AUROC: {auroc_ab:.4f} | Prec: {precision_ab:.4f} | Rec: {recall_ab:.4f} | F1: {f1_ab:.4f} | BAcc: {bacc_ab:.4f} | MCC: {mcc_ab:.4f}")

#         ############################# Testing phase ############################# 

#         print(f"\n{'='*50}")
#         print(f"Testing Best Model (from epoch {best_epoch+1})")
#         print(f"{'='*50}")
        
#         model.load_state_dict(torch.load(f"{results_dir}/best_model_k{kfold}.pt"))
#         model.eval()
        
#         test_metrics = {
#             'ag': {'auprc': 0, 'auroc': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'bacc': 0, 'mcc': 0},
#             'ab': {'auprc': 0, 'auroc': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'bacc': 0, 'mcc': 0}
#         }
        
        
#         # # Select best model based on validation AUROC
#         # min_idx = AUROC_val.index(min(AUROC_val))
#         # min_idx = (min_idx + 1) * 10
#         # # model_filepath =  "output_files/modelsave/model_k" + str(kfold) + "_" + str(min_idx)

#         # model_filepath =  f"{results_dir}/model_k" + str(kfold) + "_" + str(min_idx)
        
#         # model = torch.load(model_filepath)
#         # model.eval()  # Set model to evaluation mode
        
#         # Initialize test metrics
#         test_auprc = 0.0
#         test_auroc = 0.0
#         test_precision = 0.0
#         test_recall = 0.0
#         test_mcc = 0.0
#         ab_test_auprc = 0.0
#         ab_test_auroc = 0.0
#         ab_test_precision = 0.0
#         ab_test_recall = 0.0
#         ab_test_mcc = 0.0

#         ag_h1_all = []  # For storing antigen representations
#         ab_h1_all = []  # For storing antibody representations

#         with torch.no_grad():  # Disable gradient calculation

#             test_iter = tqdm(range(len(test_data_PECAN)), desc="Testing")
#             for j in test_iter:

#             # for j in range(len(test_data_PECAN)):
#                 # Load test data (similar to training/validation)
#                 ag_node_attr = torch.tensor(test_data_PECAN[j]["vertex_AG"], dtype=torch.float)
#                 ag_edge_ind = torch.tensor(test_data_PECAN[j]["edge_AG"], dtype=torch.long)
#                 ag_targets = torch.tensor(test_data_PECAN[j]["label_AG"], dtype=torch.float)
                
#                 ab_node_attr = torch.tensor(test_data_PECAN[j]["vertex_AB"], dtype=torch.float)
#                 ab_edge_ind = torch.tensor(test_data_PECAN[j]["edge_AB"], dtype=torch.long)
#                 ab_targets = torch.tensor(test_data_PECAN[j]["label_AB"], dtype=torch.float)
                
#                 # Move data to device
#                 ag_node_attr = ag_node_attr.to(device)
#                 ag_edge_ind = ag_edge_ind.to(device)
#                 ag_targets = ag_targets.to(device)
#                 ab_node_attr = ab_node_attr.to(device)
#                 ab_edge_ind = ab_edge_ind.to(device)
#                 ab_targets = ab_targets.to(device)
                
#                 # Load protein language model embeddings
#                 ag_esm = torch.tensor(test_data_PECAN[j]["ESM1b_AG"]).to(device)
#                 ag_esm = ag_esm.t()
#                 ag_esm = torch.unsqueeze(ag_esm, dim=0)
#                 ab_esm = torch.tensor(test_data_PECAN[j]["AbLang_AB"]).to(device)
#                 ab_esm = ab_esm.t()
#                 ab_esm = torch.unsqueeze(ab_esm, dim=0)

#                 # Prepare input tensors
#                 ag_node_attr = torch.unsqueeze(ag_node_attr, dim=0)
#                 ab_node_attr = torch.unsqueeze(ab_node_attr, dim=0)
                
#                 # Create graph structures
#                 ag_edge_ind, ab_edge_ind = CreateGearnetGraph(test_data_PECAN[j])
                
#                 # Prepare model input (note: True flag for testing mode)
#                 agab = [ag_node_attr, ag_edge_ind, ab_node_attr, ab_edge_ind, ag_esm, ab_esm, True, ag_targets, ab_targets, j]
                
#                 # Forward pass
#                 outputs = model(*agab)
                
#                 # Evaluation on test set
#                 output_ag_test = torch.flatten(outputs[0]) if j == 0 else torch.cat((output_ag_test, torch.flatten(outputs[0])), dim=0)
#                 target_ag_test = ag_targets.long() if j == 0 else torch.cat((target_ag_test, ag_targets.long()), dim=0)
#                 output_ab_test = torch.flatten(outputs[1]) if j == 0 else torch.cat((output_ab_test, torch.flatten(outputs[1])), dim=0)
#                 target_ab_test = ab_targets.long() if j == 0 else torch.cat((target_ab_test, ab_targets.long()), dim=0)
                
#                 # Calculate test metrics
#                 test_auprc_ag, test_auroc_ag, test_precision_ag, test_recall_ag, test_f1_ag, test_bacc_ag, test_mcc_ag = evalution_prot(output_ag_test, target_ag_test)
#                 test_auprc_ab, test_auroc_ab, test_precision_ab, test_recall_ab, test_f1_ab, test_bacc_ab, test_mcc_ab = evalution_prot(output_ab_test, target_ab_test)                

#                 # Update test metrics accumulation
#                 test_metrics['ag']['auprc'] += test_auprc_ag
#                 test_metrics['ag']['auroc'] += test_auroc_ag
#                 test_metrics['ag']['precision'] += test_precision_ag
#                 test_metrics['ag']['recall'] += test_recall_ag
#                 test_metrics['ag']['f1'] += test_f1_ag
#                 test_metrics['ag']['bacc'] += test_bacc_ag
#                 test_metrics['ag']['mcc'] += test_mcc_ag

#                 test_metrics['ab']['auprc'] += test_auprc_ab
#                 test_metrics['ab']['auroc'] += test_auroc_ab
#                 test_metrics['ab']['precision'] += test_precision_ab
#                 test_metrics['ab']['recall'] += test_recall_ab
#                 test_metrics['ab']['f1'] += test_f1_ab
#                 test_metrics['ab']['bacc'] += test_bacc_ab
#                 test_metrics['ab']['mcc'] += test_mcc_ab

#                 test_iter.set_postfix({
#                     'AG_AUROC': f"{test_auroc_ag:.4f}",
#                     'AB_AUROC': f"{test_auroc_ab:.4f}"
#                 })

#         # Print final test results
#         print(f"\nFold {kfold+1} Final Test Results:")
#         print(f"AG - AUPRC: {test_metrics['ag']['auprc']/len(test_data_PECAN):.4f} | "
#             f"AUROC: {test_metrics['ag']['auroc']/len(test_data_PECAN):.4f} | "
#             f"Prec: {test_metrics['ag']['precision']/len(test_data_PECAN):.4f} | "
#             f"Rec: {test_metrics['ag']['recall']/len(test_data_PECAN):.4f} | "
#             f"F1: {test_metrics['ag']['f1']/len(test_data_PECAN):.4f} | "
#             f"BAcc: {test_metrics['ag']['bacc']/len(test_data_PECAN):.4f} | "
#             f"MCC: {test_metrics['ag']['mcc']/len(test_data_PECAN):.4f}")
#         print(f"AB - AUPRC: {test_metrics['ab']['auprc']/len(test_data_PECAN):.4f} | "
#             f"AUROC: {test_metrics['ab']['auroc']/len(test_data_PECAN):.4f} | "
#             f"Prec: {test_metrics['ab']['precision']/len(test_data_PECAN):.4f} | "
#             f"Rec: {test_metrics['ab']['recall']/len(test_data_PECAN):.4f} | "
#             f"F1: {test_metrics['ab']['f1']/len(test_data_PECAN):.4f} | "
#             f"BAcc: {test_metrics['ab']['bacc']/len(test_data_PECAN):.4f} | "
#             f"MCC: {test_metrics['ab']['mcc']/len(test_data_PECAN):.4f}")
        
#         # Save test results
#         """
#         TODO: [mansoor]
#         - save test results for each fold in csv file
#         """
#         csv_path = os.path.join(results_dir, "test_results.csv")
#         file_exists = os.path.exists(csv_path)

#         with open(csv_path, 'a', newline='') as f:
#             writer = csv.writer(f)
#             if not file_exists:
#                 writer.writerow([
#                     'fold', 'best_epoch',
#                     'ag_auprc', 'ag_auroc', 'ag_precision', 'ag_recall', 'ag_f1', 'ag_bacc', 'ag_mcc',
#                     'ab_auprc', 'ab_auroc', 'ab_precision', 'ab_recall', 'ab_f1', 'ab_bacc', 'ab_mcc'
#                 ])
            
#             num_samples = len(test_data_PECAN)
#             writer.writerow([
#                 kfold+1, best_epoch+1,
#                 round(test_metrics['ag']['auprc']/num_samples, 3),
#                 round(test_metrics['ag']['auroc']/num_samples, 3),
#                 round(test_metrics['ag']['precision']/num_samples, 3),
#                 round(test_metrics['ag']['recall']/num_samples, 3),
#                 round(test_metrics['ag']['f1']/num_samples, 3),
#                 round(test_metrics['ag']['bacc']/num_samples, 3),
#                 round(test_metrics['ag']['mcc']/num_samples, 3),
#                 round(test_metrics['ab']['auprc']/num_samples, 3),
#                 round(test_metrics['ab']['auroc']/num_samples, 3),
#                 round(test_metrics['ab']['precision']/num_samples, 3),
#                 round(test_metrics['ab']['recall']/num_samples, 3),
#                 round(test_metrics['ab']['f1']/num_samples, 3),
#                 round(test_metrics['ab']['bacc']/num_samples, 3),
#                 round(test_metrics['ab']['mcc']/num_samples, 3)
#             ])




# """"
# Example usage:
#     nohup python main.py > output_asep.log 2>&1 &

# """


# #################### from WALLE ###################


# import os
# import os.path as osp
# from pathlib import Path
# from pprint import pformat, pprint
# from typing import Any, Callable, Dict, List, Optional, Tuple

# import torch
# import torch.nn as nn
# import wandb
# import yaml
# from loguru import logger
# from torch import Tensor
# from torch.optim import lr_scheduler
# from torch.utils.tensorboard import SummaryWriter
# from torch_geometric.data import Batch as PygBatch
# from torch_geometric.loader import DataLoader as PygDataLoader
# from tqdm import tqdm

# # custom
# from asep.data.asepv1_dataset import AsEPv1Dataset
# from asep.data.embedding.handle import EmbeddingHandler
# from asep.data.embedding_config import EmbeddingConfig
# from asep.model import loss as loss_module
# from asep.model.asepv1_model import LinearAbAgIntGAE, PyGAbAgIntGAE
# from asep.model.callbacks import EarlyStopper, ModelCheckpoint
# from asep.model.metric import (cal_edge_index_bg_metrics,
#                                cal_epitope_node_metrics)
# from asep.model.utils import generate_random_seed, seed_everything
# from asep.utils import time_stamp

# # ==================== Configuration ====================
# # set precision
# torch.set_float32_matmul_precision("high")

# ESM2DIM = {
#     "esm2_t6_8M_UR50D": 320,
#     "esm2_t12_35M_UR50D": 480,
#     "esm2_t30_150M_UR50D": 640,
#     "esm2_t33_650M_UR50D": 1280,
# }

# proj_dir = "/Users/mansoor/Documents/GSU/Projects/Antibody-Design/epitope-prediction/"
# DataRoot = os.path.join(proj_dir, "data/")

# # DataRoot = Path.cwd().joinpath("data")


# # ==================== Function ====================
# # PREPARE: EmbeddingConfig
# def create_embedding_config(dataset_config: Dict[str, Any]) -> EmbeddingConfig:
#     """
#     Create embedding config from config dict

#     Args:
#         dataset_config (Dict[str, Any]): dataset config

#     Returns:
#         EmbeddingConfig: embedding config
#     """
#     # assert dataset_config is a primitive dict
#     try:
#         assert isinstance(dataset_config, dict)
#     except AssertionError as e:
#         raise TypeError(f"dataset_config must be a dict, instead got {type(dataset_config)}") from e

#     if dataset_config["node_feat_type"] in ("pre_cal", "one_hot"):
#         # parse the embedding model for ab and ag
#         d = dict(
#             node_feat_type=dataset_config["node_feat_type"],
#             ab=dataset_config["ab"].copy(),
#             ag=dataset_config["ag"].copy(),
#         )
#         return EmbeddingConfig(**d)
    
#     # Handle custom embeddings
#     try:
#         ab_src = dataset_config["ab"]["custom_embedding_method_src"]
#         ab_func = EmbeddingHandler(
#             script_path=ab_src["script_path"], function_name=ab_src["method_name"]
#         ).embed
#     except Exception as e:
#         raise RuntimeError("Error loading custom embedding method for Ab.") from e

#     try:
#         ag_src = dataset_config["ag"]["custom_embedding_method_src"]
#         ag_func = EmbeddingHandler(
#             script_path=ag_src["script_path"], function_name=ag_src["method_name"]
#         ).embed
#     except Exception as e:
#         raise RuntimeError("Error loading custom embedding method for Ag.") from e

#     updated_dataset_config = dataset_config.copy()
#     updated_dataset_config["ab"]["custom_embedding_method"] = ab_func
#     updated_dataset_config["ag"]["custom_embedding_method"] = ag_func
#     return EmbeddingConfig(**updated_dataset_config)

#     # # otherwise, node_feat_type is custom, need to load function from user specified script
#     # try:
#     #     # print(dataset_config)
#     #     d = dataset_config["ab"]["custom_embedding_method_src"]
#     #     ab_func = EmbeddingHandler(
#     #         script_path=d["script_path"], function_name=d["method_name"]
#     #     ).embed
#     # except Exception as e:
#     #     raise RuntimeError(
#     #         "Error loading custom embedding method for Ab. Please check the script."
#     #     ) from e
#     # try:
#     #     d = dataset_config["ag"]["custom_embedding_method_src"]
#     #     ag_func = EmbeddingHandler(
#     #         script_path=d["script_path"], function_name=d["method_name"]
#     #     ).embed
#     # except Exception as e:
#     #     raise RuntimeError(
#     #         "Error loading custom embedding method for Ag. Please check the script."
#     #     ) from e
#     # updated_dataset_config = dataset_config.copy()
#     # updated_dataset_config["ab"]["custom_embedding_method"] = ab_func
#     # updated_dataset_config["ag"]["custom_embedding_method"] = ag_func
#     # return EmbeddingConfig(**updated_dataset_config)


# # PREPARE: dataset
# def create_asepv1_dataset(
#     root: str = None,
#     name: str = None,
#     embedding_config: EmbeddingConfig = None,
# ):
#     """
#     Create AsEPv1 dataset

#     Args:
#         root (str, optional): root directory for dataset. Defaults to None.
#             if None, set to './data'
#         name (str, optional): dataset name. Defaults to None.
#             if None, set to 'asep'
#         embedding_config (EmbeddingConfig, optional): embedding config. Defaults to None.
#             if None, use default embedding config
#             {
#                 'node_feat_type': 'pre_cal',
#                 'ab': {'embedding_model': 'igfold'},
#                 'ag': {'embedding_model': 'esm2'},
#             }

#     Returns:
#         AsEPv1Dataset: AsEPv1 dataset
#     """
#     root = root if root is not None else "./data"
#     embedding_config = embedding_config or EmbeddingConfig()
#     asepv1_dataset = AsEPv1Dataset(
#         root=root, name=name, embedding_config=embedding_config
#     )

#     # print(f"Dataset created with root: {root}, name: {name}, embedding_config: {embedding_config}")  # Debugging

#     return asepv1_dataset


# # PREPARE: dataloaders
# def create_asepv1_dataloaders(
#     asepv1_dataset: AsEPv1Dataset,
#     wandb_run: wandb.sdk.wandb_run.Run = None,
#     config: Dict[str, Any] = None,
#     split_method: str = None,
#     split_idx: Dict[str, Tensor] = None,
#     return_dataset: bool = False,
#     dev: bool = False,
# ) -> Tuple[PygDataLoader, PygDataLoader, PygDataLoader]:
#     """
#     Create dataloaders for AsEPv1 dataset

#     Args:
#         wandb_run (wandb.sdk.wandb_run.Run, optional): wandb run object. Defaults to None.
#         config (Dict[str, Any], optional): config dict. Defaults to None.
#         return_dataset (bool, optional): return dataset instead of dataloaders. Defaults to False.
#         dev (bool, optional): use dev mode. Defaults to False.
#         split_idx (Dict[str, Tensor], optional): split index. Defaults to None.
#     AsEPv1Dataset kwargs:
#         embedding_config (EmbeddingConfig, optional): embedding config. Defaults to None.
#             If None, use default EmbeddingConfig, for details, see asep.data.embedding_config.EmbeddingConfig.
#         split_method (str, optional): split method. Defaults to None. Either 'epitope_ratio' or 'epitope_group'

#     Returns:
#         Tuple[PygDataLoader, PygDataLoader, PygDataLoader]: _description_
#     """
#     # split dataset
#     split_idx = split_idx or asepv1_dataset.get_idx_split(split_method=split_method)
#     train_set = asepv1_dataset[split_idx["train"]]
#     val_set = asepv1_dataset[split_idx["val"]]
#     test_set = asepv1_dataset[split_idx["test"]]

#     # if dev, only use 100 samples
#     if dev:
#         train_set = train_set[:170]
#         val_set = val_set  # [:100]
#         test_set = test_set  # [:100]

#     # patch: if test_batch_size is not specified, use val_batch_size, otherwise use test_batch_size
#     if ("test_batch_size" not in config["hparams"].keys()) or (
#         config["hparams"]["test_batch_size"] is None
#     ):
#         config["hparams"]["test_batch_size"] = config["hparams"]["val_batch_size"]
#         print(
#             f"WARNING: test_batch_size is not specified, use val_batch_size instead: {config['hparams']['test_batch_size']}"
#         )

#     _default_kwargs = dict(follow_batch=["x_b", "x_g"], shuffle=False)
#     _default_kwargs_train = dict(
#         batch_size=config["hparams"]["train_batch_size"], **_default_kwargs
#     )
#     _default_kwargs_val = dict(
#         batch_size=config["hparams"]["val_batch_size"], **_default_kwargs
#     )
#     _default_kwargs_test = dict(
#         batch_size=config["hparams"]["test_batch_size"], **_default_kwargs
#     )
#     train_loader = PygDataLoader(train_set, **_default_kwargs_train)
#     val_loader = PygDataLoader(val_set, **_default_kwargs_val)
#     test_loader = PygDataLoader(test_set, **_default_kwargs_test)

#     # Check the dimensions of the embeddings in the batches
#     for batch in train_loader:
#         print(f"Train batch x_b shape: {batch.x_b.shape}")
#         print(f"Train batch x_g shape: {batch.x_g.shape}")
#         break

#     for batch in val_loader:
#         print(f"Val batch x_b shape: {batch.x_b.shape}")
#         print(f"Val batch x_g shape: {batch.x_g.shape}")
#         break

#     for batch in test_loader:
#         print(f"Test batch x_b shape: {batch.x_b.shape}")
#         print(f"Test batch x_g shape: {batch.x_g.shape}")
#         break


#     # save a train-set example to wandb
#     if wandb_run is not None:
#         artifact = wandb.Artifact(
#             name="train_set_example", type="dataset", description="train set example"
#         )
#         with artifact.new_file("train_set_example.pt", "wb") as f:
#             torch.save(train_set[0], f)
#         wandb_run.log_artifact(artifact)

#     if not return_dataset:
#         return train_loader, val_loader, test_loader
#     return train_set, val_set, test_set, train_loader, val_loader, test_loader


# # PREPARE: model
# def create_model(
#     config: Dict[str, Any], wandb_run: wandb.sdk.wandb_run.Run = None
# ) -> nn.Module:
#     if config["hparams"]["model_type"] == "linear":
#         model_architecture = LinearAbAgIntGAE
#     elif config["hparams"]["model_type"] == "graph":
#         model_architecture = PyGAbAgIntGAE
#     else:
#         raise ValueError("model must be either 'linear' or 'graph'")
#     # create the model
#     model = model_architecture(
#         input_ab_dim=config["hparams"]["input_ab_dim"],
#         input_ag_dim=config["hparams"]["input_ag_dim"],
#         input_ab_act=config["hparams"]["input_ab_act"],
#         input_ag_act=config["hparams"]["input_ag_act"],
#         dim_list=config["hparams"]["dim_list"],
#         act_list=config["hparams"]["act_list"],
#         decoder=config["hparams"]["decoder"],
#         try_gpu=config["try_gpu"],
#     )
#     if wandb_run is not None:
#         wandb_run.watch(model)

#     # print(f"Model created with input_ab_dim: {config['hparams']['input_ab_dim']}, input_ag_dim: {config['hparams']['input_ag_dim']}")  # Debugging

#     return model


# # PREPARE: loss callables
# def generate_loss_callables_from_config(
#     loss_config: Dict[str, Any],
# ) -> List[Tuple[str, Callable, Tensor, Dict[str, Any]]]:
#     for loss_name, kwargs in loss_config.items():
#         try:
#             assert "name" in kwargs.keys() and "w" in kwargs.keys()
#         except AssertionError as e:
#             raise KeyError("each loss term must contain keys 'name' and 'w'") from e

#     loss_callables: List[Tuple[str, Callable, Tensor, Dict[str, Any]]] = [
#         (
#             name := kwargs.get("name"),  # loss name
#             getattr(loss_module, name),  # loss function callable
#             torch.tensor(kwargs["w"]),  # loss weight
#             kwargs.get("kwargs", {}),  # other kwargs
#         )
#         for loss_name, kwargs in loss_config.items()
#     ]
#     return loss_callables


# # RUN: feed forward step
# def feed_forward_step(
#     model: nn.Module,
#     batch: PygBatch,
#     loss_callables: List[Tuple[str, Callable, Tensor, Dict]],
#     is_train: bool,
#     edge_cutoff: Optional[int] = None,
#     num_edge_cutoff: Optional[int] = None,
# ) -> Tuple[Tensor, Dict[str, Tensor], Dict[str, Tensor]]:
#     """
#     Feed forward and calculate loss & metrics for a batch of AbAg graph pairs

#     Args:
#         batch (Dict): a batch of AbAg graph pairs
#         model (nn.Module): model to be trained
#         loss_callables (List[Tuple[str, Callable, Tensor, Dict, Dict]]):
#             loss_name: (str)        => used as key in outputs
#             loss_fn: (Callable)     => the loss function callable
#             loss_wt: (Tensor)       => the weight of the loss function for calculating total loss
#             loss_fn_kwargs: (Dict)  => kwargs that are constant values

#     Returns:
#         Dict: outputs from model and loss
#     """
#     if is_train:
#         model.train()
#     else:
#         model.eval()

#     # feed forward
#     batch_result = model(batch)
#     edge_index_bg_pred = batch_result["edge_index_bg_pred"]
#     edge_index_bg_true = batch_result["edge_index_bg_true"]

#     # unpack loss_callables
#     # loss_items = {}
#     batch_loss = None
#     for loss_name, loss_fn, loss_w, loss_kwargs in loss_callables:
#         if loss_name == "edge_index_bg_rec_loss":
#             loss_v = [
#                 loss_fn(x, y, **loss_kwargs)
#                 for x, y in zip(edge_index_bg_pred, edge_index_bg_true)
#             ]
#         elif loss_name == "edge_index_bg_sum_loss":
#             loss_v = [loss_fn(x, **loss_kwargs) for x in edge_index_bg_pred]
#         # loss_items |= {loss_name: {"v": loss_v, "w": loss_w,}}
#         batch_loss = (
#             torch.stack(loss_v) * loss_w
#             if batch_loss is None
#             else batch_loss + torch.stack(loss_v) * loss_w
#         )
#         # # log loss values
#         # print(f"loss name : {loss_name}")
#         # print(f"loss weight: {loss_w}")
#         # print(f"raw      loss value: mean {torch.stack(loss_v).mean()}, std {torch.stack(loss_v).std()}")
#         # print(f"weighted loss value: mean {batch_loss.mean()}, std {batch_loss.std()}")

#     # metrics
#     batch_edge_index_bg_metrics: List[Dict[str, Tensor]] = [
#         cal_edge_index_bg_metrics(x, y, edge_cutoff)
#         for x, y in zip(edge_index_bg_pred, edge_index_bg_true)
#     ]
#     batch_edge_epi_node_metrics: List[Dict[str, Tensor]] = [
#         cal_epitope_node_metrics(x, y, num_edge_cutoff)
#         for x, y in zip(edge_index_bg_pred, edge_index_bg_true)
#     ]

#     # average loss and metrics
#     avg_loss: Tensor = batch_loss.mean()
#     avg_edge_index_bg_metrics: Dict[str, Tensor] = {
#         k: torch.stack([d[k] for d in batch_edge_index_bg_metrics]).mean()
#         for k in batch_edge_index_bg_metrics[0].keys()
#     }
#     avg_epi_node_metrics: Dict[str, Tensor] = {
#         k: torch.stack([d[k] for d in batch_edge_epi_node_metrics]).mean()
#         for k in batch_edge_epi_node_metrics[0].keys()
#     }

#     return avg_loss, avg_edge_index_bg_metrics, avg_epi_node_metrics


# # CALLBACK: on after backward
# def on_after_backward(model: nn.Module):
#     """Log gradients and model parameters norm after each backward pass"""
#     for name, param in model.named_parameters():
#         wandb.log(
#             {f"gradients/{name}": param.grad.norm(), f"params/{name}": param.norm()}
#         )


# # CALLBACK: on epoch end
# def epoch_end(
#     step_outputs: List[Tuple[Tensor, Dict[str, Tensor], Dict[str, Tensor]]]
# ) -> Tuple[Tensor, Dict[str, Tensor], Dict[str, Tensor]]:
#     """
#     Args:
#         step_outputs (List[Dict[str, Tensor]]):
#             shape (n x m)
#             `n` element list of outputs from each step (batch)
#             each element is a tuple of `m` elements - loss or metrics
#     """
#     with torch.no_grad():
#         # calculate average loss
#         avg_epoch_loss = torch.stack([x[0] for x in step_outputs]).mean()
#         # calculate average metrics
#         avg_epoch_edge_index_bg_metrics = {
#             k: torch.stack([x[1][k] for x in step_outputs]).mean()
#             for k in step_outputs[0][1].keys()
#         }
#         avg_epoch_epi_node_metrics = {
#             k: torch.stack([x[2][k] for x in step_outputs]).mean()
#             for k in step_outputs[0][2].keys()
#         }
#         return (
#             avg_epoch_loss,
#             avg_epoch_edge_index_bg_metrics,
#             avg_epoch_epi_node_metrics,
#         )


# # TRAIN helper - learning rate scheduler
# def exec_lr_scheduler(
#     ck_lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
#     config: Dict[str, Any],
#     val_epoch_metrics: Dict[str, Tensor],
# ) -> None:
#     if ck_lr_scheduler is not None:
#         if config["callbacks"]["lr_scheduler"]["step"] is not None:
#             # reduce learning rate on plateau
#             if config["callbacks"]["lr_scheduler"]["name"] == "ReduceLROnPlateau":
#                 ck_lr_scheduler.step(
#                     metrics=val_epoch_metrics[
#                         config["callbacks"]["lr_scheduler"]["step"]["metrics"]
#                     ]
#                 )
#         else:
#             ck_lr_scheduler.step()



# # MAIN function
# def train_model(
#     config: Dict,
#     wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
#     tb_writer: Optional[SummaryWriter] = None,
# ):
#     """
#     Args:
#         config: (Dict) config dict, contains all hyperparameters
#         wandb_run: (wandb.sdk.wandb_run.Run) wandb run object
#     """
#     # For debugging purpose
#     logger.debug(f"config:\n{pformat(config)}")
#     # Set num threads
#     torch.set_num_threads(config["num_threads"])

#     # --------------------
#     # Datasets
#     # --------------------
#     dev = config.get("mode") == "dev"
#     embedding_config = create_embedding_config(dataset_config=config["dataset"])
#     asepv1_dataset = create_asepv1_dataset(
#         root=config["dataset"]["root"],
#         name=config["dataset"]["name"],
#         embedding_config=embedding_config,
#     )
#     train_loader, val_loader, test_loader = create_asepv1_dataloaders(
#         asepv1_dataset=asepv1_dataset,
#         wandb_run=wandb_run,
#         config=config,
#         split_idx=config["dataset"]["split_idx"],
#         split_method=config["dataset"]["split_method"],
#         dev=dev,
#     )
#     print(f"{len(train_loader.dataset)=}")
#     print(f"{len(val_loader.dataset)=}")
#     print(f"{len(test_loader.dataset)=}")

#     # --------------------
#     # Model, Loss, Optimizer, Callbacks
#     # --------------------
#     model = create_model(config=config, wandb_run=wandb_run)

#     # Log the model architecture
#     if wandb_run is not None:
#         wandb_run.watch(model)
#     # Print out the model architecture
#     print(model)

#     # Loss and optimizer
#     loss_callables = generate_loss_callables_from_config(config["loss"])
#     optimizer = getattr(torch.optim, config["optimizer"]["name"])(
#         params=model.parameters(),
#         **config["optimizer"]["params"],
#     )

#     # Callback objects
#     ck_early_stop = (
#         EarlyStopper(**config["callbacks"]["early_stopping"])
#         if config["callbacks"]["early_stopping"] is not None
#         else None
#     )
#     # Add a model checkpoint to record best k models for node prediction
#     ck_model_ckpt = (
#         ModelCheckpoint(**config["callbacks"]["model_checkpoint"])
#         if config["callbacks"]["model_checkpoint"] is not None
#         else None
#     )
#     # Add a model checkpoint to record best k models for edge prediction
#     ck_model_ckpt_edge = (
#         ModelCheckpoint(**config["callbacks"]["model_checkpoint_edge"])
#         if config["callbacks"]["model_checkpoint_edge"] is not None
#         else None
#     )
#     # Add a learning rate scheduler
#     ck_lr_scheduler = (
#         getattr(lr_scheduler, config["callbacks"]["lr_scheduler"]["name"])(
#             optimizer=optimizer, **config["callbacks"]["lr_scheduler"]["kwargs"]
#         )
#         if config["callbacks"]["lr_scheduler"] is not None
#         else None
#     )

#     # --------------------
#     # Train Val Test Loop
#     # --------------------
#     train_step_outputs, val_step_outputs, test_step_outputs = [], [], []
#     current_epoch_idx, current_val_metric = None, None
#     for epoch_idx in range(config["hparams"]["max_epochs"]):
#         current_epoch_idx = epoch_idx
#         print(f"Epoch {epoch_idx + 1}/{config['hparams']['max_epochs']}")

#         # --------------------
#         # Training
#         # --------------------
#         model.train()
#         _default_kwargs = dict(unit="GraphPairBatch", ncols=100)
#         for batch_idx, batch in tqdm(
#             enumerate(train_loader),
#             total=len(train_loader),
#             desc=f"{'train':<5}",
#             **_default_kwargs,
#         ):
#             optimizer.zero_grad()
#             # Feed forward (batch)
#             # print(f"Batch Ab input shape: {batch.x_b.shape}, Batch Ab input shape: {batch.x_g.shape}" )
#             avg_loss, avg_edge_index_bg_metrics, avg_epi_node_metrics = feed_forward_step(
#                 model=model,
#                 batch=batch,
#                 loss_callables=loss_callables,
#                 is_train=True,
#                 edge_cutoff=config["hparams"]["edge_cutoff"],
#                 num_edge_cutoff=config["hparams"]["num_edge_cutoff"],
#             )
#             d = {
#                 "trainStep/avg_loss": avg_loss,
#                 "trainStep/avg_edge_index_bg_auprc": avg_edge_index_bg_metrics["auprc"],
#                 "trainStep/avg_edge_index_bg_auroc": avg_edge_index_bg_metrics["auroc"],  # Added AUC-ROC
#                 "trainStep/avg_edge_index_bg_mcc": avg_edge_index_bg_metrics["mcc"],
#                 "trainStep/avg_edge_index_bg_tn": avg_edge_index_bg_metrics["tn"],
#                 "trainStep/avg_edge_index_bg_fp": avg_edge_index_bg_metrics["fp"],
#                 "trainStep/avg_edge_index_bg_fn": avg_edge_index_bg_metrics["fn"],
#                 "trainStep/avg_edge_index_bg_tp": avg_edge_index_bg_metrics["tp"],
#                 "trainStep/avg_epi_node_auprc": avg_epi_node_metrics["auprc"],
#                 "trainStep/avg_epi_node_auroc": avg_epi_node_metrics["auroc"],  # Added AUC-ROC
#                 "trainStep/avg_epi_node_mcc": avg_epi_node_metrics["mcc"],
#                 "trainStep/avg_epi_node_tn": avg_epi_node_metrics["tn"],
#                 "trainStep/avg_epi_node_fp": avg_epi_node_metrics["fp"],
#                 "trainStep/avg_epi_node_fn": avg_epi_node_metrics["fn"],
#                 "trainStep/avg_epi_node_tp": avg_epi_node_metrics["tp"],
#             }
#             if wandb_run is not None:
#                 wandb_run.log(d)
#             elif tb_writer is not None:
#                 tb_writer.add_scalars(
#                     main_tag="train",
#                     tag_scalar_dict=d,
#                     global_step=epoch_idx * len(train_loader) + batch_idx,
#                 )
#             # Append to step outputs
#             train_step_outputs.append(
#                 (avg_loss, avg_edge_index_bg_metrics, avg_epi_node_metrics)
#             )
#             # Calculate gradients
#             avg_loss.backward()
#             # Update parameters
#             optimizer.step()

#         # Epoch end: calculate epoch average loss and metric
#         avg_epoch_loss, avg_epoch_edge_index_bg_metrics, avg_epoch_epi_node_metrics = epoch_end(
#             step_outputs=train_step_outputs
#         )
#         train_epoch_metrics = {
#             "trainEpoch/avg_loss": avg_epoch_loss,
#             "trainEpoch/avg_edge_index_bg_auprc": avg_epoch_edge_index_bg_metrics["auprc"],
#             "trainEpoch/avg_edge_index_bg_auroc": avg_epoch_edge_index_bg_metrics["auroc"],  # Added AUC-ROC
#             "trainEpoch/avg_edge_index_bg_mcc": avg_epoch_edge_index_bg_metrics["mcc"],
#             "trainEpoch/avg_edge_index_bg_tn": avg_epoch_edge_index_bg_metrics["tn"],
#             "trainEpoch/avg_edge_index_bg_fp": avg_epoch_edge_index_bg_metrics["fp"],
#             "trainEpoch/avg_edge_index_bg_fn": avg_epoch_edge_index_bg_metrics["fn"],
#             "trainEpoch/avg_edge_index_bg_tp": avg_epoch_edge_index_bg_metrics["tp"],
#             "trainEpoch/avg_epi_node_auprc": avg_epoch_epi_node_metrics["auprc"],
#             "trainEpoch/avg_epi_node_auroc": avg_epoch_epi_node_metrics["auroc"],  # Added AUC-ROC
#             "trainEpoch/avg_epi_node_mcc": avg_epoch_epi_node_metrics["mcc"],
#             "trainEpoch/avg_epi_node_tn": avg_epoch_epi_node_metrics["tn"],
#             "trainEpoch/avg_epi_node_fp": avg_epoch_epi_node_metrics["fp"],
#             "trainEpoch/avg_epi_node_fn": avg_epoch_epi_node_metrics["fn"],
#             "trainEpoch/avg_epi_node_tp": avg_epoch_epi_node_metrics["tp"],
#             "epoch": epoch_idx + 1,
#         }
#         if wandb_run is not None:
#             wandb_run.log(train_epoch_metrics)
#         elif tb_writer is not None:
#             tb_writer.add_scalars(
#                 main_tag="train",
#                 tag_scalar_dict=train_epoch_metrics,
#                 global_step=epoch_idx,
#             )
#         pprint(train_epoch_metrics)
#         # Free memory
#         train_step_outputs.clear()

#         # --------------------
#         # Validation
#         # --------------------
#         model.eval()
#         for batch_idx, batch in tqdm(
#             enumerate(val_loader),
#             total=len(val_loader),
#             desc=f"{'val':<5}",
#             unit="GraphPairBatch",
#             ncols=100,
#         ):
#             # Feed forward (batch)
#             avg_loss, avg_edge_index_bg_metrics, avg_epi_node_metrics = feed_forward_step(
#                 model=model,
#                 batch=batch,
#                 loss_callables=loss_callables,
#                 is_train=False,
#                 edge_cutoff=config["hparams"]["edge_cutoff"],
#                 num_edge_cutoff=config["hparams"]["num_edge_cutoff"],
#             )
#             d = {
#                 "valStep/avg_loss": avg_loss,
#                 "valStep/avg_edge_index_bg_auprc": avg_edge_index_bg_metrics["auprc"],
#                 "valStep/avg_edge_index_bg_auroc": avg_edge_index_bg_metrics["auroc"],  # Added AUC-ROC
#                 "valStep/avg_edge_index_bg_mcc": avg_edge_index_bg_metrics["mcc"],
#                 "valStep/avg_edge_index_bg_tn": avg_edge_index_bg_metrics["tn"],
#                 "valStep/avg_edge_index_bg_fp": avg_edge_index_bg_metrics["fp"],
#                 "valStep/avg_edge_index_bg_fn": avg_edge_index_bg_metrics["fn"],
#                 "valStep/avg_edge_index_bg_tp": avg_edge_index_bg_metrics["tp"],
#                 "valStep/avg_epi_node_auprc": avg_epi_node_metrics["auprc"],
#                 "valStep/avg_epi_node_auroc": avg_epi_node_metrics["auroc"],  # Added AUC-ROC
#                 "valStep/avg_epi_node_mcc": avg_epi_node_metrics["mcc"],
#                 "valStep/avg_epi_node_tn": avg_epi_node_metrics["tn"],
#                 "valStep/avg_epi_node_fp": avg_epi_node_metrics["fp"],
#                 "valStep/avg_epi_node_fn": avg_epi_node_metrics["fn"],
#                 "valStep/avg_epi_node_tp": avg_epi_node_metrics["tp"],
#             }
#             if wandb_run is not None:
#                 wandb_run.log(d)
#             elif tb_writer is not None:
#                 tb_writer.add_scalars(
#                     main_tag="val",
#                     tag_scalar_dict=d,
#                     global_step=epoch_idx * len(val_loader) + batch_idx,
#                 )
#             # Append to step outputs
#             val_step_outputs.append(
#                 (avg_loss, avg_edge_index_bg_metrics, avg_epi_node_metrics)
#             )

#         # Epoch end: calculate epoch average loss and metric
#         avg_epoch_loss, avg_epoch_edge_index_bg_metrics, avg_epoch_epi_node_metrics = epoch_end(
#             step_outputs=val_step_outputs
#         )
#         val_epoch_metrics = {
#             "valEpoch/avg_loss": avg_epoch_loss,
#             "valEpoch/avg_edge_index_bg_auprc": avg_epoch_edge_index_bg_metrics["auprc"],
#             "valEpoch/avg_edge_index_bg_auroc": avg_epoch_edge_index_bg_metrics["auroc"],  # Added AUC-ROC
#             "valEpoch/avg_edge_index_bg_mcc": avg_epoch_edge_index_bg_metrics["mcc"],
#             "valEpoch/avg_edge_index_bg_tn": avg_epoch_edge_index_bg_metrics["tn"],
#             "valEpoch/avg_edge_index_bg_fp": avg_epoch_edge_index_bg_metrics["fp"],
#             "valEpoch/avg_edge_index_bg_fn": avg_epoch_edge_index_bg_metrics["fn"],
#             "valEpoch/avg_edge_index_bg_tp": avg_epoch_edge_index_bg_metrics["tp"],
#             "valEpoch/avg_epi_node_auprc": avg_epoch_epi_node_metrics["auprc"],
#             "valEpoch/avg_epi_node_auroc": avg_epoch_epi_node_metrics["auroc"],  # Added AUC-ROC
#             "valEpoch/avg_epi_node_mcc": avg_epoch_epi_node_metrics["mcc"],
#             "valEpoch/avg_epi_node_tn": avg_epoch_epi_node_metrics["tn"],
#             "valEpoch/avg_epi_node_fp": avg_epoch_epi_node_metrics["fp"],
#             "valEpoch/avg_epi_node_fn": avg_epoch_epi_node_metrics["fn"],
#             "valEpoch/avg_epi_node_tp": avg_epoch_epi_node_metrics["tp"],
#             "epoch": epoch_idx + 1,
#         }
#         if wandb_run is not None:
#             wandb_run.log(val_epoch_metrics)
#         elif tb_writer is not None:
#             tb_writer.add_scalars(
#                 main_tag="val",
#                 tag_scalar_dict=val_epoch_metrics,
#                 global_step=epoch_idx,
#             )
#         pprint(val_epoch_metrics)
#         # Free memory
#         val_step_outputs.clear()

#         # --------------------
#         # Testing
#         # --------------------
#         model.eval()
#         for batch_idx, batch in tqdm(
#             enumerate(test_loader),
#             total=len(test_loader),
#             desc=f"{'test':<5}",
#             unit="GraphPairBatch",
#             ncols=100,
#         ):
#             # Feed forward (batch)
#             avg_loss, avg_edge_index_bg_metrics, avg_epi_node_metrics = feed_forward_step(
#                 model=model,
#                 batch=batch,
#                 loss_callables=loss_callables,
#                 is_train=False,
#                 edge_cutoff=config["hparams"]["edge_cutoff"],
#                 num_edge_cutoff=config["hparams"]["num_edge_cutoff"],
#             )
#             d = {
#                 "testStep/avg_loss": avg_loss,
#                 "testStep/avg_edge_index_bg_auprc": avg_edge_index_bg_metrics["auprc"],
#                 "testStep/avg_edge_index_bg_auroc": avg_edge_index_bg_metrics["auroc"],  # Added AUC-ROC
#                 "testStep/avg_edge_index_bg_mcc": avg_edge_index_bg_metrics["mcc"],
#                 "testStep/avg_edge_index_bg_tn": avg_edge_index_bg_metrics["tn"],
#                 "testStep/avg_edge_index_bg_fp": avg_edge_index_bg_metrics["fp"],
#                 "testStep/avg_edge_index_bg_fn": avg_edge_index_bg_metrics["fn"],
#                 "testStep/avg_edge_index_bg_tp": avg_edge_index_bg_metrics["tp"],
#                 "testStep/avg_epi_node_auprc": avg_epi_node_metrics["auprc"],
#                 "testStep/avg_epi_node_auroc": avg_epi_node_metrics["auroc"],  # Added AUC-ROC
#                 "testStep/avg_epi_node_mcc": avg_epi_node_metrics["mcc"],
#                 "testStep/avg_epi_node_tn": avg_epi_node_metrics["tn"],
#                 "testStep/avg_epi_node_fp": avg_epi_node_metrics["fp"],
#                 "testStep/avg_epi_node_fn": avg_epi_node_metrics["fn"],
#                 "testStep/avg_epi_node_tp": avg_epi_node_metrics["tp"],
#             }
#             if wandb_run is not None:
#                 wandb_run.log(d)
#             elif tb_writer is not None:
#                 tb_writer.add_scalars(
#                     main_tag="test",
#                     tag_scalar_dict=d,
#                     global_step=epoch_idx * len(test_loader) + batch_idx,
#                 )
#             # Append to step outputs
#             test_step_outputs.append(
#                 (avg_loss, avg_edge_index_bg_metrics, avg_epi_node_metrics)
#             )

#         # Epoch end: calculate epoch average loss and metric
#         avg_epoch_loss, avg_epoch_edge_index_bg_metrics, avg_epoch_epi_node_metrics = epoch_end(
#             step_outputs=test_step_outputs
#         )
#         test_epoch_metrics = {
#             "testEpoch/avg_loss": avg_epoch_loss,
#             "testEpoch/avg_edge_index_bg_auprc": avg_epoch_edge_index_bg_metrics["auprc"],
#             "testEpoch/avg_edge_index_bg_auroc": avg_epoch_edge_index_bg_metrics["auroc"],  # Added AUC-ROC
#             "testEpoch/avg_edge_index_bg_mcc": avg_epoch_edge_index_bg_metrics["mcc"],
#             "testEpoch/avg_edge_index_bg_tn": avg_epoch_edge_index_bg_metrics["tn"],
#             "testEpoch/avg_edge_index_bg_fp": avg_epoch_edge_index_bg_metrics["fp"],
#             "testEpoch/avg_edge_index_bg_fn": avg_epoch_edge_index_bg_metrics["fn"],
#             "testEpoch/avg_edge_index_bg_tp": avg_epoch_edge_index_bg_metrics["tp"],
#             "testEpoch/avg_epi_node_auprc": avg_epoch_epi_node_metrics["auprc"],
#             "testEpoch/avg_epi_node_auroc": avg_epoch_epi_node_metrics["auroc"],  # Added AUC-ROC
#             "testEpoch/avg_epi_node_mcc": avg_epoch_epi_node_metrics["mcc"],
#             "testEpoch/avg_epi_node_tn": avg_epoch_epi_node_metrics["tn"],
#             "testEpoch/avg_epi_node_fp": avg_epoch_epi_node_metrics["fp"],
#             "testEpoch/avg_epi_node_fn": avg_epoch_epi_node_metrics["fn"],
#             "testEpoch/avg_epi_node_tp": avg_epoch_epi_node_metrics["tp"],
#             "epoch": epoch_idx + 1,
#         }
#         if wandb_run is not None:
#             wandb_run.log(test_epoch_metrics)
#         elif tb_writer is not None:
#             tb_writer.add_scalars(
#                 main_tag="test",
#                 tag_scalar_dict=test_epoch_metrics,
#                 global_step=epoch_idx,
#             )
#         pprint(test_epoch_metrics)
#         # Free memory
#         test_step_outputs.clear()

#         # --------------------
#         # Callbacks
#         # --------------------
#         # Model checkpoint
#         if ck_model_ckpt is not None:
#             ck_model_ckpt.step(
#                 metrics=val_epoch_metrics,
#                 model=model,
#                 epoch=epoch_idx,
#                 optimizer=optimizer,
#             )
#         # Model checkpoint edge level
#         if ck_model_ckpt_edge is not None:
#             ck_model_ckpt_edge.step(
#                 metrics=val_epoch_metrics,
#                 model=model,
#                 epoch=epoch_idx,
#                 optimizer=optimizer,
#             )
#         # Early stopping
#         if (ck_early_stop is not None) and (
#             ck_early_stop.early_stop(epoch=epoch_idx, metrics=val_epoch_metrics)
#         ):
#             print(f"Early stopping at epoch {epoch_idx}")
#             break

#         exec_lr_scheduler(ck_lr_scheduler, config, val_epoch_metrics=val_epoch_metrics)
#         for param_group in optimizer.param_groups:
#             print(f"Epoch {epoch_idx+1}, Learning Rate: {param_group['lr']:.6f}")

#     # --------------------
#     # Save models (train finished or early stopped)
#     # --------------------
#     # Save the last model
#     ck_model_ckpt.save_last(
#         epoch=current_epoch_idx,
#         model=model,
#         optimizer=optimizer,
#         metric_value=val_epoch_metrics[
#             config["callbacks"]["model_checkpoint"]["metric_name"]
#         ],
#         upload=True,  # Upload to wandb
#         wandb_run=wandb_run,  # Wandb run object
#     )

#     # Save best k models based on provided metric_name
#     ck_model_ckpt.save_best_k(keep_interim=config["keep_interim_ckpts"])
#     ck_model_ckpt_edge.save_best_k(keep_interim=config["keep_interim_ckpts"])
#     # Upload models to wandb artifacts
#     if wandb_run is not None:
#         ck_model_ckpt.upload_best_k_to_wandb(wandb_run=wandb_run)
#         ck_model_ckpt_edge.upload_best_k_to_wandb(wandb_run=wandb_run, suffix='-edge')

#     # --------------------
#     # Testing
#     # --------------------
#     # Load the best model
#     ckpt_data = ck_model_ckpt.load_best()
#     model.load_state_dict(ckpt_data["model_state_dict"])

#     # Test time
#     model.eval()
#     test_step_outputs = []
#     with torch.no_grad():
#         for batch_idx, batch in tqdm(
#             enumerate(test_loader),
#             total=len(test_loader),
#             desc=f"{'testF':<5}",
#             unit="graph",
#             ncols=100,
#         ):
#             # Feed forward (batch)
#             avg_loss, avg_edge_index_bg_metrics, avg_epi_node_metrics = feed_forward_step(
#                 model=model,
#                 batch=batch,
#                 loss_callables=loss_callables,
#                 is_train=False,
#                 edge_cutoff=config["hparams"]["edge_cutoff"],
#                 num_edge_cutoff=config["hparams"]["num_edge_cutoff"],
#             )
#             d = {
#                 "testStepFinal/avg_loss": avg_loss,
#                 "testStepFinal/avg_edge_index_bg_auprc": avg_edge_index_bg_metrics["auprc"],
#                 "testStepFinal/avg_edge_index_bg_auroc": avg_edge_index_bg_metrics["auroc"],  # Added AUC-ROC
#                 "testStepFinal/avg_edge_index_bg_mcc": avg_edge_index_bg_metrics["mcc"],
#                 "testStepFinal/avg_edge_index_bg_tn": avg_edge_index_bg_metrics["tn"],
#                 "testStepFinal/avg_edge_index_bg_fp": avg_edge_index_bg_metrics["fp"],
#                 "testStepFinal/avg_edge_index_bg_fn": avg_edge_index_bg_metrics["fn"],
#                 "testStepFinal/avg_edge_index_bg_tp": avg_edge_index_bg_metrics["tp"],
#                 "testStepFinal/avg_epi_node_auprc": avg_epi_node_metrics["auprc"],
#                 "testStepFinal/avg_epi_node_auroc": avg_epi_node_metrics["auroc"],  # Added AUC-ROC
#                 "testStepFinal/avg_epi_node_mcc": avg_epi_node_metrics["mcc"],
#                 "testStepFinal/avg_epi_node_tn": avg_epi_node_metrics["tn"],
#                 "testStepFinal/avg_epi_node_fp": avg_epi_node_metrics["fp"],
#                 "testStepFinal/avg_epi_node_fn": avg_epi_node_metrics["fn"],
#                 "testStepFinal/avg_epi_node_tp": avg_epi_node_metrics["tp"],
#             }
#             if wandb_run is not None:
#                 wandb_run.log(d)
#             elif tb_writer is not None:
#                 tb_writer.add_scalars(
#                     main_tag="test",
#                     tag_scalar_dict=d,
#                     global_step=epoch_idx * len(test_loader) + batch_idx,
#                 )
#             # Append to step outputs
#             test_step_outputs.append(
#                 (avg_loss, avg_edge_index_bg_metrics, avg_epi_node_metrics)
#             )

#         # Epoch end: calculate epoch average loss and metric
#         avg_epoch_loss, avg_epoch_edge_index_bg_metrics, avg_epoch_epi_node_metrics = epoch_end(
#             step_outputs=test_step_outputs
#         )
#         test_epoch_metrics = {
#             "testEpochFinal/avg_loss": avg_epoch_loss,
#             "testEpochFinal/avg_edge_index_bg_auprc": avg_epoch_edge_index_bg_metrics["auprc"],
#             "testEpochFinal/avg_edge_index_bg_auroc": avg_epoch_edge_index_bg_metrics["auroc"],  # Added AUC-ROC
#             "testEpochFinal/avg_edge_index_bg_mcc": avg_epoch_edge_index_bg_metrics["mcc"],
#             "testEpochFinal/avg_edge_index_bg_tn": avg_epoch_edge_index_bg_metrics["tn"],
#             "testEpochFinal/avg_edge_index_bg_fp": avg_epoch_edge_index_bg_metrics["fp"],
#             "testEpochFinal/avg_edge_index_bg_fn": avg_epoch_edge_index_bg_metrics["fn"],
#             "testEpochFinal/avg_edge_index_bg_tp": avg_epoch_edge_index_bg_metrics["tp"],
#             "testEpochFinal/avg_epi_node_auprc": avg_epoch_epi_node_metrics["auprc"],
#             "testEpochFinal/avg_epi_node_auroc": avg_epoch_epi_node_metrics["auroc"],  # Added AUC-ROC
#             "testEpochFinal/avg_epi_node_mcc": avg_epoch_epi_node_metrics["mcc"],
#             "testEpochFinal/avg_epi_node_tn": avg_epoch_epi_node_metrics["tn"],
#             "testEpochFinal/avg_epi_node_fp": avg_epoch_epi_node_metrics["fp"],
#             "testEpochFinal/avg_epi_node_fn": avg_epoch_epi_node_metrics["fn"],
#             "testEpochFinal/avg_epi_node_tp": avg_epoch_epi_node_metrics["tp"],
#         }
#         if wandb_run is not None:
#             wandb_run.log(test_epoch_metrics)
#         elif tb_writer is not None:
#             tb_writer.add_scalars(
#                 main_tag="test",
#                 tag_scalar_dict=test_epoch_metrics,
#                 global_step=epoch_idx,
#             )
#         # Free memory
#         test_step_outputs.clear()



