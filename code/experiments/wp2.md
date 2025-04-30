
# WP2 – Implementation Checklist
Concrete file-level actions.

---

## 2-A  Adaptive / Focal BCE
| file | action |
|------|--------|
| `model/loss.py` | • Add `focal_bce(pred, target, gamma, alpha)` implementation. |
| `conf/loss/loss.yaml` | • New keys:<br>`node_prediction.type: focal`<br>`focal.gamma: 2` |
| `main.py` | • In `train_epoch` / `validate_epoch` route to `focal_bce` when `cfg.loss.node_prediction.type == "focal"`. |

## 2-B  Edge-Aware Auxiliary Loss
| file | action |
|------|--------|
| `conf/loss/loss.yaml` | • Add `edge_prediction.lambda: 0.3`. |
| `main.py` | • Multiply `edge_loss *= cfg.loss.edge_prediction.lambda`. |
| `ablation.py` | • Allow override `loss.edge_prediction.lambda=0.0` to switch it off. |

## 2-C  Self-Supervised Warm-up (GraphCL)
| file | action |
|------|--------|
| `loss/graphcl.py` | ➕ GraphCL NT-Xent loss on two augmented graphs. |
| `model/model.py` | • Add `augment_graph(batch, mode)` util (edge-drop, feature-mask). |
| `main.py` | • If `cfg.loss.selfsup.pretrain_epochs > 0`, run pre-train loop **before** CV fine-tuning; save weights to temp file and reload. |
| `conf/loss/loss.yaml` | • Section: `selfsup: {pretrain_epochs: 50, temperature: 0.2}`. |

## 2-D  Modern LR Schedules & SWA
| file | action |
|------|--------|
| `conf/hparams/hparams.yaml` | • Add `optim:` block with<br>&nbsp;&nbsp;`scheduler: cosine`<br>&nbsp;&nbsp;`warmup_steps: 500`. |
| `main.py` | • Create `torch.optim.lr_scheduler.CosineAnnealingLR`. <br>• If `cfg.optim.swa.enabled`, wrap optimizer with `torch.optim.swa_utils.SWALoader` after warm-up. |
| `model/callbacks.py` | • New `SWACallback` to swap averaged weights at the end. |

## 2-E  GradNorm for Loss Balancing
| file | action |
|------|--------|
| `model/loss.py` | • Add `GradNormBalancer` class that keeps per-task weights. |
| `main.py` | • Replace static `loss = node + edge + cl` with `balancer.step({"node": node_loss, ...})`. |
| `conf/loss/loss.yaml` | • Boolean flag `gradnorm: true`. |

---

## Testing
1. **Unit** – `pytest tests/test_focal_bce.py`, `test_gradnorm.py`.
2. **Smoke** –  
   ```bash
   python main.py mode=dev \
          loss.node_prediction.type=focal \
          loss.edge_prediction.lambda=0.3 \
          optim.scheduler=cosine \
          wandb.tags="[WP2,smoke]"
    ```

| Experiment ID | New Loss                | Scheduler    | Self-Sup Pretrain | Comment                |
|---------------|-------------------------|--------------|-------------------|------------------------|
| L1            | focal γ=2               | cosine       | ✗                 | baseline imbalance fix |
| L2            | focal γ=2               | cosine+SWA   | ✗                 | stability              |
| L3            | focal γ=2 + edge λ=0.3  | cosine+SWA   | ✗                 | edge aux               |
| L4            | same as L3              | cosine+SWA   | GraphCL 50ep      | warm-up                |

Record each as a separate W&B run; aggregate in WP2_losses.csv.

