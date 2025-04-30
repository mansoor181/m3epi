# WP4 – Data Augmentation & Self-Supervised Learning

---

## 4-A  Graph augmentations + contrastive pre-train
| file | change |
|------|--------|
| `model/aug_transforms.py` | ➕  `EdgeDrop`, `NodeMask`, `CoordJitter` (applied inside `PairData`). |
| `pretrain.py` | ➕  implements **DGI** and **GraphCL** pipelines; loads either real AsEP graphs or AF-Multimer pseudo graphs. |
| `conf/mode/pretrain.yaml` | new mode with `epochs=200` and `contrastive.name=dgi`. |
| `main.py` | ▸ if `cfg.mode.mode == "pretrain"` call `pretrain_loop()` and exit. |

## 4-B  AlphaFold-Multimer pseudo-corpus
| file | change |
|------|--------|
| `data/generate_afm_pairs.py` | ➕  downloads UniRef90 pairs, prepares FASTA. |
| `data/run_afm.sh` | ➕  SLURM script to run AF-Multimer on cluster. |
| `preprocess.py` | ▸ add `--pseudo_dir` flag; parse PDBs and emit graphs to `data/pseudo_graphs/`. |

## 4-C  Multi-task paratope + epitope
| file | change |
|------|--------|
| `model/model.py` | ▸ add `self.para_head` (MLP) and output key `'paratope_prob'`. |
| `model/loss.py`  | ▸ add `multitask_loss(epi_pred, para_pred, epi_y, para_y, λ_para)`. |
| `conf/loss/multitask.yaml` | ➕  `lambda_para: 0.3`. |
| `main.py`        | ▸ switch loss if `cfg.model.multitask`. |

---

## Validation matrix

| exp id | pre-train | augment | multi-task | notes |
|--------|-----------|---------|------------|-------|
| A0 | ✗ | ✗ | ✗ | WP3 baseline |
| A1 | DGI (real) | ✓ | ✗ | |
| A2 | GraphCL (real) | ✓ | ✗ | |
| A3 | DGI (pseudo) | ✓ | ✗ | |
| A4 | best of A1–A3 | ✓ | ✓ | final WP4 |

Each run: 3 seeds × 5-fold CV.  Log to W&B group **wp4_ssl**.

---

## Done / To-do table
- [ ] Implement `aug_transforms.py`
- [ ] `pretrain.py` (DGI, GraphCL)
- [ ] AF-Multimer dataset generation
- [ ] Multi-task heads & loss
- [ ] Retrain + log WP4 metrics
