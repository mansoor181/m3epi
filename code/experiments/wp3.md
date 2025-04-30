# WP3 – Representation & Multi-Modal Fusion

---

## 3-A  Import sequence-only priors
| file | change |
|------|--------|
| `data/preprocess.py` | ▸ Run ParaAntiProt / ESMBind, save scores in pickle field `seq_prior` (shape `[N]`). |
| `conf/model/model.yaml` | ▸ New switch `use_seq_prior: true`. |
| `model/model.py` | ▸ During graph construction, append `seq_prior.unsqueeze(1)` to Ag/Ab node features when flag is on. |

## 3-B  Geometric descriptors
| file | change |
|------|--------|
| `data/generate_pssm.py` | ▸ Call MSMS/STRIDE; write 6-D `geo_feat` per residue. |
| `model/model.py` | ▸ Concatenate `geo_feat` channels if `cfg.model.use_geo_feat`. |
| `conf/model/model.yaml` | ▸ `use_geo_feat: true`. |

## 3-C  Dual-channel encoders + fusion
| file | change |
|------|--------|
| `model/model.py` | ▸ Split features into `x_plm` & `x_vec`; feed into two `GraphEncoder`s. <br>▸ Call `fusion_layer()` ⇒ fused embedding. |
| `model/feature_fusion.py` | ➕ `ConcatFusion`, `GatedFusion`, `CrossAttentionFusion`. |
| `conf/model/model.yaml` | ▸ Section: `fusion: {type: concat, dim: 64}`. |
| `ablation.py` | ▸ Allow override `fusion.type=gate` etc. |

## 3-D  Relative position buckets
| file | change |
|------|--------|
| `data/preprocess.py` | ▸ Compute distance bucket id for every edge, store `edge_attr`. |
| `model/model.py` | ▸ Use `edge_attr` in GAT (`edge_dim=edge_attr_dim`). |
| `conf/model/model.yaml` | ▸ `edge.pos_bucket: true` & bucket boundaries. |

## 3-E  Hyper-edge message passing
| file | change |
|------|--------|
| `model/hyperconv.py` | Implement small `HyperEdgeConv`. |
| `model/model.py` | ▸ If `cfg.model.use_hyper=true`, add one `HyperEdgeConv` layer after GNN encoders. |
| `conf/model/model.yaml` | ▸ `use_hyper: true`, `hyper.k: 3`, `hyper.hidden: 32`. |

---

## Testing & Validation
1. **Pre-processing:**  
   ```bash
   python data/preprocess.py --add_seq_prior --add_geo_feat
    ```

1. **Smoke run**:
    ```python
    python main.py mode=dev \
    model.fusion.type=cross_attn \
    model.use_geo_feat=true \
    model.edge.pos_bucket=true \
    wandb.tags="[WP3,smoke]"
    ```

## Experiment Grid for WP3

| ID   | Fusion | Geo Feat | Seq Prior | Hyper-Edge | Expected                  |
|------|--------|----------|-----------|------------|----------------------------|
| F-1  | concat | ✗       | ✗        | ✗         | Baseline (WP2 weights)   |
| F-2  | concat | ✓       | ✗        | ✗         | Geom only                |
| F-3  | gate   | ✓       | ✓        | ✗         | Dual-view fused           |
| F-4  | xattn  | ✓       | ✓        | ✓         | Full WP3 model           |

### Tracking
- Track all runs in W&B under the "wp3_fusion" group.
- Export aggregated CSV to `results/wp3/`.
