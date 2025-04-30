
# WP1 – Implementation Checklist
(Refer to experiments.tex for scientific rationale; this file is the engineering to-do.)

## 1-A  Equivariant GNN
| file | action |
|------|--------|
| `model/egnn_layer.py` | ➕ new lightweight EGNN/SE3 layer (use `torch_geometric.nn.EGNN` or your own). |
| `model/model.py` | • Extend `GraphLayer` factory: accept `"EGNN"` / `"SE3"`. <br>• When the selected model is equivariant, pass `coords` along with features in `forward`. |
| `main.py` | • In `PairData` add tensors `coord_g`, `coord_b` (Ag / Ab Cartesian). <br>• When building the dataloader, copy coordinates from `vertex_*` arrays or separate `coord_*` keys. |
| `conf/model/model.yaml` | • Add `name: "EGNN"` option and a boolean `equivariant: true`. <br>• If equivariant, remove residual flag (not needed). |
| `conf/hparams/hparams.yaml` | • Increase `train.batch_size` if EGNN is lighter than GAT; otherwise keep. |

## 1-B  Surface-patch graphs
| file | action |
|------|--------|
| `data/preprocess.py` | • After loading Ag PDB, call MaSIF pipeline to triangulate and compute 6-D patch features. <br>• Build a k-NN (or radius) graph on patches, store as `patch_vertex` + `patch_edge`. |
| `model/surface_net.py` | ➕ tiny PointNet/MLP that encodes patch features to 64-D. |
| `model/model.py` | • If `cfg.data.use_surface`, run `surface_net` and **concatenate** its output to antigen residue embeddings before decoding. |
| `conf/model/model.yaml` | • Add `encoder.antigen.input_dim` += 64 when surface is used. |
| `conf/config.yaml` | • Introduce flag `data.use_surface: false` (default); override to `true` in experiments. |

## 1-C  Upgraded PLMs
| file | action |
|------|--------|
| `data/preprocess.py` | • Pre-compute AntiBERTy-14B heavy+light, and ESM-2-650M embeddings; save into `plm_cache/*.pt` keyed by PDB id. |
| `utils.py` → `load_data` | • When `cfg.data.use_new_plm`, replace old embeddings with cached tensors. |
| `model/model.py` | • Insert a gating MLP: `concat([plm, geom]) → σ(α)·plm + (1-σ(α))·geom`. |
| `conf/model/model.yaml` | • Update `input_dim` for antibody (now 14B→1024) and antigen (650M→1280). |
| `conf/config.yaml` | • Flag `data.use_new_plm: false` (override to true in the experiment). |

---

## Testing & Logging
1. **Unit tests**:  
   * `pytest tests/test_egnn_forward.py` (batch invariance, equivariance check).  
   * `pytest tests/test_surface_graph.py` (number of patches ≥ surface residues).

2. **Quick smoke run**  
   ```bash
   python main.py mode=dev \
       model.name=EGNN \
       data.use_surface=true \
       data.use_new_plm=true \
       wandb.tags="[WP1,smoke]"
    ```
Target: completes 2 epochs on 10 complexes without error.

Full CV experiment (for paper table)

```bash
python main.py \
    model.name=EGNN \
    data.use_surface=true \
    data.use_new_plm=true \
    hparams.train.num_epochs=200 \
    wandb.notes="WP1 full run"
```

