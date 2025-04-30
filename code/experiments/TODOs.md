
### TODO: [04-15-25]

1. create and add different GNN models: GCN, GAT, GIN
    - make changes in the following files:
        - model.py
        - config yaml files: hparams, model
    - try other related strategies such as residual connections
1. add new losses:
    1. contrastive learning InfoNCE loss (https://github.com/WangZhiwei9/MIPE.git)
    2. gradient-weighted NCE loss (https://github.com/RingBDStack/ReGCL.git)
    - update files: loss config, main.py
1. modes of experiments: dev, train, sweep (tuning)
    - update main.py: 
        - sliced data loading for dev
        - wandb sweep and config files for tuning

### TODO: [04-18-25]

1. data configurations: 
    - PLM-based node embeddings: `plm`
    - MIPE-like node embeddings such as one-hot, aa_profile, etc: `vec`
    - files to update: preprocess.py
    - try different dataset split settings: random, antigen-to-epitope ratio
    NOTE:
    - how about use two different encoders?
        - one for processing `plm` and one for `vec`
        - then pass through inner product decoder and take average of the adjacency matrices
1. generate t-SNE plots for visualizing the embedding projections:
    - binding vs non-binding nodes
    - ag-ab bipartite graph edges
    - see the effect of different loss functions and models
1. only save the best model checkpoint for each experiment with proper filename:
    - create folders for gnn model names, e.g., GCN, GAT, GIN, ...
    - update callbacks.py
    - filename: gnn_loss_epoch_lr_mcc.pt

### TODO: [04-20-25]

1. save results csv file when performing sweep experiments or in train mode
    - when doing k-fold cross validation, also create a summary csv file
    where the results for that particular experiment are averaged (mean ± std)
1. how to perform ablation studies and analyze the results? 
    - what tables and figures? AUROC, AUPRC plots
    - shell script for ablation studies?
    - abltation studies:
        1. model type: GCN, GAT, GIN, GraphSAGE, 
            TODO: implement other GNNs such as LSPE 
            (learning structural and positional encodings), GraphWalk, etc
        2. loss: CE only, CE + infonce, CE + gwnce
        3. decoder type: inner product, MLP, cross attention
    - aggregate results for five random seeds
1. save and report train time in results: update main code

### TODO: [04-22-25]

1. make contrastive losses (infonce or gwnce) a config choice
    - see the impact of CE only, CE + infonce, CE + gwnce
1. add test model code and perform train-test split
1. create full asep graphs dataset and run ablation studies

### TODO: [04-26-25]

1. modify the decoder output interaction map from max probability thresholding to thresholding of row-wise sum of probabilities 

1. run sweep experiments for best ablation run
  - run sweeps in parallel on multiple gpus
  - run ablation experiments in parallel on multiple gpus

### TODO: [04-28-25]

1. self-contained results filenames for ablation experiments

1. add pre-trained sequence-based binding site prediction models to generate m-dimensional sequence embeddings
  - protein-ligand models for antigen: ESMBind
  - paratope prediction models for antibody: ParaAntiProt, Paragraph
  1. fuse these sequence embeddings with GNN embeddings
    - files to modify: 
      - preprocess.py=> add the embedding vectors to tensor pkl file
          - after generating sequence embeddings, mask using seqres2cdr and seqres2surf
      - model.py=> concat the sequence and graph embeddings using cross attention before decoder
  1. append the sequence embeddings with graph node embedding and then pass through GNN
    - files to modify are preprocess.py
  1. fine-tune 

1. epitope prediction for antigen alone (antibody-agnostic)
    - exp settings: ag_alone, complex
    - tasks: epi_node, bipartite_link

### TODO: [high priority]

1. add analysis notebooks: 
    - plot distributions of the classification metrics for each test sample (as in MaSIF analysis notebooks)
    - draw architectures for each WP
    

<!-- ### Explore:
- self-supervised GNNs: graph augmentations such as removing and predicting nodes, edges 
- predict graph descriptors such as node degree, edges, etc
- k-mean clustering in graphs for binding nodes vs non-binding nodes 
- hypergraph substructure and molecular fingerprints
- create edge graph for bipartite link prediction -->


## Completed TODOs:

### 8 Single best checkpoint per run, nice file-names

| Step | Details |
|------|---------|
| **Why** | Make results reproducible and easy to grep when you have dozens of runs. |
| **Code Touch** | `model/callbacks.py` |
| **Details** | - Add `arg run_name` so the folder is `.../checkpoints/<run_name>/`<br>- Compose filename:<br>`fname = f"{cfg.model.name}_{cfg.loss.contrastive.name}_E{epoch:02d}_lr{cfg.hparams.train.learning_rate:.0e}_mcc{val_mcc:.3f}.pt"` |
| **Config** | Add to `callbacks.yaml`:<br>`dirpath: ${results_dir}/checkpoints/${now:%Y%m%d_%H%M}_${mode}` |
| **W&B** | Log an artifact with the same name – you can download it later in one click. |
| **Deliverable** | Table S1 in appendix that lists run-ID ↔ checkpoint-file. |

### 9 Results CSV per run and k-fold summary

| Step | Details |
|------|---------|
| **Why** | Automatable stats & pretty tables without re-running Python. |
| **Code Touch** | At the end of `main.py` after the `print("\n=== Final ===")` block:<br><br>```python<br>import pandas as pd, csv, pathlib<br>row = {<br>    "run_id": wandb.run.id if cfg.logging_method=="wandb" else datetime.now().isoformat(),<br>    "model": cfg.model.name,<br>    "loss": cfg.loss.contrastive.name,<br>    "lr": cfg.hparams.train.learning_rate,<br>    **{f"mean_{k}": v for k,v in avg.items()},<br>    **{f"std_{k}": v for k,v in std.items()}<br>}<br>out_csv = pathlib.Path(cfg.results_dir) / "summary.csv"<br>pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)<br>``` |
| **Analysis** | Python/pandas one-liner produces LaTeX table from `summary.csv`. |

### 10 Ablation-study scaffolding

#### 10.1 Dimensions of the Ablation

| Factor | Values |
|--------|--------|
| GNN architecture | GCN, GAT, GIN, GraphSAGE, (future: LSPE, GraphWalk) |
| Decoder | Dot-product, MLP (1–2 hidden layers), Cross-Attention |
| Loss | CE only, CE + InfoNCE, CE + GW-NCE |
| Seeds | 5 distinct random seeds |

#### 10.2 Experimental Matrix

Loop over every combination of (GNN, Decoder, Loss).

For each:
- In train mode, run 5 × k-fold (e.g. 5 seeds × 5 folds = 25 runs).
- Collect and average:
  - Node metrics: AUROC, AUPRC, MCC
  - Edge metrics if desired
  - Training time
- Save:
  - Per-fold CSV (as above)
  - One aggregated CSV over seeds: `.../ablation/<model>/<decoder>/<loss>/summary/agg_across_seeds.csv`

#### 10.3 Figures & Tables

- Bar charts of “mean ± std” for each metric across architectures / losses / decoders.
- Line plots or heatmaps to show interaction (e.g. GNN vs Loss).
- ROC / PR curves for the top 3 configurations.

#### 10-A Shell helper

```bash
# ablate.sh
models=(GCN GAT GIN)           # later add GraphSAGE …
losses=(ce infonce gwnce)
for m in "${models[@]}"; do
  for l in "${losses[@]}"; do
    python main.py model.name=\$m loss.contrastive.name=$l \
        wandb.tags="[ablation]" wandb.name="${m}_\${l}"
  done
done
```
#### 10-B Hydra template
- Add conf/mode/ablation.yaml:

```yaml
mode: train
hparams:
  train:
    num_epochs: 10
    kfolds: 3
wandb:
  tags: ["ablation"]
```


#### 10-C Paper Artefacts
- **Table 2**: MCC/AUPRC per (model, loss) ± std
- **Figure 3**: Bar-plot of ΔMCC versus CE-only baseline

### 11 Explicit Test Split

| Step       | Details |
|------------|---------|
| Data       | Pre-split dataset into train/val/test indices once, save to `split.pkl` so every model sees the same test set |
| Config     | `conf/mode/test.yaml`: `mode: test`, `hparams.train.kfolds: 1` |
| main.py    | If `cfg.mode=="test"` skip CV and evaluate once on the frozen test set. Return metrics JSON so CI can assert performance ≥ threshold |
| Deliverable | Section "Generalisation" – report blind-test MCC & 95% CI |

### 12 Wall-time Profiling

| Step       | Details |
|------------|---------|
| Code Touch | Wrap epoch loops with `time.perf_counter()`; accumulate `train_sec` and `val_sec` |
| W&B logging | `wandb.log({"epoch_train_time": train_t, "epoch_val_time": val_t})` |
| Analysis   | Scatter plot training-time vs MCC over all runs to show efficiency trade-off (Figure S4) |

### 13 Contrastive-loss as True Config Knob
Already implemented in:
- `loss.yaml`
- `main.py`

### Extra Polish
1. Provide `loss.contrastive.name: "none"` path that skips InfoNCE entirely
2. In `visualize_embeddings.py`, set dot-color shape depending on loss (for "latent-space separation improves with InfoNCE" figure)

### Good-practice Checklist
(Applies to every experiment)

| Item | Rationale | Tip |
|------|-----------|-----|
| 1 seed ≠ research | Show robustness | Always run 3-5 seeds (Hydra sweep: `seed=range(3)`) |
| Determinism | Reviewers love it | `torch.use_deterministic_algorithms(True)` during test |
| Unit tests | Stops silent NaNs | `pytest -q tests/` before every sweep |
| Tag everything | Future-you will forget | `wandb.tags: ["paper_v1", cfg.model.name, cfg.loss.contrastive.name]` |
| Notebook-less plots | CI-friendly | Save PDF/PNG via Matplotlib, script in `/analysis/` |

### Suggested Execution Order
1. Smoke-test: `mode=dev` on laptop (< 1 min)
2. Baseline grid: Run `ablate.sh` on a single GPU node overnight
3. Pick best trio (model, loss, hyper-params) → run `mode=train` with 5 folds × 5 seeds
4. Final blind test with the best checkpoint
5. Embedding t-SNE using that checkpoint
6. Write results: Tables + figures straight from CSV/PNG folders