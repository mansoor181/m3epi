<!-- A Python module for paratope and epitope prediction
![image](https://github.com/WangZhiwei9/MIPE/blob/main/Overview.jpg) -->

# M3Epi: Multi-modal, multi-task, multi-network (GNN, LSTM, CL) Epitope Prediction

<!-- ## Requirements

This project relies on specific Python packages to ensure its proper functioning. The required packages and their versions are listed in the `requirements.txt` file. -->

<!-- ## Data

Dataset Files (pickle format) can be downloaded from: https://drive.google.com/drive/folders/1bvGZQnOs6XOA17NsiaZ4eVjvn94SOM3u?usp=drive_link -->

<!-- - Antibody embeddings generated using [AbLang](https://github.com/oxpig/AbLang.git) -->


## Code

Our code files are packaged in zip format, and the directory structure is as follows.

```
code/
├── conf/
│   ├── config.yaml
│   ├── sweep.yaml
│   ├── callbacks/callbacks.yaml
│   ├── hparams/hparams.yaml
│   ├── loss/loss.yaml
│   ├── model/model.yaml
│   ├── metric/metric.yaml
│   ├── wandb/wandb.yaml
│   └── mode/
│       ├── dev.yaml
│       └── train.yaml
├── data/
│   ├── generate_pssm.py
│   ├── preprocess.py
│   ├── preprocess.ipynb
├── model/
│   ├── __init__.py
│   ├── callbacks.py
│   ├── CrossAttention.py
│   ├── loss.py
│   ├── metric.py
│   └── model.py
├── m3epi.ipynb
├── main.py
├── sweep.py
├── visualize_embeddings.py
├── utils.py
├─requirements.txt
└─README.md
```

```
results/
  ablation/         # select the best model/framework setup based on CV and seeds 
    <model_name>/           # e.g. GIN/
      <decoder_type>/           # e.g. MLP/Dot
        <loss>/             # or “ce+infonce”
          checkpoints/
            GIN_mlp_ce+gwnce_ep05_lr1e-3_bs32_val_loss0.1234.pt
          summary/        
            GIN_mlp_ce+gwnce_ep05_lr1e-3_bs32_val_loss0.1234.csv # all runs
            GIN_mlp_ce+gwnce_ep05_lr1e-3_bs32_val_loss0.1234_cv_aggregated.csv # k-fold cv aggregated
            GIN_mlp_ce+gwnce_results_summary.csv # aggregated for multiple seeds
  sweep/      # tune the hyperparameters of the best model 
    sweep_id/
      run1.yaml
      run2.yaml
  final/    # train the selected best models on the sweeped hparams and generate results summary stats
    <model_name>/           # e.g. GIN/
      <decoder_type>/           # e.g. MLP/Dot
        <loss>/             # or “ce+infonce”
          checkpoints/      # don't save checkpoints for sweeps until best config is found
            GIN_mlp_ce+gwnce_ep05_lr1e-3_bs32_val_loss0.1234.pt
          summary/        
            GIN_mlp_ce+gwnce_ep05_lr1e-3_bs32_val_loss0.1234.csv # all runs
            GIN_mlp_ce+gwnce_ep05_lr1e-3_bs32_val_loss0.1234_cv_aggregated.csv # k-fold cv aggregated
            GIN_mlp_ce+gwnce_results_summary.csv # aggregated for multiple seeds
  figures/
```

Training for the M3Epi model with the dataset

```
python main.py
```


PSI-BLAST installation:
- sudo apt-get install ncbi-blast+
- wget https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz
- gzip -d < uniref50.fasta.gz > uniref.fasta
- makeblastdb -in uniref.fasta -dbtype prot -out blastdb/uniref50_db
- psiblast -query seq.fasta -db uniref50_db -num_iterations 3 -out_ascii_pssm query.pssm -out output.txt


### --------------------------------------------------------------- 

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
    

### Explore:
- self-supervised GNNs: graph augmentations such as removing and predicting nodes, edges 
- predict graph descriptors such as node degree, edges, etc
- k-mean clustering in graphs for binding nodes vs non-binding nodes 
- hypergraph substructure and molecular fingerprints
- create edge graph for bipartite link prediction



### --------------------------------------------------------------- 


## NOTE:
“Results are reported as mean ± standard-deviation over 5 random seeds; each seed value is itself the average over a 5-fold CV on the training set. The held-out test set is never used for model selection.”



# Ablation Study Procedure

This document outlines the 4-step procedure for conducting ablation studies: dev → train (CV+HP) → test → aggregate. Below are the changes made to various files to support this procedure.

## File Changes

| File (relative to `code/`) | Action | Reason |
|-----------------------------|--------|--------|
| `conf/mode/test.yaml` (new) | ```yaml<br># ← NEW FILE<br>mode: test<br><br>hparams:        # reuse LR, batch, … from defaults<br>auto_resume: false  # don’t load previous ckpt<br><br># 80 / 20 hold-out split will be created on the fly<br># (see utils.py)<br>``` | Declares the test mode so Hydra can be called with `+mode=test`. |
| `conf/config.yaml` | ```yaml<br>defaults:<br>  - _self_<br>  - mode: train  # dev / test are picked via +mode=xxx<br>  …<br>``` | Lets you override the mode from CLI (`mode=test`). |
| `utils.py` | ```python<br>def train_test_split(data, seed, test_ratio=0.2):<br>    rng = np.random.default_rng(seed)<br>    idx = rng.permutation(len(data))<br>    split = int(len(data) * (1 - test_ratio))<br>    return data[idx[\:split]], data[idx[split:]]<br>``` | Adds a helper function for a single random split that depends on the seed. |
| `main.py` | ```python<br>if cfg.mode.mode == "test":<br>    train_data, test_data = train_test_split(data, cfg.seed)<br>else:<br>    train_data, test_data = data, None<br><br>if cfg.mode.mode == "test":<br>    kf_iter = [(range(len(train_data)), [])]  # single “fold”<br>else:<br>    kf_iter = kf.split(train_data)<br>for fold, (train_idx, val_idx) in enumerate(kf_iter):<br>    # (inside loop use train_data[...])<br><br>if cfg.mode.mode == "test":<br>    test_dl = create_dataloader(test_data, cfg)<br>    test_loss, test_met = validate_epoch(model, test_dl, device, met, cfg)<br>    print("\n=== Test ===")<br>    for k,v in test_met.items():<br>        print(f"{k}: {v:.4f}")<br>    # write CSV next step (callbacks)<br>``` | - Separates train data for CV vs. test data.<br>- Ensures no CV loop in test.<br>- Prints and later writes the test metrics. |
| `model/callbacks.py` | ```python<br>if self.config is not None and self.config.mode.mode == "test":<br>    # don’t save checkpoints during test runs<br>    return False<br>``` | Avoids polluting checkpoints during final testing. |
| `ablation.py` | ```python<br>cmd = [PYTHON, "main.py"] + [<br>        "mode=test",                # not train<br>        f"model.name={model}",<br>        f"model.decoder.type={decoder}",<br>        f"loss.contrastive.name={loss_name}",<br>        f"seed={seed}",<br>]<br><br># Parsing block still reads the “=== Test ===” section (leave as is).<br><br># Remove k-fold aggregation – instead aggregate across seeds only (already done).<br>``` | Ablation launcher now runs pure test runs for each variant–seed pair. |
| `paths in ablation.py` | Make sure `PROJECT_ROOT` points to `results/hgraphepi/m3epi/ablation/<model>/<decoder>/<loss>/summary` and use it when saving `raw_ablation_results.csv` and `aggregated_ablation_results.csv`. (You can build it exactly like in callbacks). | Keeps the same hierarchy as training checkpoints. |

## 4-Step Procedure

1. **Development (dev)**: Initial development and testing of the model.
2. **Training (train)**: Conduct cross-validation (CV) and hyperparameter tuning (HP).
3. **Testing (test)**: Evaluate the model on a hold-out test set.
4. **Aggregation**: Aggregate results across different seeds and variants.


This procedure ensures a systematic approach to model development, validation, and testing, providing reliable and reproducible results.
```bash
# 1) hyper-parameter search in train mode with CV
python sweep.py +mode=train …

# 2) once best HP picked, run ablation
python ablation.py

# 3) Testing on a single split (optional)
python main.py +mode=test  \
     model.name=GIN model.decoder.type=attention \
     loss.contrastive.name=gwnce \
     seed=17

```

ablation.py will now:

1. call main.py mode=test … for every (GNN, decoder, loss, seed)

2. parse the “=== Test ===” block, save raw CSVs and the across-seed means±std

3. no cross-validation takes place in these runs – they use the HP chosen previously.

### Table 1: Performance Metrics for Different Variants (ablation.py)

| GNN / Decoder / Loss       | MCC ↑       | AUROC ↑     | AUPRC ↑     |
|----------------------------|-------------|-------------|-------------|
| **GIN · attention · GW-NCE** | **0.312 ± 0.014** | 0.701 ± 0.009 | 0.428 ± 0.012 |
| GIN · attention · InfoNCE  | 0.297 ± 0.016 | …           | …           |
| …                          | …           | …           | …           |

**Notes:**
- Each row represents a different variant.
- The best MCC value in each GNN block is bolded.
- Metrics are reported as mean ± standard deviation.






######################################################



## 8 Single “best” checkpoint per run, nice file‑names


step	details
Why	Make results reproducible and easy to grep when you have dozens of runs.
Code touch	model/callbacks.py
• add arg run_name so the folder is …/checkpoints/<run_name>/
• compose filename:
fname = f"{cfg.model.name}_{cfg.loss.contrastive.name}_E{epoch:02d}_lr{cfg.hparams.train.learning_rate:.0e}_mcc{val_mcc:.3f}.pt"
Config	Add to callbacks.yaml
dirpath: ${results_dir}/checkpoints/${now:%Y%m%d_%H%M}_${mode}
W&B	Log an artifact with the same name – you can download it later in one click.
Deliverable	Table S1 in appendix that lists run‑ID ↔ checkpoint‑file.
## 9 Results CSV per run and k‑fold summary


step	details
Why	Automatable stats & pretty tables without re‑running Python.
Code touch	At the end of main.py after the print("\n=== Final ===") block: ```python
import pandas as pd, csv, pathlib	
row = {"run_id": wandb.run.id if cfg.logging_method=="wandb" else datetime.now().isoformat(),	
css
Copy
Edit
   "model": cfg.model.name,
   "loss": cfg.loss.contrastive.name,
   "lr": cfg.hparams.train.learning_rate,
   **{f"mean_{k}": v for k,v in avg.items()},
   **{f"std_{k}": v for k,v in std.items()}}
out_csv = pathlib.Path(cfg.results_dir) / "summary.csv" pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)``` | | Analysis | Python/pandas one‑liner produces LaTeX table from summary.csv. |

## 10 Ablation‑study scaffolding

### 10.1 Dimensions of the Ablation

Factor	Values
GNN architecture	GCN, GAT, GIN, GraphSAGE, (future: LSPE, GraphWalk)
Decoder	Dot‐product, MLP (1–2 hidden layers), Cross‐Attention
Loss	CE only, CE + InfoNCE, CE + GW‑NCE
Seeds	5 distinct random seeds
### 10.2 Experimental Matrix
Loop over every combination of (GNN, Decoder, Loss).

For each:

In train mode, run 5 × k‑fold (e.g. 5 seeds × 5 folds = 25 runs).

Collect and average:

Node metrics: AUROC, AUPRC, MCC

Edge metrics if desired

Training time

Save:

Per‐fold CSV (as above)

One aggregated CSV over seeds:
.../ablation/<model>/<decoder>/<loss>/summary/agg_across_seeds.csv

### 10.3 Figures & Tables
Bar charts of “mean ± std” for each metric across architectures / losses / decoders.

Line plots or heatmaps to show interaction (e.g. GNN vs Loss).

ROC / PR curves for the top 3 configurations.

10‑A  Shell helper
bash
Copy
Edit
# ablate.sh
models=(GCN GAT GIN)           # later add GraphSAGE …
losses=(ce infonce gwnce)
for m in "${models[@]}"; do
  for l in "${losses[@]}"; do
    python main.py model.name=$m loss.contrastive.name=$l \
        wandb.tags="[ablation]" wandb.name="${m}_${l}"
  done
done
10‑B  Hydra template
Add conf/mode/ablation.yaml:

yaml
Copy
Edit
mode: train
hparams:
  train:
    num_epochs: 10
    kfolds: 3
wandb:
  tags: ["ablation"]
Run: python main.py +mode=ablation …

10‑C  Paper artefacts
Table 2 – MCC/AUPRC per (model, loss) ± std.

Figure 3 – bar‑plot of ΔMCC versus CE‑only baseline.

## 11 Explicit test split


step	details
Data	Pre‑split dataset into train/val/test indices once, save to split.pkl so every model sees the same test set.
Config	conf/mode/test.yaml:
mode: test hparams.train.kfolds: 1
main.py	If cfg.mode=="test" skip CV and evaluate once on the frozen test set.
Return metrics JSON so CI can assert performance ≥ threshold.
Deliverable	Section “Generalisation” – report blind‑test MCC & 95 % CI.
## 12 Wall‑time profiling


step	details
Code touch	Wrap epoch loops with time.perf_counter(); accumulate train_sec and val_sec.
W&B logging	wandb.log({"epoch_train_time": train_t, "epoch_val_time": val_t})
Analysis	Scatter plot training‑time vs MCC over all runs to show efficiency trade‑off (Figure S4).
## 13 Contrastive‑loss as true config knob

Already done in loss.yaml & main.py.
Extra polish

Provide loss.contrastive.name: "none" path that skips InfoNCE entirely.

In visualize_embeddings.py set dot‑color shape depending on loss (nice figure for “latent‑space separation improves with InfoNCE”).

Good‑practice checklist (applies to every experiment)

item	rationale	tip
1 seed ≠ research	Show robustness	always run 3–5 seeds (Hydra sweep: seed=range(3))
Determinism	reviewers love it	torch.use_deterministic_algorithms(True) during test
Unit tests	stops silent NaNs	pytest -q tests/ before every sweep
Tag everything	future‑you will forget	wandb.tags: ["paper_v1", cfg.model.name, cfg.loss.contrastive.name]
Notebook‑less plots	CI‑friendly	save PDF/PNG via Matplotlib, script in /analysis/
Suggested execution order
Smoke‑test: mode=dev on laptop (< 1 min)

Baseline grid: run ablate.sh on a single GPU node overnight

Pick best trio (model, loss, hyper‑params) → run mode=train with 5 folds × 5 seeds

Final blind test with the best checkpoint

Embedding t‑SNE using that checkpoint

Write results: tables + figures straight from CSV / PNG folders

That workflow keeps experiments reproducible, your W&B dashboard tidy, and paper‑ready artefacts one command away. Good luck — ping me when you’re ready for the next punch‑list!


