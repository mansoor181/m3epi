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
│       ├── train.yaml
│       └── test.yaml
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
├── ablation.py
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
    summary/  # aggregated for multiple seeds and raw results
      20250428T150547_a07d971_bs64_ne2_lr1e-3_threshold0.5_GCN-dot-infonce_agg.csv 
      20250428T150547_a07d971_bs64_ne2_lr1e-3_threshold0.5_GCN-dot-infonce_raw.csv
    logs/
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





