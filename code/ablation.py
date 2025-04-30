#!/usr/bin/env python3
import os, sys, time, itertools, argparse
import subprocess, torch
import pandas as pd
from pathlib import Path

from datetime import datetime
import subprocess as sp

import warnings

warnings.filterwarnings("ignore")

# ──────── USER SWEEP SETUP ────────────────────────────────────────
GNNs     = ["GCN", "GAT", "GIN"]
Decoders = ["dot", "attention", "mlp"]
Losses   = ["infonce", "gwnce"]
Seeds    = [42, 43, 44, 45, 46]


# test ablations
GNNs     = ["GIN"] # "GAT"] #, "GIN"]
Decoders = ["dot"] # , "attention"]
Losses   = ["infonce"] #, "gwnce"]
Seeds    = [42, 43 ] #, 44, 45, 46]

# where to dump results
RESULTS_DIR = os.getcwd() + "/../../../results/hgraphepi/m3epi/ablation"
summary_dir = os.path.join(RESULTS_DIR, "summary")
logs_dir   = os.path.join(RESULTS_DIR, "logs")

# Make sure results dir exists
Path(summary_dir).mkdir(parents=True, exist_ok=True)
Path(logs_dir).mkdir(parents=True, exist_ok=True)

# ensure our code is importable
CODE_DIR = Path(__file__).parent.resolve()
env = os.environ.copy()
env["PYTHONPATH"] = str(CODE_DIR)
PYTHON = sys.executable

# ──────── PARSER ──────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Run ablation experiments, optionally parallelized across GPUs"
)
parser.add_argument(
    "--multi_gpu",
    action="store_true",
    help="Run experiments in parallel across all available GPUs",
)
parser.add_argument(
    "--gpu_id",
    type=int,
    default=0,
    help="CUDA GPU ID to pin for sequential mode",
)
# capture all remaining args as Hydra overrides
parser.add_argument(
    'overrides', nargs=argparse.REMAINDER,
    help="Hydra overrides, e.g. key=val"
)
args = parser.parse_args()

# ──────── CREATE RESULTS FILENAMES ─────────────────────────────────────────

# build overrides_dict
overrides_dict = {}
for o in args.overrides:
    if "=" in o:
        k, v = o.split("=", 1)
        overrides_dict[k] = v

# abbreviate only the keys 
key_map = {
    "hparams.train.batch_size":    "bs",
    "hparams.train.learning_rate": "lr",
    "hparams.train.num_epochs":    "ne",
    "hparams.train.kfolds":        "kf",
}

parts = []
for k, v in overrides_dict.items():
    if k in ("wandb.notes", "wandb.tags"):
        continue
    short = key_map.get(k, k.split(".")[-1])
    parts.append(f"{short}{v}")

param_str = "_".join(parts) if parts else "default"

# timestamp / git‐SHA / param / model strings for file names
ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
try:
    sha = sp.check_output(["git","rev-parse","--short","HEAD"]).decode().strip()
except Exception:
    sha = "nogit"


# join all models/decoders/losses (or if you only have one each, it'll be that combo)
model_str = (
    f"{'+'.join(GNNs)}-"
    f"{'+'.join(Decoders)}-"
    f"{'+'.join(Losses)}"
)

# final paths
raw_path = os.path.join(summary_dir, f"{ts}_{sha}_{param_str}_{model_str}_raw.csv")
agg_path = os.path.join(summary_dir, f"{ts}_{sha}_{param_str}_{model_str}_agg.csv")

# parse Hydra overrides
USER_OVERRIDES = args.overrides

# ──────── BUILD ALL TASKS ─────────────────────────────────────────
all_tasks = []
for model, decoder, loss_name, seed in itertools.product(GNNs, Decoders, Losses, Seeds):
    base_overrides = [
        "mode=test",                           # HOLD-OUT test
        f"model.name={model}",
        f"model.decoder.type={decoder}",
        f"seed={seed}",
        f"loss.contrastive.name={loss_name}"
    ]
    cmd = [PYTHON, "main.py"] + USER_OVERRIDES + base_overrides
    log_file = os.path.join(logs_dir, f"log_{model}_{decoder}_{loss_name}_{seed}.txt")
    all_tasks.append((cmd, log_file))

print("Total ablation experiments:", len(all_tasks))

# ──────── EXECUTION ────────────────────────────────────────────────
records = []
gpu_count = torch.cuda.device_count() if (args.multi_gpu and torch.cuda.is_available()) else 0

# pin GPU for sequential mode
if not args.multi_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(f"Sequential mode: pinned to GPU {args.gpu_id}")


def parse_metrics_from_lines(lines):
    in_test = False
    metas = {}
    for line in lines:
        if line.strip().startswith("===") and "Test" in line:
            in_test = True
            continue
        if in_test and ":" in line:
            k, v = line.split(":", 1)
            metas[k.strip()] = float(v.strip())
    return metas

if args.multi_gpu and gpu_count > 0:
    print(f"--multi_gpu: running up to {gpu_count} jobs in parallel")
    tasks = []
    for idx, (cmd, log_file) in enumerate(all_tasks):
        gpu_id = idx % gpu_count
        e = env.copy()
        e["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        tasks.append((cmd, log_file, e))

    active = []
    while tasks or active:
        while len(active) < gpu_count and tasks:
            cmd, log_file, e = tasks.pop(0)
            print("▶︎ Launch:", " ".join(cmd))
            f = open(log_file, "w")
            p = subprocess.Popen(
                cmd,
                cwd=str(CODE_DIR),
                env=e,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active.append((p, log_file, f, cmd))
        time.sleep(1)
        for (p, log_file, f, cmd) in list(active):
            if p.poll() is not None:
                f.close()
                active.remove((p, log_file, f, cmd))
                lines = log_file.read_text().splitlines()
                metas = parse_metrics_from_lines(lines)
                if metas:
                    rec = {
                        "model": model,
                        "decoder": decoder,
                        "loss": loss_name,
                        "seed": seed,
                        "duration_s": None,
                        **metas
                    }
                    records.append(rec)
                    # flush raw results so far
                    pd.DataFrame(records).to_csv(raw_path, index=False)

else:
    print("Sequential execution of all tasks")
    for cmd, _ in all_tasks:
        print("▶︎ Running:", " ".join(cmd))
        start = time.time()
        proc = subprocess.run(
            cmd,
            cwd=str(CODE_DIR),
            env=env,
            capture_output=True,
            text=True
        )
        duration = time.time() - start
        if proc.returncode != 0:
            print("   ❌ failed:", proc.stderr.splitlines()[-1])
            continue
        metas = parse_metrics_from_lines(proc.stdout.splitlines())
        if metas:
            rec = {
                    "model": model,
                    "decoder": decoder,
                    "loss": loss_name,
                    "seed": seed,
                    "duration_s": duration,
                    **metas
                }
            records.append(rec)
            # print(records)
            
            # flush raw results so far
            pd.DataFrame(records).to_csv(raw_path, index=False)



# ─── Aggregate across seeds ────────────────────────────────────────
df = pd.DataFrame(records)
if not df.empty:
    group_cols = ["model","decoder","loss"]
    # group_cols = ["model.name","model.decoder.type","loss.contrastive.name"]

    # 1) compute mean & std
    agg = (
        df
        .groupby(group_cols)
        .agg({
            "mcc":          ["mean","std"],
            "auroc":        ["mean","std"],
            "auprc":        ["mean","std"],
            "precision":    ["mean","std"],
            "recall":       ["mean","std"],
            "f1":           ["mean","std"],
            "duration_s":   ["mean","std"],
        })
    )

    # 2) flatten multi‐index
    agg.columns = ["_".join(col) for col in agg.columns]
    agg = agg.reset_index()

    # 3) for each metric, combine mean & std into one column
    metrics = ["mcc","auroc","auprc","precision","recall","f1","duration_s"]
    for m in metrics:
        mean_c = f"{m}_mean"
        std_c  = f"{m}_std"
        agg[m] = agg.apply(
            lambda row: f"{row[mean_c]:.3f}±{row[std_c]:.4f}",
            axis=1
        )
        # drop the now‐redundant columns
        agg.drop([mean_c, std_c], axis=1, inplace=True)

    # 4) reorder to keep group_cols first
    agg = agg[group_cols + metrics]
    
    agg.to_csv(agg_path, index=False)
    print("→ saved aggregated results to", agg_path)

else:
    print("⚠️ no successful runs to aggregate.")



"""
Example usage: 
python ablation.py --multi_gpu \

python ablation.py --gpu_id 0 \
  hparams.train.batch_size=64 \
  wandb.notes="ablation" \
  wandb.tags=["ablation"] \
  hparams.train.num_epochs=2 \
  hparams.train.learning_rate=1e-3 \
  model.decoder.threshold=0.5 \
  num_threads=1
"""

"""
python ablation.py --gpu_id 0 \
  hparams.train.batch_size=64 \
  wandb.notes="ablation" \
  wandb.tags=["ablation"] \
  hparams.train.num_epochs=2 \
  hparams.train.learning_rate=1e-3 \
  model.decoder.threshold=0.5 \
  num_threads=1

FIXME: can't add multiple wandb tags
"""

"""
python ablation.py --gpu_id 0 \
    wandb.notes="ablation"   \
    wandb.tags=["ablation"]   \
    /home/mansoor/epitope_prediction/results/hgraphepi/m3epi/wandb/sweep-ir6r1ivi/config-n864pdx2.yaml
"""

"""
python ablation.py --gpu_id 0   hparams.train.batch_size=128   wandb.notes="GIN ablation"   wandb.tags=["ablation", "GIN"]   hparams.train.num_epochs=350   hparams.train.learning_rate=5.467765138902218e-05   model.decoder.threshold=0.511343179376244 loss.gwnce.cut_rate=1.0798225460589037 loss.gwnce.cut_way=2.486187013261815 model.dropout=0.1
"""


# #!/usr/bin/env python3
# import os, sys, time, itertools
# import subprocess
# import pandas as pd
# from pathlib import Path
# import warnings
# warnings.filterwarnings("ignore")

# # # 10.1  — factors to sweep
# # GNNs     = ["GCN", "GAT"] #, "GIN"]
# # Decoders = ["dot"] # , "attention"]
# # Losses   = ["infonce"] #, "gwnce"]
# # Seeds    = [42, 43 ] #, 44, 45, 46]

# GNNs     = ["GCN", "GAT", "GIN"]
# Decoders = ["dot", "attention", "mlp"]
# Losses   = ["infonce", "gwnce"]
# Seeds    = [42, 43, 44, 45, 46]

# """
# TODO: [mansoor]
# - load the best sweep config and run ablation.py
# """

# # grab all the user‐provided hydra overrides
# # e.g. ["hparams.train.batch_size=64", "model.name=GIN", …]
# USER_OVERRIDES = [arg for arg in sys.argv[1:] if "=" in arg]

# # sys.exit("Stopping the script.")
# results_dir = os.getcwd() + "/../../../results/hgraphepi/m3epi/ablation"

# # Paths
# CODE_DIR     = Path(__file__).parent.resolve()
# # PROJECT_ROOT = CODE_DIR.parent / "results" / "ablation"
# # PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

# # ensure our code is importable
# env = os.environ.copy()
# env["PYTHONPATH"] = str(CODE_DIR)
# PYTHON = sys.executable

# records = []
# # total ablation experiments = len(GNNs) * len(Decoders) * len(Losses) * len(Seeds)
# print("Total ablation experiments:", len(GNNs) * len(Decoders) * len(Losses) * len(Seeds))
# for model, decoder, loss_name, seed in itertools.product(GNNs, Decoders, Losses, Seeds):
#     overrides = [
#         "mode=test",                           # HOLD-OUT test
#         f"model.name={model}",
#         f"model.decoder.type={decoder}",
#         f"seed={seed}",
#         f"loss.contrastive.name={loss_name}"
#     ]
#     # cmd = [PYTHON, "main.py"] + overrides
#     cmd = [PYTHON, "main.py"] + USER_OVERRIDES + overrides

#     print("▶︎ Running:", " ".join(cmd))
#     start = time.time()
#     proc = subprocess.run(
#         cmd,
#         cwd=str(CODE_DIR),
#         env=env,
#         capture_output=True,
#         text=True
#     )
#     duration = time.time() - start

#     if proc.returncode != 0:
#         print("   ❌ failed:", proc.stderr.strip().splitlines()[-1])
#         continue

#     # parse the “=== Test ===” block
#     in_test = False
#     metas = {}
#     for line in proc.stdout.splitlines():
#         if line.strip().startswith("===") and "Test" in line:
#             in_test = True
#             continue
#         if in_test and ":" in line:
#             key, val = line.split(":", 1)
#             metas[key.strip()] = float(val.strip())

#     if metas:
#         rec = {
#             "model": model,
#             "decoder": decoder,
#             "loss": loss_name,
#             "seed": seed,
#             "duration_s": duration,
#             **metas
#         }
#         records.append(rec)

# # Dump RAW
# raw_csv = os.path.join(results_dir, "raw_ablation_results.csv")
# pd.DataFrame(records).to_csv(raw_csv, index=False)
# print("→ saved raw results to", raw_csv)

# # ─── Aggregate across seeds ────────────────────────────────────────
# df = pd.DataFrame(records)
# if not df.empty:
#     group_cols = ["model","decoder","loss"]

#     # 1) compute mean & std
#     agg = (
#         df
#         .groupby(group_cols)
#         .agg({
#             "mcc":          ["mean","std"],
#             "auroc":        ["mean","std"],
#             "auprc":        ["mean","std"],
#             "precision":    ["mean","std"],
#             "recall":       ["mean","std"],
#             "f1":           ["mean","std"],
#             "duration_s":   ["mean","std"],
#         })
#     )

#     # 2) flatten multi‐index
#     agg.columns = ["_".join(col) for col in agg.columns]
#     agg = agg.reset_index()

#     # 3) for each metric, combine mean & std into one column
#     metrics = ["mcc","auroc","auprc","precision","recall","f1","duration_s"]
#     for m in metrics:
#         mean_c = f"{m}_mean"
#         std_c  = f"{m}_std"
#         agg[m] = agg.apply(
#             lambda row: f"{row[mean_c]:.4f}±{row[std_c]:.5f}",
#             axis=1
#         )
#         # drop the now‐redundant columns
#         agg.drop([mean_c, std_c], axis=1, inplace=True)

#     # 4) reorder to keep group_cols first
#     agg = agg[group_cols + metrics]

#     # 5) write out
#     agg_csv = os.path.join(results_dir, "aggregated_ablation_results.csv")
#     agg.to_csv(agg_csv, index=False)
#     print("→ saved aggregated results to", agg_csv)
# else:
#     print("⚠️ no successful runs to aggregate.")



"""
Example usage:
python ablation.py \
  hparams.train.batch_size=64 \
  hparams.train.num_epochs=15 \
  hparams.train.kfolds=5 \
  hparams.train.learning_rate=1e-3 \
  model.decoder.threshold=0.5 \
  num_threads=1
"""



# import os
# import subprocess
# import time
# import itertools
# import pandas as pd
# from pathlib import Path
# import sys

# # 10.1  —  factors
# # GNNs     = ["GCN", "GAT", "GIN" ] #, "GraphSAGE"]
# GNNs     = ["GCN", "GAT"] #, "GraphSAGE"]
# # Decoders = ["dot", "mlp", "cross"]
# Decoders = ["dot", "attention"]
# # Losses   = ["", "infonce", "gwnce"]
# Losses   = ["infonce", "gwnce"]
# # Seeds    = [42, 43, 44, 45, 46]
# Seeds    = [42, 43]

# CODE_DIR     = Path(__file__).parent.resolve()     # <your_project>/code

# # assume you run this from code/ so main.py is here
# MAIN = "python main.py"
# PROJECT_ROOT = os.path.join(os.getcwd(), "/../../../results/hgraphepi/m3epi/ablation")
# # print(PROJECT_ROOT)

# # Paths
# # PROJECT_ROOT = CODE_DIR.parent / "results" / "ablation"

# # Make sure results dir exists
# PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

# # Base env for subprocesses: add code/ to PYTHONPATH
# env = os.environ.copy()
# env["PYTHONPATH"] = str(CODE_DIR)
# PYTHON = sys.executable                  # invoke same Python

# # os.environ.setdefault("PYTHONPATH", os.getcwd())

# records = []

# for model, decoder, loss_name, seed in itertools.product(GNNs, Decoders, Losses, Seeds):
#     overrides = [
#         "mode=train",
#         f"model.name={model}",
#         f"model.decoder.type={decoder}",
#         f"seed={seed}"
#     ]
#     # decide contrastive
#     if loss_name == "ce":
#         # skip any contrastive term
#         overrides.append("loss.contrastive.name=none")
#     else:
#         overrides.append(f"loss.contrastive.name={loss_name}")

#     cmd = ["python3.10", "main.py"] + overrides
#     print("▶︎ Running:", " ".join(cmd))
#     start = time.time()
#     proc = subprocess.run(
#         cmd,
#         cwd=str(CODE_DIR),
#         env=env,
#         capture_output=True,
#         text=True
#     )
#     duration = time.time() - start

#     if proc.returncode != 0:
#         print("   ❌ failed:", proc.stderr.splitlines()[-1])
#         continue

#     # parse “=== Final ===” block
#     final = False
#     metrics = {}
#     for line in proc.stdout.splitlines():
#         if line.strip().startswith("==="):
#             final = True
#             continue
#         if final and ":" in line:
#             key, val = line.split(":", 1)
#             mean, std = val.strip().split("±")
#             metrics[f"{key.strip()}_mean"] = float(mean)
#             metrics[f"{key.strip()}_std"]  = float(std)

#     # only record if we actually parsed something
#     if metrics:
#         rec = {
#             "model": model,
#             "decoder": decoder,
#             "loss": loss_name,
#             "seed": seed,
#             "duration_s": duration
#         }
#         rec.update(metrics)
#         records.append(rec)

# # ─── Dump raw CSV ──────────────────────────────────────────────────
# raw_csv = PROJECT_ROOT / "raw_ablation_results.csv"
# pd.DataFrame(records).to_csv(raw_csv, index=False)
# print("→ saved raw results to", raw_csv)

# # ─── Aggregate across seeds ────────────────────────────────────────
# df = pd.DataFrame(records)
# if not df.empty:
#     group_cols = ["model", "decoder", "loss"]
#     agg = df.groupby(group_cols).agg({
#         "mcc_mean":                 ["mean","std"],
#         "auroc_mean":               ["mean","std"],
#         "average_precision_mean":   ["mean","std"],
#         "duration_s":               ["mean","std"]
#     }).reset_index()
#     # flatten column index
#     agg.columns = [
#         "_".join(filter(None, col)).strip("_")
#         for col in agg.columns.values
#     ]
#     agg_csv = PROJECT_ROOT / "aggregated_ablation_results.csv"
#     agg.to_csv(agg_csv, index=False)
#     print("→ saved aggregated results to", agg_csv)
# else:
#     print("⚠️  No successful runs to aggregate.")
