#!/usr/bin/env python3
import os, sys, time, itertools
import subprocess
import pandas as pd
from pathlib import Path

# 10.1  — factors to sweep
GNNs     = ["GCN", "GAT"] #, "GIN"]
Decoders = ["dot"] # , "attention"]
Losses   = ["infonce", "gwnce"]
Seeds    = [42, 43 ] #, 44, 45, 46]

"""
TODO: [mansoor]
- load the best sweep config and run ablation.py
"""

# sys.exit("Stopping the script.")
results_dir = os.getcwd() + "/../../../results/hgraphepi/m3epi/ablation"

# Paths
CODE_DIR     = Path(__file__).parent.resolve()
# PROJECT_ROOT = CODE_DIR.parent / "results" / "ablation"
# PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

# ensure our code is importable
env = os.environ.copy()
env["PYTHONPATH"] = str(CODE_DIR)
PYTHON = sys.executable

records = []
for model, decoder, loss_name, seed in itertools.product(GNNs, Decoders, Losses, Seeds):
    overrides = [
        "mode=test",                           # HOLD-OUT test
        f"model.name={model}",
        f"model.decoder.type={decoder}",
        f"seed={seed}",
        f"loss.contrastive.name={loss_name}"
    ]
    cmd = [PYTHON, "main.py"] + overrides
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
        print("   ❌ failed:", proc.stderr.strip().splitlines()[-1])
        continue

    # parse the “=== Test ===” block
    in_test = False
    metas = {}
    for line in proc.stdout.splitlines():
        if line.strip().startswith("===") and "Test" in line:
            in_test = True
            continue
        if in_test and ":" in line:
            key, val = line.split(":", 1)
            metas[key.strip()] = float(val.strip())

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

# Dump RAW
raw_csv = os.path.join(results_dir, "raw_ablation_results.csv")
pd.DataFrame(records).to_csv(raw_csv, index=False)
print("→ saved raw results to", raw_csv)

# ─── Aggregate across seeds ────────────────────────────────────────
df = pd.DataFrame(records)
if not df.empty:
    group_cols = ["model","decoder","loss"]

    # 1) compute mean & std
    agg = (
        df
        .groupby(group_cols)
        .agg({
            "mcc":          ["mean","std"],
            "auroc":        ["mean","std"],
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
    metrics = ["mcc","auroc","precision","recall","f1","duration_s"]
    for m in metrics:
        mean_c = f"{m}_mean"
        std_c  = f"{m}_std"
        agg[m] = agg.apply(
            lambda row: f"{row[mean_c]:.4f}±{row[std_c]:.5f}",
            axis=1
        )
        # drop the now‐redundant columns
        agg.drop([mean_c, std_c], axis=1, inplace=True)

    # 4) reorder to keep group_cols first
    agg = agg[group_cols + metrics]

    # 5) write out
    agg_csv = os.path.join(results_dir, "aggregated_ablation_results.csv")
    agg.to_csv(agg_csv, index=False)
    print("→ saved aggregated results to", agg_csv)
else:
    print("⚠️ no successful runs to aggregate.")






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
