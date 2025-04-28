import os
import logging
import sys
import subprocess
import yaml
import wandb
import argparse
import warnings

# ─────────────────────────────────────────────────────────────────────────────
# Silence the W&B malloc garbage and only log errors
# ─────────────────────────────────────────────────────────────────────────────
os.environ["WANDB_SILENT"] = "true"
logging.getLogger("wandb").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Parse command-line arguments for GPU selection
parser = argparse.ArgumentParser(description="Launch a single W&B hyperparameter sweep agent on a specified GPU")
parser.add_argument(
    "--gpu_id",
    type=int,
    default=0,
    help="CUDA GPU ID to pin for this sweep run (e.g. 0, 1, 2)",
)
parser.add_argument(
    "--count",
    type=int,
    default=5,
    help="Number of sweep runs to launch on the specified GPU",
)
args = parser.parse_args()

# Load the sweep configuration
def load_sweep_config():
    config_path = os.path.join(os.getcwd(), "conf", "sweep.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

sweep_config = load_sweep_config()
print("Launching sweep program:", sweep_config.get("program"))
print("Hyperparameters to sweep:", list(sweep_config.get("parameters", {}).keys()))

if __name__ == "__main__":
    # Ensure project root is on PYTHONPATH for Hydra
    os.environ.setdefault("PYTHONPATH", os.getcwd())

    # Pin the process to the selected GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(f"Pinned to CUDA_VISIBLE_DEVICES={args.gpu_id}")

    # Register the sweep
    sweep_id = wandb.sweep(
        sweep_config,
        project="m3epi",
        entity="alibilab-gsu",
    )
    """
    FIXME: 
    - set "WANDB_DIR" env variable in zshrc
        - sweep configs are now saved in wandb dir
    - sweep config parameters are not passed to wandb agent main.py (not overridden)
    - running multiple agents on the same GPU is not supported by wandb because they have same parent process
        - instead, run multiple sweep scripts on the specified GPUs 
    """


    # Launch a single in-process agent to run until the sweep is complete
    print(f"Starting W&B agent on GPU {args.gpu_id}, sweep ID: {sweep_id}")
    wandb.agent(sweep_id, count=args.count)




"""
Example usage:
    - define the sweep config in conf/sweep.yaml (for example, model-wise etc.)
    - run the sweep script with the desired GPU ID and count

python sweep.py --gpu_id 2 --count 5
"""




# import os, logging

# # ─────────────────────────────────────────────────────────────────────────────
# # Silence the W&B malloc garbage on macOS and only log errors
# # ─────────────────────────────────────────────────────────────────────────────
# os.environ["WANDB_SILENT"] = "true"
# logging.getLogger("wandb").setLevel(logging.ERROR)

# import sys
# import subprocess
# import yaml
# import wandb, torch

# import warnings
# warnings.filterwarnings("ignore")

# # load the sweep config
# with open(os.path.join(os.getcwd(), "conf", "sweep.yaml")) as f:
#     sweep_config = yaml.safe_load(f)

# print("Launching sweep:", sweep_config["program"])
# print("Hyperparameters:", list(sweep_config["parameters"].keys()))

# if __name__ == "__main__":
#     # Make sure code/ is importable, and hydra, omegaconf etc are on PYTHONPATH
#     os.environ.setdefault("PYTHONPATH", os.getcwd())

#     sweep_id = wandb.sweep(
#         sweep_config,
#         project="m3epi",
#         entity="alibilab-gsu",
#     )

#     # out_dir= os.path.join(os.getcwd(), "/../../../results/hgraphepi/m3epi/sweep")
    
#     # os.environ["WANDB_DIR"] = str(out_dir)

#     # how many GPUs are visible?

#     gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
#     print(sweep_config["parameters"]["hparams.train.multi_gpu"].values)

#     if gpu_count > 1 and sweep_config["parameters"].hparams.train.multi_gpu == True:
#         print(f"Detected {gpu_count} GPUs → launching one agent per card")
#         # Launch one agent run per GPU 
#         for gpu_id in range(gpu_count):
#             env = os.environ.copy()
#             env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
#             subprocess.Popen(
#                 [sys.executable, "-m", "wandb", "agent", sweep_id, "--count=2"],
#                 env=env,
#             )
#     else:
#         print("Single‐GPU or CPU only → running one in‐process agent")
#         # `count=1` will keep it pulling new runs until the sweep is done
#         wandb.agent(sweep_id, count=2)


