import os
import sys
import subprocess
import yaml
import wandb

# load the sweep config
with open(os.path.join(os.getcwd(), "conf", "sweep.yaml")) as f:
    sweep_config = yaml.safe_load(f)

print("Launching sweep:", sweep_config["program"])
print("Hyperparameters:", list(sweep_config["parameters"].keys()))

if __name__ == "__main__":
    # Make sure code/ is importable, and hydra, omegaconf etc are on PYTHONPATH
    os.environ.setdefault("PYTHONPATH", os.getcwd())

    sweep_id = wandb.sweep(
        sweep_config,
        project="m3epi",
        entity="alibilab-gsu",
    )

    # Launch a handful of parallel agents
    wandb.agent(
        sweep_id,
        function=lambda: subprocess.run(
            [
                # Use the exact same interpreter that launched this script:
                sys.executable,
                "main.py",
                "num_threads=1"
            ],
            env=os.environ,
            check=True
        ),
        count=2,
    )
