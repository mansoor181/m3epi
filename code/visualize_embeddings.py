import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PygDataLoader

from model.model import M3EPI
from utils import load_data, seed_everything, get_device

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.append( os.path.abspath(os.path.join(os.getcwd(),  '../code')))

torch.set_float32_matmul_precision("high")

class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_b":
            return self.x_b.size(0)
        if key == "edge_index_g":
            return self.x_g.size(0)
        if key == "edge_index_ag_ab":
            # ensure the right increment for batching
            return torch.tensor([[self.x_b.size(0)], [self.x_g.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)


def create_dataloader(dataset, cfg):
    data_list = []
    for item in dataset:
        pair = PairData(
            x_b=torch.tensor(item["vertex_AB"], dtype=torch.float),
            edge_index_b=torch.tensor(item["edge_AB"], dtype=torch.long),
            y_b=torch.tensor(item["label_AB"], dtype=torch.float),
            x_g=torch.tensor(item["vertex_AG"], dtype=torch.float),
            edge_index_g=torch.tensor(item["edge_AG"], dtype=torch.long),
            y_g=torch.tensor(item["label_AG"], dtype=torch.float),
            edge_index_ag_ab=torch.tensor(item["edge_AGAB"], dtype=torch.long),
        )
        data_list.append(pair)
    return PygDataLoader(
        data_list,
        batch_size=cfg.hparams.train.batch_size,
        shuffle=True,
        follow_batch=["x_g", "x_b"]
    )

def extract_embeddings(model, loader, device):
    model.eval()
    ag_embs, ab_embs = [], []
    ag_labels, ab_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            ag_embs.append(out['ag_embed'].cpu())
            ab_embs.append(out['ab_embed'].cpu())
            ag_labels.append(batch.y_g.cpu())
            ab_labels.append(batch.y_b.cpu())
    return (
        torch.cat(ag_embs).numpy(),
        torch.cat(ag_labels).numpy(),
        torch.cat(ab_embs).numpy(),
        torch.cat(ab_labels).numpy()
    )

def plot_tsne(emb, labels, title, out_path, label_mapping):
    tsne = TSNE(n_components=2, random_state=42)
    emb2d = tsne.fit_transform(emb)

    # Create a colormap
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('viridis', len(unique_labels))

    plt.figure(figsize=(6, 6))
    for i, label in enumerate(unique_labels):
        idx = labels == label
        plt.scatter(emb2d[idx, 0], emb2d[idx, 1], c=[colors(i)], alpha=0.7, marker="*", label=label_mapping[label])

    plt.title(title)
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"→ saved {out_path}")
    # plt.show()
    plt.close()

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    device = get_device()

    # load the data‐list containing Ab-Ag graph pairs
    data = load_data(os.path.join(cfg.data_dir, "asep_mipe_transformed_100_examples.pkl"))

    # one big batch so we get every node's embedding
    loader = create_dataloader(data[:20], cfg=cfg)

    # load your checkpoint (must have saved the config too!)
    ckpt = torch.load(cfg.visualize.checkpoint, map_location="cpu")

    # rehydrate the saved plain‐dict into a DictConfig
    model_cfg = OmegaConf.create(ckpt["config"])

    """
    NOTE:
    - model M3EPI takes (Ag-Ab) pair producing processed embeddings (Ag'-Ab')
    """
    model = M3EPI(model_cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])

    # extract
    ag_emb, ag_lbl, ab_emb, ab_lbl = extract_embeddings(model, loader, device)

    # ensure out_dir exists
    os.makedirs(cfg.visualize.out_dir, exist_ok=True)
    label_mapping = {0: "non-binding", 1: "binding"}

    # plot antigens
    plot_tsne(
        ag_emb, ag_lbl,
        "Antigen: epitope vs non‑epitope",
        os.path.join(cfg.visualize.out_dir, "tsne_antigen.png"),
        label_mapping
    )

    # plot antibodies
    plot_tsne(
        ab_emb, ab_lbl,
        "Antibody: paratope vs non‑paratope",
        os.path.join(cfg.visualize.out_dir, "tsne_antibody.png"),
        label_mapping
    )

if __name__ == "__main__":
    main()
