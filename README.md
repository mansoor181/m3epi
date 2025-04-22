<!-- A Python module for paratope and epitope prediction
![image](https://github.com/WangZhiwei9/MIPE/blob/main/Overview.jpg) -->

# M3Epi: Multi-modal, multi-task, multi-network (GNN, LSTM, CL) Epitope Prediction

## Requirements

This project relies on specific Python packages to ensure its proper functioning. The required packages and their versions are listed in the `requirements.txt` file.

## Data

Dataset Files (pickle format) can be downloaded from: https://drive.google.com/drive/folders/1bvGZQnOs6XOA17NsiaZ4eVjvn94SOM3u?usp=drive_link

- Antibody embeddings generated using [AbLang](https://github.com/oxpig/AbLang.git)

## TODO: [mansoor]
- refactor train_model.py from walle and main.py from mipe
- perform train, val, test in this file with wandb logging
- perform k-fold cross validation
- define modes of working: train, dev, tunning, sweep

Ablation studies:
1. create and add different GNN models: GCN, GAT, GIN
    - make changes in the following files:
        - model.py
        - config yaml files: hparams, model
2. add contrastive learning InfoNCE loss, gradient-weighted NCE loss
    - update files: loss config, main.py
3. modes of experiments: dev, train, sweep (tuning)
    - update main.py: 
        - sliced data loading for dev
        - wandb sweep and config files for tuning
4. epitope prediction for antigen alone (antibody-agnostic)
    - exp settings: ag_alone, complex
    - tasks: epi_pred, bipartite_link
5. data configurations: 
    - PLM-based node embeddings: `plm`
    - MIPE-like node embeddings such as one-hot, aa_profile, etc: `vec`
    - files to update: preprocess.py
    NOTE:
    - how about use two different encoders?
        - one for processing `plm` and one for `vec`
        - then pass through inner product decoder and take average of the adjacency matrices
6. pre-trained sequence-based binding site prediction models
    - protein-ligand models for antigen: ESMBind
    - paratope prediction models for antibody: ParaAntiProt, Paragraph

Explore:
- self-supervised GNNs: graph augmentations such as removing and predicting nodes, edges 
- predict graph descriptors such as node degree, edges, etc
- k-mean clustering in graphs for binding nodes vs non-binding nodes 
- hypergraph substructure and molecular fingerprints

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
