
"""
TODO: [mansoor]

Basic utilities:
    Random seed setting
    Device selection
    Data loading
    Model saving/loading
Logging utilities:
    WandB initialization
    Directory creation
    Logging setup
Training helpers:
    Class weight calculation
    Metrics formatting
Type hints and docstrings for better code readability and IDE support
"""



import os
import random
import numpy as np
import torch
import wandb
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
from omegaconf import OmegaConf

def train_test_split(data: Any, seed: int, test_ratio: float = 0.2) -> tuple:
    """
    Perform a single random train/test split.
    Returns (train_data, test_data).
    TODO:
    - use sklearn's train-test split function
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    perm = rng.permutation(n)
    cut = int(n * (1 - test_ratio))
    train_idx, test_idx = perm[:cut], perm[cut:]
    train_data = [data[i] for i in train_idx]
    test_data  = [data[i] for i in test_idx]
    return train_data, test_data


def seed_everything(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """Get the device to use for computations."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def load_data(data_path: str) -> Dict:
    """
    Load preprocessed data from pickle file.
    
    Args:
        data_path: Path to the pickle file containing the data
        
    Returns:
        Dictionary containing:
        - complex_code: PDB complex identifiers
        - coord_AG: Antigen coordinates
        - label_AG: Antigen labels (epitope/non-epitope)
        - coord_AB: Antibody coordinates
        - label_AB: Antibody labels (paratope/non-paratope) 
        - edge_AGAB: Edges between AG-AB nodes
        - edge_AB: Edges within antibody graph
        - edge_AG: Edges within antigen graph
        - vertex_AB: Antibody node features
        - vertex_AG: Antigen node features
        - AbLang_AB: Antibody language model embeddings
        - ESM1b_AG: Antigen language model embeddings
    """
    data = torch.load(data_path)
    return data


def initialize_wandb(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb_cfg = cfg_dict.get('wandb', {})

    # Pull tags out, ensure it's a list of str
    raw_tags = wandb_cfg.get('tags', None)
    tags = None
    if raw_tags:
        # force everything to str, skip non-iterables
        tags = tuple(str(t) for t in raw_tags if isinstance(t, (str, int, float)))
    
    wandb.init(
        project=wandb_cfg['project'],
        entity=wandb_cfg['entity'],
        name=wandb_cfg.get('name', None),
        group=wandb_cfg.get('group', None),
        # only pass tags if non-empty tuple
        **({'tags': tags} if tags else {}),
        notes=wandb_cfg.get('notes', None),
        config=cfg_dict,
        dir=wandb_cfg.get('save_dir', './wandb'),
        mode=wandb_cfg.get('mode', 'online'),
        resume=wandb_cfg.get('resume', 'allow'),
        anonymous=wandb_cfg.get('anonymous', 'allow'),
    )




def save_model(model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               epoch: int,
               loss: float,
               metrics: Dict[str, float],
               path: Union[str, Path]) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch number  
        loss: Current loss value
        metrics: Dictionary of metric values
        path: Path to save checkpoint
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }, path)

def load_model(model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               path: Union[str, Path]) -> tuple:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model to load weights into
        optimizer: Optimizer to load state into 
        path: Path to checkpoint file
        
    Returns:
        Tuple containing:
        - epoch number
        - model with loaded weights
        - optimizer with loaded state
        - loss value
        - metrics dictionary
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return (
        checkpoint['epoch'],
        model,
        optimizer,
        checkpoint['loss'],
        checkpoint['metrics']
    )

def get_run_dir(base_dir: Union[str, Path], run_name: str) -> Path:
    """Create and return directory for current run."""
    run_dir = Path(base_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def setup_logging(run_dir: Path) -> None:
    """Setup logging to file."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(run_dir / 'train.log'),
            logging.StreamHandler()
        ]
    )

def calculate_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        labels: Binary labels tensor
        
    Returns:
        Tensor of class weights
    """
    num_samples = len(labels)
    num_positives = labels.sum().item()
    num_negatives = num_samples - num_positives
    
    pos_weight = num_samples / (2 * num_positives)
    neg_weight = num_samples / (2 * num_negatives)
    
    weights = torch.zeros_like(labels, dtype=torch.float)
    weights[labels == 1] = pos_weight
    weights[labels == 0] = neg_weight
    
    return weights

def format_metrics(metrics: Dict[str, float], prefix: str = '') -> str:
    """Format metrics dictionary into string for printing."""
    return ' | '.join([
        f"{prefix}{k}: {v:.4f}" for k, v in metrics.items()
    ])


# import os
# import torch, random
# import numpy as np
# from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryMatthewsCorrCoef, BinaryAveragePrecision, BinaryAUROC, BinaryF1Score,  BinaryAccuracy

# # from model.loss import NTXentLoss  # Importing NTXentLoss from a custom module


# # Set up device configuration (use GPU if available, otherwise CPU)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def seed_everything(seed):
#     # Set Python random seed
#     random.seed(seed)

#     # Set NumPy random seed
#     np.random.seed(seed)

#     # Set PyTorch random seed
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# def generate_random_seed() -> int:
#     """ Generate a random seed using os.urandom
#     credit: https://stackoverflow.com/q/57416925
#     """
#     return int.from_bytes(
#         os.urandom(4),
#         byteorder="big"
#     )



def get_k_fold_data(K, i, X):
    """
    Split data into K folds for cross-validation and return train/val/test sets for fold i.
    
    Args:
        K (int): Number of folds
        i (int): Current fold index (0-based)
        X (array-like): Data to be split
        
    Returns:
        tuple: (X_train, X_val, X_test) - Training, validation and test sets for fold i
    """
    assert K > 1  # Ensure we have at least 2 folds
    fold_size = len(X) // K  # Calculate size of each fold

    X_train, X_val, X_test = None, None, None

    # Prepare fold indices
    tmp_list = list(range(K))
    idx_i = tmp_list.index(i)
    del tmp_list[idx_i]
    v = tmp_list[-1]  # Last remaining index will be validation set

    # Split data into folds
    for j in range(K):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part = X[idx]
        
        # Assign to test, val or train sets
        if j == i:
            X_test = X_part
        elif j == v:
            X_val = X_part
        elif X_train is None:
            X_train = X_part
        else:
            X_train = np.concatenate((X_train, X_part), axis=0)
    return X_train, X_val, X_test




# ----------------------------------------------------

# def evalution_prot(preds, targets):
#     """
#     Evaluate model predictions against targets using multiple metrics.
    
#     Args:
#         preds (torch.Tensor): Model predictions
#         targets (torch.Tensor): Ground truth labels
        
#     Returns:
#         tuple: (AUPRC, AUROC, precision, recall, f1, bacc, MCC)
#     """
#     # AUROC (Area Under Receiver Operating Characteristic curve)
#     auroc = BinaryAUROC().to(device)
#     auroc.update(preds, targets)
#     auroc_i = auroc.compute().item()
    
#     # AUPRC (Area Under Precision-Recall Curve)
#     auprc = BinaryAveragePrecision().to(device)
#     auprc.update(preds, targets)
#     auprc_i = auprc.compute().item()
    
#     # Precision
#     precision = BinaryPrecision().to(device)
#     precision_i = precision(preds, targets).item()
    
#     # Recall
#     recall = BinaryRecall().to(device)
#     recall_i = recall(preds, targets).item()
    
#     # F1 Score
#     f1 = BinaryF1Score().to(device)
#     f1_i = f1(preds, targets).item()
    
#     # Balanced Accuracy (using regular accuracy for binary classification)
#     bacc = BinaryAccuracy().to(device)
#     bacc_i = bacc(preds, targets).item()
    
#     # MCC (Matthews Correlation Coefficient)
#     mcc = BinaryMatthewsCorrCoef().to(device)
#     mcc_i = mcc(preds, targets).item()

#     return auprc_i, auroc_i, precision_i, recall_i, f1_i, bacc_i, mcc_i


# def consine_inter(A, B):
#     """
#     Compute cosine similarity between corresponding vectors in A and B.
    
#     Args:
#         A (torch.Tensor): First set of vectors
#         B (torch.Tensor): Second set of vectors
        
#     Returns:
#         torch.Tensor: Cosine similarity scores
#     """
#     dot_product = torch.sum(A * B, dim=1)
#     norm_A = torch.norm(A, dim=1)
#     norm_B = torch.norm(B, dim=1)
#     # Add small epsilon to avoid division by zero
#     cosine_similarity = dot_product / ((norm_A * norm_B) + 1e-8)
#     return cosine_similarity

# def dis_pairs(coord_1, coord_2):
#     """
#     Calculate Euclidean distance between two 3D coordinates.
    
#     Args:
#         coord_1 (list): First coordinate [x,y,z]
#         coord_2 (list): Second coordinate [x,y,z]
        
#     Returns:
#         float: Euclidean distance between the coordinates
#     """
#     # Extract coordinates
#     coord_1_x = coord_1[-3]
#     coord_1_y = coord_1[-2]
#     coord_1_z = coord_1[-1]
#     coord_2_x = coord_2[-3]
#     coord_2_y = coord_2[-2]
#     coord_2_z = coord_2[-1]
    
#     # Calculate Euclidean distance
#     distance = np.sqrt((float(coord_1_x) - float(coord_2_x)) ** 2 + 
#                (float(coord_1_y) - float(coord_2_y)) ** 2 + 
#                (float(coord_1_z) - float(coord_2_z)) ** 2)
#     return distance

# def index_mink(data, k):
#     """
#     Find indices of the k smallest values in a list.
    
#     Args:
#         data (list): Input data
#         k (int): Number of smallest values to find
        
#     Returns:
#         list: Indices of the k smallest values
#     """
#     Lst = data[:]  # Create a copy of the input list
#     index_k = []
#     for i in range(k):
#         index_i = Lst.index(min(Lst))  # Find index of current minimum
#         index_k.append(index_i)
#         Lst[index_i] = float('inf')  # Replace found minimum with infinity
#     return index_k

# def CreateGearnetGraph(data):
#     """
#     - Create Gearnet graph structures for AG and AB from input data.
#     - Create and combine 3 types of edges: sequential, radius, and kNN.
    
#     Args:
#         data (dict): Input data containing edge and coordinate information
        
#     Returns:
#         tuple: (ag_edge_ind, ab_edge_ind) - Graph structures for AG and AB
#     """
#     from torchdrug import data as drugdata
    
#     # Process AG (Antigen) graph
#     edge_AG_radius = (np.array(data["edge_AG"] + ([[1] * len(data["edge_AG"][0])])).T).tolist()
#     num_nodes_AG = max(max(np.array(edge_AG_radius)[:, 0]), max(np.array(edge_AG_radius)[:, 1])) + 1
    
#     # Create sequential edges
#     edge_AG_seq = []
#     for p in range(num_nodes_AG - 1):
#         edge_AG_seq.append([p, p + 1, 0])
    
#     # Create 10-nearest neighbor edges
#     edge_AG_10nearest = []
#     for p in range(num_nodes_AG):
#         dis_pq = []
#         for q in range(num_nodes_AG):
#             dis_pq.append(dis_pairs(data["coord_AG"][p], data["coord_AG"][q]))
#         near10_q = index_mink(dis_pq, 11)  # Get 11 nearest (including self)
#         del near10_q[near10_q.index(p)]  # Remove self
#         near10_AG_p = list(map(lambda x: [p, x, 2], near10_q))
#         edge_AG_10nearest = edge_AG_10nearest + near10_AG_p
    
#     # Combine all edge types
#     edge_AG = edge_AG_seq + edge_AG_radius + edge_AG_10nearest
#     graph_AG = drugdata.Graph(edge_AG, num_node=num_nodes_AG, num_relation=3).to(device)
#     node_embedding_AG = torch.tensor(data["vertex_AG"], dtype=torch.float).to(device)
#     ag_edge_ind = [graph_AG, node_embedding_AG]
    
#     # Process AB (Antibody) graph (similar to AG)
#     edge_AB_radius = (np.array(data["edge_AB"] + ([[1] * len(data["edge_AB"][0])])).T).tolist()
#     num_nodes_AB = max(max(np.array(edge_AB_radius)[:, 0]), max(np.array(edge_AB_radius)[:, 1])) + 1
    
#     edge_AB_seq = []
#     for p in range(num_nodes_AG - 1):  # Note: This uses num_nodes_AG which might be a bug
#         edge_AG_seq.append([p, p + 1, 0])
    
#     edge_AB_10nearest = []
#     for p in range(num_nodes_AB):
#         dis_pq = []
#         for q in range(num_nodes_AB):
#             dis_pq.append(dis_pairs(data["coord_AB"][p], data["coord_AB"][q]))
#         near10_q = index_mink(dis_pq, 11)
#         del near10_q[near10_q.index(p)]
#         near10_AB_p = list(map(lambda x: [p, x, 2], near10_q))
#         edge_AB_10nearest = edge_AB_10nearest + near10_AB_p
    
#     edge_AB = torch.tensor(edge_AB_seq + edge_AB_radius + edge_AB_10nearest)
#     graph_AB = drugdata.Graph(edge_AB, num_node=num_nodes_AB, num_relation=3).to(device)
#     node_embedding_AB = torch.tensor(data["vertex_AB"], dtype=torch.float).to(device)
#     ab_edge_ind = [graph_AB, node_embedding_AB]

#     return ag_edge_ind, ab_edge_ind

# def CreateKnearestEdge(data):
#     """
#     Create k-nearest neighbor edges for AG and AB graphs.
    
#     Args:
#         data (dict): Input data containing coordinate information
        
#     Returns:
#         tuple: (edge_AG_10nearest, edge_AB_10nearest) - Edge lists for AG and AB
#     """
#     # Process AG (Antigen) edges
#     num_nodes_AG = data["vertex_AB"].shape[0]  # Note: This uses vertex_AB which might be a bug
#     edge_AG_10nearest_p = []
#     edge_AG_10nearest_q = []
    
#     for p in range(num_nodes_AG):
#         dis_pq = []
#         for q in range(num_nodes_AG):
#             dis_pq.append(dis_pairs(data["coord_AG"][p], data["coord_AG"][q]))
#         near10_q = index_mink(dis_pq, 11)
#         del near10_q[near10_q.index(p)]
#         near10_p = [p] * 10
#         edge_AG_10nearest_p = edge_AG_10nearest_p + near10_p
#         edge_AG_10nearest_q = edge_AG_10nearest_q + near10_q
#     edge_AG_10nearest = [edge_AG_10nearest_p, edge_AG_10nearest_q]
    
#     # Process AB (Antibody) edges (similar to AG)
#     num_nodes_AB = data["vertex_AB"].shape[0]
#     edge_AB_10nearest_p = []
#     edge_AB_10nearest_q = []
#     for p in range(num_nodes_AB):
#         dis_pq = []
#         for q in range(num_nodes_AB):
#             dis_pq.append(dis_pairs(data["coord_AB"][p], data["coord_AB"][q]))
#         near10_q = index_mink(dis_pq, 11)
#         del near10_q[near10_q.index(p)]
#         near10_p = [p] * 10
#         edge_AB_10nearest_p = edge_AB_10nearest_p + near10_p
#         edge_AB_10nearest_q = edge_AB_10nearest_q + near10_q
#     edge_AB_10nearest = [edge_AB_10nearest_p, edge_AB_10nearest_q]
#     return edge_AG_10nearest, edge_AB_10nearest









# # Set up device configuration (use GPU if available, otherwise CPU)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def get_k_fold_data(K, i, X):
#     """
#     Split data into K folds for cross-validation and return train/val/test sets for fold i.
    
#     Args:
#         K (int): Number of folds
#         i (int): Current fold index (0-based)
#         X (array-like): Data to be split
        
#     Returns:
#         tuple: (X_train, X_val, X_test) - Training, validation and test sets for fold i
#     """
#     assert K > 1  # Ensure we have at least 2 folds
#     fold_size = len(X) // K  # Calculate size of each fold

#     X_train, X_val, X_test = None, None, None

#     # Prepare fold indices
#     tmp_list = list(range(K))
#     idx_i = tmp_list.index(i)
#     del tmp_list[idx_i]
#     v = tmp_list[-1]  # Last remaining index will be validation set

#     # Split data into folds
#     for j in range(K):
#         idx = slice(j * fold_size, (j + 1) * fold_size)
#         X_part = X[idx]
        
#         # Assign to test, val or train sets
#         if j == i:
#             X_test = X_part
#         elif j == v:
#             X_val = X_part
#         elif X_train is None:
#             X_train = X_part
#         else:
#             X_train = np.concatenate((X_train, X_part), axis=0)
#     return X_train, X_val, X_test


# def consine_inter(A, B):
#     """
#     Compute cosine similarity between corresponding vectors in A and B.
    
#     Args:
#         A (torch.Tensor): First set of vectors
#         B (torch.Tensor): Second set of vectors
        
#     Returns:
#         torch.Tensor: Cosine similarity scores
#     """
#     dot_product = torch.sum(A * B, dim=1)
#     norm_A = torch.norm(A, dim=1)
#     norm_B = torch.norm(B, dim=1)
#     # Add small epsilon to avoid division by zero
#     cosine_similarity = dot_product / ((norm_A * norm_B) + 1e-8)
#     return cosine_similarity

# def dis_pairs(coord_1, coord_2):
#     """
#     Calculate Euclidean distance between two 3D coordinates.
    
#     Args:
#         coord_1 (list): First coordinate [x,y,z]
#         coord_2 (list): Second coordinate [x,y,z]
        
#     Returns:
#         float: Euclidean distance between the coordinates
#     """
#     # Extract coordinates
#     coord_1_x = coord_1[-3]
#     coord_1_y = coord_1[-2]
#     coord_1_z = coord_1[-1]
#     coord_2_x = coord_2[-3]
#     coord_2_y = coord_2[-2]
#     coord_2_z = coord_2[-1]
    
#     # Calculate Euclidean distance
#     distance = np.sqrt((float(coord_1_x) - float(coord_2_x)) ** 2 + 
#                (float(coord_1_y) - float(coord_2_y)) ** 2 + 
#                (float(coord_1_z) - float(coord_2_z)) ** 2)
#     return distance

# def index_mink(data, k):
#     """
#     Find indices of the k smallest values in a list.
    
#     Args:
#         data (list): Input data
#         k (int): Number of smallest values to find
        
#     Returns:
#         list: Indices of the k smallest values
#     """
#     Lst = data[:]  # Create a copy of the input list
#     index_k = []
#     for i in range(k):
#         index_i = Lst.index(min(Lst))  # Find index of current minimum
#         index_k.append(index_i)
#         Lst[index_i] = float('inf')  # Replace found minimum with infinity
#     return index_k

# def CreateGearnetGraph(data):
#     """
#     - Create Gearnet graph structures for AG and AB from input data.
#     - Create and combine 3 types of edges: sequential, radius, and kNN.
    
#     Args:
#         data (dict): Input data containing edge and coordinate information
        
#     Returns:
#         tuple: (ag_edge_ind, ab_edge_ind) - Graph structures for AG and AB
#     """
#     from torchdrug import data as drugdata
    
#     # Process AG (Antigen) graph
#     edge_AG_radius = (np.array(data["edge_AG"] + ([[1] * len(data["edge_AG"][0])])).T).tolist()
#     num_nodes_AG = max(max(np.array(edge_AG_radius)[:, 0]), max(np.array(edge_AG_radius)[:, 1])) + 1
    
#     # Create sequential edges
#     edge_AG_seq = []
#     for p in range(num_nodes_AG - 1):
#         edge_AG_seq.append([p, p + 1, 0])
    
#     # Create 10-nearest neighbor edges
#     edge_AG_10nearest = []
#     for p in range(num_nodes_AG):
#         dis_pq = []
#         for q in range(num_nodes_AG):
#             dis_pq.append(dis_pairs(data["coord_AG"][p], data["coord_AG"][q]))
#         near10_q = index_mink(dis_pq, 11)  # Get 11 nearest (including self)
#         del near10_q[near10_q.index(p)]  # Remove self
#         near10_AG_p = list(map(lambda x: [p, x, 2], near10_q))
#         edge_AG_10nearest = edge_AG_10nearest + near10_AG_p
    
#     # Combine all edge types
#     edge_AG = edge_AG_seq + edge_AG_radius + edge_AG_10nearest
#     graph_AG = drugdata.Graph(edge_AG, num_node=num_nodes_AG, num_relation=3).to(device)
#     node_embedding_AG = torch.tensor(data["vertex_AG"], dtype=torch.float).to(device)
#     ag_edge_ind = [graph_AG, node_embedding_AG]
    
#     # Process AB (Antibody) graph (similar to AG)
#     edge_AB_radius = (np.array(data["edge_AB"] + ([[1] * len(data["edge_AB"][0])])).T).tolist()
#     num_nodes_AB = max(max(np.array(edge_AB_radius)[:, 0]), max(np.array(edge_AB_radius)[:, 1])) + 1
    
#     edge_AB_seq = []
#     for p in range(num_nodes_AG - 1):  # Note: This uses num_nodes_AG which might be a bug
#         edge_AG_seq.append([p, p + 1, 0])
    
#     edge_AB_10nearest = []
#     for p in range(num_nodes_AB):
#         dis_pq = []
#         for q in range(num_nodes_AB):
#             dis_pq.append(dis_pairs(data["coord_AB"][p], data["coord_AB"][q]))
#         near10_q = index_mink(dis_pq, 11)
#         del near10_q[near10_q.index(p)]
#         near10_AB_p = list(map(lambda x: [p, x, 2], near10_q))
#         edge_AB_10nearest = edge_AB_10nearest + near10_AB_p
    
#     edge_AB = torch.tensor(edge_AB_seq + edge_AB_radius + edge_AB_10nearest)
#     graph_AB = drugdata.Graph(edge_AB, num_node=num_nodes_AB, num_relation=3).to(device)
#     node_embedding_AB = torch.tensor(data["vertex_AB"], dtype=torch.float).to(device)
#     ab_edge_ind = [graph_AB, node_embedding_AB]

#     return ag_edge_ind, ab_edge_ind

# def CreateKnearestEdge(data):
#     """
#     Create k-nearest neighbor edges for AG and AB graphs.
    
#     Args:
#         data (dict): Input data containing coordinate information
        
#     Returns:
#         tuple: (edge_AG_10nearest, edge_AB_10nearest) - Edge lists for AG and AB
#     """
#     # Process AG (Antigen) edges
#     num_nodes_AG = data["vertex_AB"].shape[0]  # Note: This uses vertex_AB which might be a bug
#     edge_AG_10nearest_p = []
#     edge_AG_10nearest_q = []
    
#     for p in range(num_nodes_AG):
#         dis_pq = []
#         for q in range(num_nodes_AG):
#             dis_pq.append(dis_pairs(data["coord_AG"][p], data["coord_AG"][q]))
#         near10_q = index_mink(dis_pq, 11)
#         del near10_q[near10_q.index(p)]
#         near10_p = [p] * 10
#         edge_AG_10nearest_p = edge_AG_10nearest_p + near10_p
#         edge_AG_10nearest_q = edge_AG_10nearest_q + near10_q
#     edge_AG_10nearest = [edge_AG_10nearest_p, edge_AG_10nearest_q]
    
#     # Process AB (Antibody) edges (similar to AG)
#     num_nodes_AB = data["vertex_AB"].shape[0]
#     edge_AB_10nearest_p = []
#     edge_AB_10nearest_q = []
#     for p in range(num_nodes_AB):
#         dis_pq = []
#         for q in range(num_nodes_AB):
#             dis_pq.append(dis_pairs(data["coord_AB"][p], data["coord_AB"][q]))
#         near10_q = index_mink(dis_pq, 11)
#         del near10_q[near10_q.index(p)]
#         near10_p = [p] * 10
#         edge_AB_10nearest_p = edge_AB_10nearest_p + near10_p
#         edge_AB_10nearest_q = edge_AB_10nearest_q + near10_q
#     edge_AB_10nearest = [edge_AB_10nearest_p, edge_AB_10nearest_q]
#     return edge_AG_10nearest, edge_AB_10nearest













