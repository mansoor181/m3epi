import torch
from torchmetrics import Metric
from torchmetrics.classification import (
    BinaryPrecision, 
    BinaryRecall,
    BinaryF1Score,
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryAUROC,
    BinaryMatthewsCorrCoef
)

class EpitopeMetrics:
    def __init__(self):
        self.metrics = {
            'precision': BinaryPrecision(),
            'recall': BinaryRecall(),
            'f1': BinaryF1Score(),
            'accuracy': BinaryAccuracy(),
            'average_precision': BinaryAveragePrecision(),
            'auroc': BinaryAUROC(),
            'mcc': BinaryMatthewsCorrCoef()
        }
        
    def update(self, preds, targets):
        for metric in self.metrics.values():
            metric.update(preds, targets)
            
    def compute(self):
        return {name: metric.compute() for name, metric in self.metrics.items()}
    
    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

class EdgePredictionMetrics(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct_edges", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_edges", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, preds, targets):
        self.correct_edges += torch.sum(preds == targets)
        self.total_edges += targets.numel()
        
    def compute(self):
        return self.correct_edges.float() / self.total_edges










# from typing import Dict
# import torch
# from torch import Tensor
# from torcheval.metrics import BinaryAUPRC, BinaryAUROC, BinaryConfusionMatrix
# from torcheval.metrics.functional import binary_auprc, binary_auroc
# from torchmetrics.functional import matthews_corrcoef
# from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryMatthewsCorrCoef, BinaryAveragePrecision, BinaryAUROC, BinaryF1Score,  BinaryAccuracy


# # Set up device configuration (use GPU if available, otherwise CPU)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ######### from MIPE ##############
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



# ######### from WALLE ##############

# def cal_edge_index_bg_auprc(
#     edge_index_bg_pred: Tensor,
#     edge_index_bg_true: Tensor,
#     edge_cutoff: float = 0.5,
# ) -> Tensor:
#     """
#     Compute AUPRC for bipartite link prediction.
#     """
#     with torch.no_grad():
#         t = edge_index_bg_true.reshape(-1).long().cpu()
#         p = (edge_index_bg_pred > edge_cutoff).reshape(-1).long().cpu()
#         return binary_auprc(p, t)


# def cal_edge_index_bg_auroc(
#     edge_index_bg_pred: Tensor,
#     edge_index_bg_true: Tensor,
# ) -> Tensor:
#     """
#     Compute AUC-ROC for bipartite link prediction.
#     """
#     with torch.no_grad():
#         t = edge_index_bg_true.reshape(-1).long().cpu()
#         p = edge_index_bg_pred.reshape(-1).cpu()
#         return binary_auroc(p, t)


# def cal_epitope_node_auprc(
#     edge_index_bg_pred: Tensor,
#     edge_index_bg_true: Tensor,
#     num_edge_cutoff: int,  # used to determine epitope residue from edges,
# ) -> Tensor:
#     """
#     Compute AUPRC for epitope node prediction.
#     """
#     with torch.no_grad():
#         # Get epitope idx
#         t = (edge_index_bg_true.sum(dim=0) > 0).reshape(-1).long()
#         p = (edge_index_bg_pred.sum(dim=0) > num_edge_cutoff).reshape(-1).long()
#         return binary_auprc(p, t)


# def cal_epitope_node_auroc(
#     edge_index_bg_pred: Tensor,
#     edge_index_bg_true: Tensor,
# ) -> Tensor:
#     """
#     Compute AUC-ROC for epitope node prediction.
#     """
#     with torch.no_grad():
#         # Get epitope idx
#         t = (edge_index_bg_true.sum(dim=0) > 0).reshape(-1).long().cpu()
#         p = edge_index_bg_pred.sum(dim=0).reshape(-1).cpu()
#         return binary_auroc(p, t)


# def cal_edge_index_bg_metrics(
#     edge_index_bg_pred: Tensor,
#     edge_index_bg_true: Tensor,
#     edge_cutoff: float = 0.5,
# ) -> Dict:
#     """
#     Compute metrics for bipartite link prediction:
#     - AUPRC, AUC-ROC, MCC, TP, TN, FP, FN.
#     """
#     with torch.no_grad():
#         t = edge_index_bg_true.reshape(-1).long().cpu()
#         p = edge_index_bg_pred.reshape(-1).cpu()

#         # AUPRC
#         auprc = BinaryAUPRC().update(input=p, target=t).compute()

#         # AUC-ROC
#         auroc = BinaryAUROC().update(input=p, target=t).compute()

#         # Confusion matrix
#         tn, fp, fn, tp = (
#             BinaryConfusionMatrix(threshold=edge_cutoff)
#             .update(input=p, target=t)
#             .compute()
#             .reshape(-1)
#         )

#         # MCC
#         mcc = (tp * tn - fp * fn) / (
#             torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-7
#         )

#         return {
#             "tn": tn,
#             "fp": fp,
#             "fn": fn,
#             "tp": tp,
#             "auprc": auprc,
#             "auroc": auroc,
#             "mcc": mcc,
#         }


# def cal_epitope_node_metrics(
#     edge_index_bg_pred: Tensor,
#     edge_index_bg_true: Tensor,
#     num_edge_cutoff: int,  # used to determine epitope residue from edges,
# ) -> Dict:
#     """
#     Compute metrics for epitope node prediction:
#     - AUPRC, AUC-ROC, MCC, TP, TN, FP, FN.
#     """
#     with torch.no_grad():
#         # Get epitope idx
#         t = (edge_index_bg_true.sum(dim=0) > 0).reshape(-1).long().cpu()
#         p = (edge_index_bg_pred.sum(dim=0) > num_edge_cutoff).reshape(-1).long().cpu()

#         # AUPRC
#         auprc = BinaryAUPRC().update(input=p, target=t).compute()

#         # AUC-ROC
#         auroc = BinaryAUROC().update(input=p, target=t).compute()

#         # Confusion matrix
#         tn, fp, fn, tp = (
#             BinaryConfusionMatrix().update(input=p, target=t).compute().reshape(-1)
#         )

#         # MCC
#         mcc = (tp * tn - fp * fn) / torch.sqrt(
#             (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-7
#         )

#         return {
#             "tn": tn,
#             "fp": fp,
#             "fn": fn,
#             "tp": tp,
#             "auprc": auprc,
#             "auroc": auroc,
#             "mcc": mcc,
#         }