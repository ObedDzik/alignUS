import torch
from medAI.metrics import calculate_binary_classification_metrics


class BinaryClassificationEvaluator:
    def __init__(self, log_images=False):
        self.all_logits = []
        self.all_labels = []
        self.log_images = log_images

    def update(self, logits, labels):
        self.all_logits.append(logits.detach().cpu())
        self.all_labels.append(labels.detach().cpu())

    def compute_metrics(self, prefix=""):
        if not self.all_logits: 
            return {} # never been updated, don't return metrics.

        all_logits = torch.cat(self.all_logits, dim=0)
        scores = all_logits.softmax(dim=1)[:, 1]
        all_labels = torch.cat(self.all_labels, dim=0)
        metrics = calculate_binary_classification_metrics(
            scores, all_labels, log_images=self.log_images
        )
        if prefix: 
            metrics = {f"{prefix}{key}": value for key, value in metrics.items()}
        return metrics

    def reset(self): 
        self.all_logits = []
        self.all_labels = []
