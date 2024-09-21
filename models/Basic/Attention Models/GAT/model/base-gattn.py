import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class BaseGAttN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BaseGAttN, self).__init__()
        # Use GCNConv from PyTorch Geometric
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    @staticmethod
    def loss(logits, labels, class_weights):
        loss = F.nll_loss(logits, labels, weight=class_weights)
        return loss

    @staticmethod
    def training(loss, model, lr, l2_coef):
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss
    
    @staticmethod
    def masked_softmax_cross_entropy(logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        loss = F.nll_loss(logits, labels, reduction='none')
        mask = mask.float()
        mask /= mask.mean()
        loss *= mask
        return loss.mean()

    @staticmethod
    def masked_accuracy(logits, labels, mask):
        """Accuracy with masking."""
        preds = logits.argmax(dim=1)
        correct = (preds == labels).float()
        mask = mask.float()
        mask /= mask.mean()
        correct *= mask
        return correct.mean()

    @staticmethod
    def micro_f1(logits, labels, mask):
        """Micro F1 score with masking."""
        preds = torch.round(torch.sigmoid(logits))

        # Count true positives, false positives, etc.
        tp = ((preds * labels * mask).sum()).float()
        tn = (((1 - preds) * (1 - labels) * mask).sum()).float()
        fp = ((preds * (1 - labels) * mask).sum()).float()
        fn = (((1 - preds) * labels * mask).sum()).float()

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        fmeasure = 2 * precision * recall / (precision + recall + 1e-10)
        return fmeasure
