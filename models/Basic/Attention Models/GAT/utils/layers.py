import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class AttentionHead(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0, residual=False):
        super(AttentionHead, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.f1 = torch.nn.Conv1d(out_channels, 1, kernel_size=1)
        self.f2 = torch.nn.Conv1d(out_channels, 1, kernel_size=1)
        self.dropout = dropout
        self.residual = residual

    def forward(self, x, edge_index, bias_mat):
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply conv1 to generate features
        seq_fts = self.conv1(x)

        # Self-attention logits
        f_1 = self.f1(seq_fts)
        f_2 = self.f2(seq_fts)
        logits = f_1 + f_2.permute(0, 2, 1)  # Broadcasting sum

        # Softmax attention coefficients
        coefs = F.softmax(F.leaky_relu(logits) + bias_mat, dim=-1)

        if self.dropout > 0:
            coefs = F.dropout(coefs, p=self.dropout, training=self.training)
            seq_fts = F.dropout(seq_fts, p=self.dropout, training=self.training)

        # Apply attention to features
        vals = torch.matmul(coefs, seq_fts)

        # Residual connection
        if self.residual:
            if x.shape[-1] != vals.shape[-1]:
                vals += self.conv1(x)  # Residual adjustment
            else:
                vals += x

        return F.elu(vals)


class SparseAttentionHead(torch.nn.Module):
    def __init__(self, in_channels, out_channels, nb_nodes, dropout=0.0, residual=False):
        super(SparseAttentionHead, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.f1 = torch.nn.Conv1d(out_channels, 1, kernel_size=1)
        self.f2 = torch.nn.Conv1d(out_channels, 1, kernel_size=1)
        self.nb_nodes = nb_nodes
        self.dropout = dropout
        self.residual = residual

    def forward(self, x, edge_index, adj_mat):
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply conv1 to generate features
        seq_fts = self.conv1(x)

        # Self-attention logits
        f_1 = self.f1(seq_fts).reshape(self.nb_nodes, 1)
        f_2 = self.f2(seq_fts).reshape(self.nb_nodes, 1)

        # Attention coefficients
        logits = adj_mat * f_1 + adj_mat * f_2.T
        coefs = F.softmax(F.leaky_relu(logits), dim=-1)

        if self.dropout > 0:
            coefs = F.dropout(coefs, p=self.dropout, training=self.training)
            seq_fts = F.dropout(seq_fts, p=self.dropout, training=self.training)

        # Multiply attention coefficients by node features
        vals = torch.matmul(coefs, seq_fts.squeeze())

        # Residual connection
        if self.residual:
            if x.shape[-1] != vals.shape[-1]:
                vals += self.conv1(x).squeeze()  # Residual adjustment
            else:
                vals += x.squeeze()

        return F.elu(vals.unsqueeze(0))