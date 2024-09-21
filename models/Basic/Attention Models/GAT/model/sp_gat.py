import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class SpGAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hid_units, n_heads, dropout, activation=F.elu, residual=False):
        super(SpGAT, self).__init__()
        
        self.n_heads = n_heads
        self.activation = activation
        self.dropout = dropout
        self.residual = residual

        # Initial attention layers
        self.attentions = torch.nn.ModuleList([
            GATConv(in_channels, hid_units, heads=n_heads[0], dropout=dropout, concat=True)
            for _ in range(n_heads[0])
        ])
        
        # Hidden attention layers
        self.hidden_attentions = []
        for i in range(1, len(hid_units)):
            self.hidden_attentions.append(torch.nn.ModuleList([
                GATConv(hid_units * n_heads[i-1], hid_units, heads=n_heads[i], dropout=dropout, concat=True)
                for _ in range(n_heads[i])
            ]))

        # Output attention layer
        self.out_attention = GATConv(hid_units * n_heads[-2], out_channels, heads=n_heads[-1], concat=False, dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Apply attention heads in the first layer
        x = torch.cat([att(x, edge_index) for att in self.attentions], dim=1)
        x = F.dropout(self.activation(x), p=self.dropout, training=self.training)

        # Apply hidden attention layers
        for layer in self.hidden_attentions:
            x_old = x
            x = torch.cat([att(x, edge_index) for att in layer], dim=1)
            if self.residual:
                x += x_old  # Residual connection

            x = F.dropout(self.activation(x), p=self.dropout, training=self.training)

        # Final output layer
        x = self.out_attention(x, edge_index)
        return F.log_softmax(x, dim=1)
