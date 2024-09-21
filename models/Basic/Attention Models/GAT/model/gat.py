import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hid_units, n_heads, dropout, activation=F.elu, residual=False):
        super(GAT, self).__init__()

        self.n_heads = n_heads
        self.activation = activation
        self.dropout = dropout
        self.residual = residual

        # First GAT Layer
        self.attentions = torch.nn.ModuleList([
            GATConv(in_channels, hid_units[0], heads=n_heads[0], dropout=dropout, concat=True)
        ])
        
        # Hidden GAT layers
        self.hidden_attentions = torch.nn.ModuleList()
        for i in range(1, len(hid_units)):
            self.hidden_attentions.append(
                GATConv(hid_units[i-1] * n_heads[i-1], hid_units[i], heads=n_heads[i], dropout=dropout, concat=True)
            )

        # Final GAT Layer - updated to match the hidden layer output
        self.out_attention = GATConv(hid_units[-1] * n_heads[-1], out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Apply attention heads in the first layer
        x = torch.cat([att(x, edge_index) for att in self.attentions], dim=1)
        x = F.dropout(self.activation(x), p=self.dropout, training=self.training)

        # Apply hidden attention layers
        for layer in self.hidden_attentions:
            x_old = x  # Save for residual connection
            x = layer(x, edge_index)
            
            # Adjust for residual connection
            if self.residual and x.shape == x_old.shape:
                x += x_old  # Residual connection only if dimensions match

            x = F.dropout(self.activation(x), p=self.dropout, training=self.training)

        # Final output layer
        x = self.out_attention(x, edge_index)
        return F.log_softmax(x, dim=1)
