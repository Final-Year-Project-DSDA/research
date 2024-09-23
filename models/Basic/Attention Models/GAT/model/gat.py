import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hid_units, n_heads, dropout, num_classes, activation=F.elu, residual=False):
        super(GAT, self).__init__()

        self.n_heads = n_heads
        self.activation = activation
        self.dropout = dropout
        self.residual = residual

        # Define GAT layers
        self.attentions = torch.nn.ModuleList([
            GATConv(in_channels, hid_units[0], heads=n_heads[0], dropout=dropout, concat=True)
        ])
        
        self.hidden_attentions = torch.nn.ModuleList()
        for i in range(1, len(hid_units)):
            self.hidden_attentions.append(
                GATConv(hid_units[i-1] * n_heads[i-1], hid_units[i], heads=n_heads[i], dropout=dropout, concat=True)
            )

        self.out_attention = GATConv(hid_units[-1] * n_heads[-1], hid_units[-1], heads=1, concat=False, dropout=dropout)

        # Dense layer to output the correct number of classes
        self.dense = torch.nn.Linear(hid_units[-1], num_classes)  # Output num_classes dimensions

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Apply attention layers
        x = torch.cat([att(x, edge_index) for att in self.attentions], dim=1)
        x = F.dropout(self.activation(x), p=self.dropout, training=self.training)

        for layer in self.hidden_attentions:
            x_old = x
            x = layer(x, edge_index)
            if self.residual:
                x += x_old
            x = F.dropout(self.activation(x), p=self.dropout, training=self.training)

        x = self.out_attention(x, edge_index)
        x = self.dense(x)

        return F.log_softmax(x, dim=1)  # Return log-softmax for NLL loss
