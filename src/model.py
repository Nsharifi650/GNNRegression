import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Optional
from src.config import Config


class GNNModel(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_dim1: int = 16,
        hidden_dim2: int = 8,
        dropout: float = 0.1
    ):
        super(GNNModel, self).__init__()
        self.dropout = dropout

        self.conv1 = GATConv(num_node_features, hidden_dim1)
        self.conv2 = GATConv(hidden_dim1, hidden_dim2)
        self.fc = nn.Linear(hidden_dim2, 1)  

    def forward(self, data, target_node_idx: Optional[int] = None) -> torch.Tensor:
        x, edge_index = data.x.clone(), data.edge_index

        if target_node_idx is not None:
            x[target_node_idx] = torch.zeros_like(x[target_node_idx])

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.fc(x)

        return out


def build_model_from_config(config: Config, input_dim: int) -> GNNModel:
    model = GNNModel(
        num_node_features=input_dim,
        hidden_dim1=config.model.hidden_dim1,
        hidden_dim2=config.model.hidden_dim2,
        dropout=config.model.dropout
    )
    return model
