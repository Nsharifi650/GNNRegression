import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler
from typing import List
from src.config import Config


def load_and_scale_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path).dropna()
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled


def build_edge_index(edge_list: List[List[int]]) -> torch.Tensor:
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    reversed_edges = edge_index[[1, 0], :]
    undirected_edges = torch.cat([edge_index, reversed_edges], dim=1)
    return undirected_edges


def create_graphs(
    df_scaled: pd.DataFrame, nodes_order: List[str], edge_index: torch.Tensor
) -> List[Data]:
    graphs = []

    for _, row in df_scaled.iterrows():
        node_features = []
        node_data_mask = []

        for node in nodes_order:
            if node in df_scaled.columns:
                node_features.append([row[node]])
                node_data_mask.append(1.0)
            else:
                node_features.append([0.0])
                node_data_mask.append(0.0)

        x = torch.tensor(node_features, dtype=torch.float)
        mask = torch.tensor(node_data_mask, dtype=torch.float)
        graphs.append(Data(x=x, edge_index=edge_index, mask=mask))

    return graphs


def process_and_save_graphs(config: Config) -> None:
    df_scaled = load_and_scale_data(config.data.raw_csv_path)
    edge_index = build_edge_index(config.graph.edge_list)
    graphs = create_graphs(df_scaled, config.graph.nodes_order, edge_index)
    torch.save(graphs, config.data.processed_graph_path)
    print(f"Saved {len(graphs)} graphs to {config.data.processed_graph_path}")
