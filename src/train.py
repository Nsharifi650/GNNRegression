from typing import List

import torch
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.nn import MSELoss
from torch_geometric.data import Data

from src.config import Config
from src.model import GNNModel



def split_dataset(graphs: List[Data], test_size: float = 0.05, seed: int = 42):
    indices = list(range(len(graphs)))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed)
    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]
    return train_graphs, test_graphs

def train_model(
    model: GNNModel,
    train_graphs: List[Data],
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    target_node_idx: int = 4
):
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = MSELoss()

    model.train()

    for epoch in range(num_epochs):
        accumulated_loss = 0.0

        for data in train_loader:
            optimizer.zero_grad()
            loss = 0.0

            # Loop over each graph in the batch
            for graph in data.to_data_list():  # Unpack batch into individual graphs
                mask = graph.mask
                for i in range(graph.num_nodes):
                    if mask[i] == 1 and i != target_node_idx:
                        output = model(graph, i)
                        target = graph.x[i]
                        prediction = output[i].view(1)
                        loss += criterion(prediction, target)

            loss.backward()
            optimizer.step()
            accumulated_loss += loss.item()

        average_loss = accumulated_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.6f}")

def run_training_pipeline(config: Config, graphs_path: str):
    print("Loading graphs...")
    graphs = torch.load(graphs_path, weights_only=False)

    train_graphs, _ = split_dataset(graphs, test_size=0.05)

    model = GNNModel(
        num_node_features=1,
        hidden_dim1=config.model.hidden_dim1,
        hidden_dim2=config.model.hidden_dim2,
        dropout=config.model.dropout
    )

    if config.training.pretrained:
        print("Loading pretrained model...")
        model.load_state_dict(torch.load(config.training.model_path))

    train_model(
        model=model,
        train_graphs=train_graphs,
        num_epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        learning_rate=config.training.lr,
        weight_decay=config.training.weight_decay
    )

    torch.save(model.state_dict(), config.training.save_path)
    print(f"Model saved to {config.training.save_path}")
