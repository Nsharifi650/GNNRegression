from src.config import load_config
from src.graph_builder import process_and_save_graphs
from src.train import run_training_pipeline
from src.evaluate import evaluate_model, plot_predictions
from src.model import GNNModel
from sklearn.model_selection import train_test_split
import torch


def main():
    config = load_config("config/config.yaml")
    process_and_save_graphs(config)
    graphs = torch.load(config.data.processed_graph_path, weights_only=False)
    train_size = 1 - config.training.test_size if hasattr(config.training, "test_size") else 0.95
    train_graphs, test_graphs = train_test_split(graphs, test_size=1 - train_size, random_state=42)
    run_training_pipeline(config, graphs_path=config.data.processed_graph_path)

    model = GNNModel(
        num_node_features=1,
        hidden_dim1=config.model.hidden_dim1,
        hidden_dim2=config.model.hidden_dim2,
        dropout=config.model.dropout
    )
    model.load_state_dict(torch.load(config.training.save_path))

    target_node_indices = config.evaluation.target_node_indices if config.evaluation else [1, 2, 3]
    actual, pred = evaluate_model(model, test_graphs, target_node_indices)

    plot_predictions(actual, pred)


if __name__ == "__main__":
    main()
