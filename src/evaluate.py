import torch
from torch_geometric.data import DataLoader
from typing import List, Tuple
from src.model import GNNModel
from torch_geometric.data import Data
import plotly.graph_objects as go


def evaluate_model(model: GNNModel, test_graphs: List[Data], target_node_indices: List[int]) -> Tuple[List[float], List[float]]:
    model.eval()
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)

    actual = []
    pred = []

    with torch.no_grad():
        for data in test_loader:
            mask = data.mask
            for i in target_node_indices:
                if mask[i] == 1:
                    output = model(data, i)
                    prediction = output[i].view(1)
                    target = data.x[i]

                    actual.append(target.item())
                    pred.append(prediction.item())

    return actual, pred


def plot_predictions(actual: List[float], pred: List[float]):
    scatter_trace = go.Scatter(
        x=actual,
        y=pred,
        mode='markers',
        marker=dict(
            size=10,
            opacity=0.5,
            color='rgba(255,255,255,0)',
            line=dict(
                width=2,
                color='rgba(152, 0, 0, .8)',
            )
        ),
        name='Actual vs Predicted'
    )

    line_trace = go.Scatter(
        x=[min(actual), max(actual)],
        y=[min(actual), max(actual)],
        mode='lines',
        marker=dict(color='blue'),
        name='Perfect Prediction'
    )

    layout = dict(
        title='Actual vs Predicted Values',
        xaxis=dict(title='Actual Values'),
        yaxis=dict(title='Predicted Values'),
        autosize=False,
        width=800,
        height=600
    )

    fig = go.Figure(data=[scatter_trace, line_trace], layout=layout)
    fig.show()
