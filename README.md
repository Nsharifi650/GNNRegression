# GNN Regression for Sensor Data
This project implements a Graph Neural Network (GNN) using PyTorch Geometric to perform regression on sensor data. Each observation is represented as a graph, where nodes correspond to sensors and edges capture connectivity. The model learns to predict missing or masked node values using neighborhood information.

The pipeline includes:

Data preprocessing and graph construction
Configurable model training
Evaluation and visualization of predictions
Configuration is managed via a YAML file for modularity and reproducibility.

