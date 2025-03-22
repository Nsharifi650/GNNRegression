from typing import List
from typing import Optional
import yaml

from pydantic import BaseModel

class GraphConfig(BaseModel):
    nodes_order: List[str]
    edge_list: List[List[int]]

class DataPaths(BaseModel):
    raw_csv_path: str
    processed_graph_path: str

class Config(BaseModel):
    data: DataPaths
    graph: GraphConfig

def load_config(path: str) -> Config:
    with open(path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return Config(**config_dict)

class ModelConfig(BaseModel):
    hidden_dim1: int
    hidden_dim2: int
    dropout: float


class TrainingConfig(BaseModel):
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    pretrained: bool
    model_path: str
    save_path: str


class EvalConfig(BaseModel):
    target_node_indices: List[int]

class Config(BaseModel):
    data: DataPaths
    graph: GraphConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: Optional[EvalConfig] = None

