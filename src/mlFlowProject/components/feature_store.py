import pandas as pd
from pathlib import Path
from mlFlowProject.utils.common import save_json, load_json
from mlFlowProject.entity.config_entity import DataTransformationConfig

class FeatureStore:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def save_features(self, features: pd.DataFrame, file_name: str):
        file_path = Path(self.config.root_dir) / file_name
        features.to_csv(file_path, index=False)
        print(f"Features saved to {file_path}")

    def load_features(self, file_name: str) -> pd.DataFrame:
        file_path = Path(self.config.root_dir) / file_name
        features = pd.read_csv(file_path)
        print(f"Features loaded from {file_path}")
        return features

    def save_metadata(self, metadata: dict, file_name: str):
        file_path = Path(self.config.root_dir) / file_name
        save_json(file_path, metadata)
        print(f"Metadata saved to {file_path}")

    def load_metadata(self, file_name: str) -> dict:
        file_path = Path(self.config.root_dir) / file_name
        metadata = load_json(file_path)
        print(f"Metadata loaded from {file_path}")
        return metadata
