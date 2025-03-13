from mlFlowProject.config.configuration import ConfigurationManager
from mlFlowProject.components.data_transformation import DataTransformation
from mlFlowProject.components.feature_store import FeatureStore
from mlFlowProject import logger
from pathlib import Path
import pandas as pd

STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(config=data_transformation_config)
                X, y = data_transformation.preprocess_data(pd.read_csv(data_transformation_config.data_path))

                feature_store = FeatureStore(config=data_transformation_config)
                feature_store.save_features(X, "features.csv")
                feature_store.save_metadata({"target_column": data_transformation_config.target_column}, "metadata.json")

                data_transformation.train_test_splitting()

            else:
                raise Exception("Data schema is not valid")

        except Exception as e:
            print(e)

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
