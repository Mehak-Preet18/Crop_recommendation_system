"""Fetch Data for Training"""

import pandas as pd  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.utils import shuffle  # type: ignore
from src.preprocess import DatasetPreprocessor


_TEST_SIZE = 0.3
_RANDOM_STATE = 42


class RawData:
    """Raw Data Fetcher"""

    def __init__(self, logger):
        self.logger = logger
        self.preprocessor = DatasetPreprocessor(self.logger)
        self.dataset_path = "./data/data.csv"
        self.predict_class = "label"

    def read_dataset(self):
        """Read Dataset from CSV."""
        self.logger.info("Reading Dataset...")
        dataset = pd.read_csv(self.dataset_path)
        dataset = shuffle(dataset, random_state=_RANDOM_STATE)
        return dataset

    def preprocess_dataset(self, dataset):
        """Preprocess Dataset."""
        self.logger.info("Preprocessing Dataset...")
        predict = dataset[self.predict_class]
        features = dataset.drop(columns=[self.predict_class])
        return self.preprocessor.preprocess(features=features, predict=predict)

    def split_dataset(self, dataset):
        """Split Dataset into Train and Test Sets."""
        self.logger.info("Splitting Dataset...")
        x_train, x_test, y_train, y_test = train_test_split(
            dataset["features"],
            dataset["predict"],
            test_size=_TEST_SIZE,
            random_state=_RANDOM_STATE,
        )
        return {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    def get_data(self):
        """Get Data."""
        self.logger.info("Fetching Data...")
        data = self.read_dataset()
        data = self.preprocess_dataset(data)
        data = self.split_dataset(data)
        return data
