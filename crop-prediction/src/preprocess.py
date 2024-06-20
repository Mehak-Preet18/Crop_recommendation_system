"""Preprocessing of the Dataset."""

import json
import joblib  # type: ignore
from sklearn.impute import SimpleImputer  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
import pandas as pd  # type: ignore


_STRATEGY = "mean"


class DatasetPreprocessor:
    """Dataset Preprocessor."""

    def __init__(self, logger):
        self.logger = logger
        self.filename_encoder = "./encoders/encode.json"
        self.filename_scalar = "./feature_scaler/scale.pkl"

    def encode_categorical_data(self, predict):
        """Encode categorical data."""
        self.logger.info("Encoding categorical data...")
        label_mapping = {value: idx for idx, value in enumerate(predict.unique())}
        with open(self.filename_encoder, "w", encoding="utf-8") as f:
            json.dump(label_mapping, f)
        return predict.map(label_mapping)

    def handle_missing_data(self, features):
        """Handle missing data."""
        self.logger.info("Handling missing data...")
        imputer = SimpleImputer(strategy=_STRATEGY)
        handle_data = imputer.fit_transform(features)
        return pd.DataFrame(handle_data, columns=features.columns)

    def feature_scaling(self, features):
        """Feature Scaling for prediction."""
        self.logger.info("Performing feature scaling...")
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        joblib.dump(scaler, self.filename_scalar)
        return scaled_features

    def preprocess(self, features, predict):
        """Preprocess the data."""
        self.logger.info("Preprocessing data...")
        encoded_predict = self.encode_categorical_data(predict)
        handled_features = self.handle_missing_data(features)
        scaled_features = self.feature_scaling(handled_features)
        return {"features": scaled_features, "predict": encoded_predict}
