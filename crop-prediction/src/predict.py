"""Preprocessing of the Dataset."""

import json
import pandas as pd  # type: ignore
import joblib  # type: ignore


class PredictionMaker:
    """Preprocess the data for prediction."""

    def __init__(self, logger):
        self.logger = logger
        self.filename_encoder = "./encoders/encode.json"
        self.filename_scalar = "./feature_scaler/scale.pkl"
        self.filename_model = "./models"

    def decode_categorical_data(self, prediction):
        """Decode prediction class."""
        self.logger.info("Decoding prediction class...")
        label_mapping = {}
        prediction_class = "None"
        with open(self.filename_encoder, "r", encoding="utf-8") as f:
            label_mapping = json.load(f)
        for name, encoding in label_mapping.items():
            if prediction == encoding:
                prediction_class = name
                return prediction_class
        return prediction_class

    def feature_scaling(self, features):
        """Feature Scaling for prediction."""
        self.logger.info("Feature scaling for prediction...")
        scaler = joblib.load(self.filename_scalar)
        return scaler.transform(features)

    def predict(self, features, model):
        """Predict the class."""
        self.logger.info("Predicting class...")
        return model.predict(features)

    def load_model(self, model_name):
        """Load the model."""
        self.logger.info("Loading model...")
        return joblib.load(f"{self.filename_model}/{model_name}.pkl")

    def make_prediction(self, features, model_name="SimpleLayeredClassifier"):
        """Make prediction."""
        self.logger.info("Making prediction...")
        features = pd.DataFrame(features, index=[0])
        model = self.load_model(model_name)
        scaled_features = self.feature_scaling(features)
        prediction = self.predict(scaled_features, model)
        output = self.decode_categorical_data(prediction)
        self.logger.info(f"Prediction: {output}")
        return output
