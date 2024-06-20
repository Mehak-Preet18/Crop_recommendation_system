"""Train and save Models."""

import json
import logging
import joblib  # type: ignore
import pandas as pd  # type: ignore
from sklearn.model_selection import GridSearchCV  # type: ignore
from sklearn.svm import SVC  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
from sklearn.gaussian_process import GaussianProcessClassifier  # type: ignore
from sklearn.tree import DecisionTreeClassifier  # type: ignore
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    StackingClassifier,
)  # type: ignore
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)  # type: ignore
from xgboost import XGBClassifier  # type: ignore
from lightgbm import LGBMClassifier  # type: ignore
from src.data import RawData

_param_grids = {
    "SVC": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]},
    "KNeighborsClassifier": {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"],
    },
    "DecisionTreeClassifier": {
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
    },
    "RandomForestClassifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
    },
    "LGBMClassifier": {"num_leaves": [31, 50, 100], "learning_rate": [0.05, 0.1, 0.2]},
    "AdaBoostClassifier": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 1.0],
    },
    "XGBClassifier": {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "n_estimators": [50, 100, 200],
    },
}

_classifiers = {
    "SVC": SVC(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "LGBMClassifier": LGBMClassifier(),
    "AdaBoostClassifier": AdaBoostClassifier(),
    "XGBClassifier": XGBClassifier(),
}

_models = [
    "KNeighborsClassifier",
    "DecisionTreeClassifier",
    "SVC",
    "RandomForestClassifier",
    "AdaBoostClassifier",
    "LGBMClassifier",
    "XGBClassifier",
    "SimpleLayeredClassifier",
    "MultiLayerClassifier",
]


class ModelTrainer:
    """Model Trainer."""

    def __init__(self, logger):
        self.logger = logger
        self.data_fetcher = RawData(self.logger)
        self.models = {name: None for name in _models}
        self.best_params = {name: None for name in _models}
        self.evaluation_results = {name: None for name in _models}
        self.trained_classifiers = {name: None for name in _models}
        self.result_filename = "./result/evaluation_results.json"
        self.best_params_filename = "./models/best_params.json"

    def train_models(self, x_train, y_train, models):
        """Train Models."""
        self.calculate_params(x_train, y_train)
        for name in models:
            self.logger.info(f"Training model {name}")
            if name in _classifiers:
                self.models[name] = self.trained_classifiers[name]
            elif name == "SimpleLayeredClassifier":
                self.models[name] = self.sl_classfier(x_train, y_train)
            elif name == "MultiLayerClassifier":
                self.models[name] = self.ml_classfier(x_train, y_train)
            else:
                raise ValueError(f"Unsupported training model {name}")
        self.logger.info("Training Models Complete.")

    def sl_classfier(self, x_train, y_train):
        """Simple Layered Classifier."""
        best_params = self.best_params
        base_classifiers = [
            ("svc", SVC(**best_params["SVC"])),
            ("knn", KNeighborsClassifier(**best_params["KNeighborsClassifier"])),
            ("dt", DecisionTreeClassifier(**best_params["DecisionTreeClassifier"])),
        ]
        final_estimator = RandomForestClassifier(
            **best_params["RandomForestClassifier"]
        )
        classifier_layered = StackingClassifier(
            estimators=base_classifiers,
            final_estimator=final_estimator,
            passthrough=True,
            n_jobs=-1,
        )
        params = {
            "final_estimator__n_estimators": [50, 100, 200],
            "final_estimator__max_depth": [None, 10, 20],
        }
        self.logger.info("Training Simple Layered Classifier...")
        search = GridSearchCV(classifier_layered, param_grid=params)
        search.fit(x_train, y_train)
        self.best_params["SimpleLayeredClassifier"] = search.best_params_
        return search

    def ml_classfier(self, x_train, y_train):
        """Multi Layer Classifier."""
        best_params = self.best_params
        multi_base_classifiers = [
            ("lgbm", LGBMClassifier(**best_params["LGBMClassifier"])),
            ("ada", AdaBoostClassifier(**best_params["AdaBoostClassifier"])),
            ("rd", RandomForestClassifier(**best_params["RandomForestClassifier"])),
            ("gpc", GaussianProcessClassifier()),
        ]
        multi_final_estimator = XGBClassifier(**best_params["XGBClassifier"])
        multi_classifier_layered = StackingClassifier(
            estimators=multi_base_classifiers,
            final_estimator=multi_final_estimator,
            passthrough=True,
            n_jobs=-1,
        )
        params = {
            "final_estimator__learning_rate": [0.1, 0.3, 0.5],
            "final_estimator__n_estimators": [50, 100, 200],
        }
        self.logger.info("Training Multi Layer Classifier...")
        multi_search = GridSearchCV(multi_classifier_layered, param_grid=params)
        multi_search.fit(x_train, y_train)
        self.best_params["MultiLayerClassifier"] = multi_search.best_params_
        return multi_search

    def calculate_params(self, x_train, y_train):
        """Calculate best parameters for models."""
        self.logger.info("Calculating best parameters for models...")
        for name, clf in _classifiers.items():
            self.logger.info(f"Calculating best parameters for {name}...")
            grid_search = GridSearchCV(clf, _param_grids[name], cv=3, n_jobs=-1)
            grid_search.fit(x_train, y_train)
            self.best_params[name] = grid_search.best_params_
            self.trained_classifiers[name] = grid_search

    def save_models(self):
        """Save Models."""
        for name, model in self.models.items():
            if model is not None:
                joblib.dump(model, f"./models/{name}.pkl")
                self.logger.info(f"Model {name} saved successfully.")

    def save_params(self):
        """Save Best Parameters."""
        with open(self.best_params_filename, "w", encoding="utf-8") as f:
            json.dump(self.best_params, f)
        self.logger.info("Best parameters saved successfully.")

    def evaluate_models(self, dataset):
        """Evaluate Models."""
        x_test, y_test = dataset["x_test"], dataset["y_test"]
        self.logger.info("Evaluating models...")
        for name, model in self.models.items():
            if model is not None:
                y_pred = model.predict(x_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="macro")
                recall = recall_score(y_test, y_pred, average="macro")
                f1 = f1_score(y_test, y_pred, average="macro")
                confusion = confusion_matrix(y_test, y_pred).tolist()
                self.evaluation_results[name] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "confusion_matrix": confusion,
                }
                self.logger.info(f"Evaluating model {name}...")
        with open(self.result_filename, "w", encoding="utf-8") as f:
            json.dump(self.evaluation_results, f)

    def train_evaluate_save_models(self, models=None):
        """Train, Evaluate and Save Models."""
        self.logger.info("Training Models...")
        dataset = self.data_fetcher.get_data()
        if models is None:
            models = _models
        self.train_models(
            x_train=dataset["x_train"], y_train=dataset["y_train"], models=models
        )
        self.save_models()
        self.save_params()
        self.evaluate_models(dataset)
        self.logger.info("Models trained and saved successfully.")
