"""Visualise the evaluation results of the model."""

import json
import plotly.graph_objects as go  # type: ignore


class ResultVisualiser:
    """Class for evaluating and plotting model metrics."""

    def __init__(self, logger):
        self.logger = logger
        self.file_path = "./result/evaluation_results.json"
        self.output_dir = "./results_plot/"
        self.data = None

    def read_json_file(self):
        """Read the JSON file."""
        self.logger.info("Reading Evaluation Results's JSON file...")
        with open(self.file_path, "r", encoding="utf-8") as file:
            self.data = json.load(file)

    def plot_confusion_matrix(self, confusion_matrix):
        """Plot the confusion matrix."""
        self.logger.info("Plotting confusion matrix...")
        fig = go.Figure(data=go.Heatmap(z=confusion_matrix, colorscale="Viridis"))
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Class",
            yaxis_title="True Class",
        )
        return fig

    def plot_metrics(self, model_name, classfication_report):
        """Plot the evaluation metrics."""
        self.logger.info("Plotting evaluation metrics Bar Plot...")
        fig = go.Figure(
            data=go.Bar(
                x=list(classfication_report.keys()),
                y=list(classfication_report.values()),
            )
        )
        fig.update_layout(
            title=f"Model Evaluation Metrics for {model_name}",
            xaxis_title="Metric",
            yaxis_title="Score",
        )
        return fig

    def plot_interactive(self):
        """Read JSON data and plot the metrics interactively."""
        self.logger.info("Plotting evaluation metrics and save interactively...")
        self.read_json_file()
        plots = {}
        for model_name, metrics in self.data.items():
            classfication_report = {
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1 Score": metrics["f1"],
            }
            confusion_matrix = metrics["confusion_matrix"]
            metrics_plot = self.plot_metrics(model_name, classfication_report)
            confusion_matrix_plot = self.plot_confusion_matrix(confusion_matrix)
            metrics_plot.write_image(f"{self.output_dir}{model_name}_metrics_plot.svg")
            confusion_matrix_plot.write_image(
                f"{self.output_dir}{model_name}_confusion_matrix_plot.svg"
            )
            plots[model_name] = {
                "metrics_plot": metrics_plot,
                "confusion_matrix_plot": confusion_matrix_plot,
            }
        return plots
