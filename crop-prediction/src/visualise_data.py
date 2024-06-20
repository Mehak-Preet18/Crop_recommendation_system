"""Visualise the dataset."""

import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore


class VisualiseData:
    """Visualise the dataset."""

    def __init__(self, logger):
        self.logger = logger
        self.data = None
        self.filename_data = "./data/data.csv"
        self.exclude = "label"
        self.output_dir = "./plots/"

    def load_data(self):
        """Load the data."""
        self.logger.info("Loading data...")
        self.data = pd.read_csv(self.filename_data)

    def plot_unique_label_counts(self):
        """Plot data available for each unique label value."""
        self.logger.info("Plotting unique label counts...")
        label_counts = self.data[self.exclude].value_counts()
        fig_unique_label = px.bar(
            x=label_counts.index,
            y=label_counts.values,
            labels={"x": "Label", "y": "Count"},
            title="Data Available for Each Unique Label",
        )
        return [fig_unique_label]  # Return as a list

    def plot_range_of_values(self):
        """Plot range of value for each column for each label value."""
        columns = self.data.columns.difference([self.exclude])
        fig_range_values = []
        for _, label_value in enumerate(self.data["label"].unique()):
            self.logger.info(f"Plotting range of values for label {label_value}...")
            label_data = self.data[self.data["label"] == label_value]
            fig = px.box(
                label_data,
                y=list(columns),
                title=f"Range of Values for Label {label_value}",
            )
            fig.update_layout(showlegend=False)
            fig_range_values.append(fig)
        return fig_range_values

    def save_plots(self, plots):
        """Save plots as SVG files."""
        for fig in plots:
            x = fig.layout["title"]["text"]
            self.logger.info(f"Saving plot {x}...")
            fig.write_image(f"{self.output_dir}{x}.svg")

    def visualize_data(self):
        """Visualize the dataset."""
        self.logger.info("Visualizing the dataset...")
        self.load_data()
        unique_label_plot = self.plot_unique_label_counts()
        range_of_values_plots = self.plot_range_of_values()
        print(unique_label_plot + range_of_values_plots)
        self.save_plots(unique_label_plot + range_of_values_plots)
        return unique_label_plot + range_of_values_plots
