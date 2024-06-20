"""Interface to run the project."""

from src.train import ModelTrainer
from src.predict import PredictionMaker
from src.log_utils import setup_logger
from src.visualise_data import VisualiseData
from src.visualise_result import ResultVisualiser


def main():
    """Main function to run the project."""
    logger = setup_logger()
    model_trainer = ModelTrainer(logger)
    data_visualiser = VisualiseData(logger)
    result_visualiser = ResultVisualiser(logger)
    prediction_maker = PredictionMaker(logger)

    model_trainer.train_evaluate_save_models(models=None)
    # model_trainer.train_evaluate_save_models(models=["SomeModel"])

    list_of_plots1 = data_visualiser.visualize_data()

    list_of_plots2 = result_visualiser.plot_interactive()

    input_fields = {
        "nitrogen": 90,
        "phosphorous": 42,
        "potassium": 43,
        "temperature": 20.87974371,
        "humidity": 82.00274423,
        "ph": 6.502985292,
        "rainfall": 202.9355362,
    }
    model = "SimpleLayeredClassifier"
    print(prediction_maker.make_prediction(input_fields, model_name=model))
    plots = list_of_plots1 + list_of_plots2
    return plots


if __name__ == "__main__":
    main()
