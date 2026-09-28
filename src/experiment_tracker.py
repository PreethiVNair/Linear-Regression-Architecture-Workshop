import os
import pandas as pd
from datetime import datetime


def save_experiment(
    results_path,
    model_name,
    feature,
    learning_rate,
    iterations,
    test_size,
    metrics
):
    """Save the results of a machine learning experiment."""

    experiment = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "feature": feature,
        "learning_rate": learning_rate,
        "iterations": iterations,
        "test_size": test_size,
        "MSE": metrics["MSE"],
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "R2": metrics["R2"]
    }

    new_result = pd.DataFrame([experiment])

    if os.path.exists(results_path):
        new_result.to_csv(
            results_path,
            mode="a",
            header=False,
            index=False
        )
    else:
        new_result.to_csv(
            results_path,
            index=False
        )

    print("Experiment saved successfully.")


if __name__ == "__main__":
    print("experiment_tracker.py is working.")