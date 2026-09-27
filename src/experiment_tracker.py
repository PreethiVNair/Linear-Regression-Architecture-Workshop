from pathlib import Path
from datetime import datetime
import pandas as pd


def save_experiment(
    feature,
    test_size,
    learning_rate,
    iterations,
    rmse,
    mae,
    r2
):
    """Save the results of a linear regression experiment."""

    results_path = Path("experiments/results.csv")

    # Create experiments folder if it does not exist
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Information from the current experiment
    experiment = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature": feature,
        "test_size": test_size,
        "learning_rate": learning_rate,
        "iterations": iterations,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

    # Convert experiment information into a DataFrame
    experiment_df = pd.DataFrame([experiment])

    # Check whether results.csv already exists
    if results_path.exists():
        experiment_df.to_csv(
            results_path,
            mode="a",
            header=False,
            index=False
        )
    else:
        experiment_df.to_csv(
            results_path,
            index=False
        )

    print("Experiment results saved to:", results_path)
    
    
    
if __name__ == "__main__":

    # Temporary values used only to test the experiment tracker
    save_experiment(
        feature="MedInc",
        test_size=0.2,
        learning_rate=0.01,
        iterations=1000,
        rmse=0.72,
        mae=0.54,
        r2=0.58
    )