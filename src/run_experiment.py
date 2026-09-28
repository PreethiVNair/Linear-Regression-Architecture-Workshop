from data_loader import load_csv, load_config
from preprocessing import prepare_data
from model import gradient_descent, train_sklearn_model
from evaluation import evaluate_model
from experiment_tracker import save_experiment


def main():
   
    # 1. Load experiment configuration
    config = load_config("configs/experiment_config.yaml")

    # Get settings from configuration
    csv_path = config["data"]["csv_path"]

    feature = config["model"]["selected_feature"]
    target = config["model"]["target"]

    learning_rate = config["training"]["learning_rate"]
    iterations = config["training"]["iterations"]
    test_size = config["training"]["test_size"]
    random_state = config["training"]["random_state"]

    results_path = config["experiments"]["results_path"]


    
    # 2. Load dataset
    data = load_csv(csv_path)

    print("Dataset loaded successfully.")
    print("Dataset shape:", data.shape)


    # 3. Prepare the data
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        data=data,
        feature=feature,
        target=target,
        test_size=test_size,
        random_state=random_state
    )

    print("Data preprocessing completed.")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))


    # 4. Train linear regression from scratch
    theta0, theta1, cost_history = gradient_descent(
        X_train,
        y_train,
        learning_rate=learning_rate,
        iterations=iterations
    )

    print("From-scratch model training completed.")


    # 5. Make predictions using from-scratch model
    y_pred_scratch = theta0 + theta1 * X_test.flatten()


    # 6. Evaluate from-scratch model
    scratch_metrics = evaluate_model(
        y_test,
        y_pred_scratch
    )

    print("From-scratch model evaluation:")
    print(scratch_metrics)


    # 7. Save from-scratch experiment results
    save_experiment(
        results_path=results_path,
        model_name="From Scratch Linear Regression",
        feature=feature,
        learning_rate=learning_rate,
        iterations=iterations,
        test_size=test_size,
        metrics=scratch_metrics
    )


    # 8. Train Scikit-Learn linear regression model
    sklearn_model = train_sklearn_model(
        X_train,
        y_train
    )

    print("Scikit-learn model training completed.")


    # 9. Make Scikit-Learn predictions
    y_pred_sklearn = sklearn_model.predict(X_test)


    # 10. Evaluate Scikit-Learn model
    sklearn_metrics = evaluate_model(
        y_test,
        y_pred_sklearn
    )

    print("Scikit-learn model evaluation:")
    print(sklearn_metrics)


    # 11. Save Scikit-Learn experiment results
    save_experiment(
        results_path=results_path,
        model_name="Scikit-Learn Linear Regression",
        feature=feature,
        learning_rate=learning_rate,
        iterations=iterations,
        test_size=test_size,
        metrics=sklearn_metrics
    )

    print("Experiment pipeline completed successfully.")


if __name__ == "__main__":
    main()