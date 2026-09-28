import numpy as np
from sklearn.linear_model import LinearRegression


def gradient_descent(
    X,
    y,
    learning_rate=0.01,
    iterations=1000
):
    """Train linear regression from scratch."""

    m = len(y)
    X = X.flatten()

    theta0 = 0.0
    theta1 = 0.0

    cost_history = []

    for _ in range(iterations):

        # Make predictions
        predictions = theta0 + theta1 * X

        # Calculate errors
        errors = predictions - y

        # Calculate MSE
        mse = (1 / m) * np.sum(errors ** 2)

        cost_history.append(mse)

        # Calculate gradients
        d_theta0 = (2 / m) * np.sum(errors)

        d_theta1 = (
            (2 / m) * np.sum(errors * X)
        )

        # Update parameters
        theta0 -= learning_rate * d_theta0
        theta1 -= learning_rate * d_theta1

    return theta0, theta1, cost_history


def train_sklearn_model(X_train, y_train):
    """Train scikit-learn linear regression."""

    model = LinearRegression()

    model.fit(X_train, y_train)

    return model


if __name__ == "__main__":
    print("model.py is working.")