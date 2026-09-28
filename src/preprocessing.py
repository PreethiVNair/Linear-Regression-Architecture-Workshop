from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data(data, feature, target, test_size=0.2, random_state=42):

    # Remove rows with missing feature or target
    clean_data = data.dropna(
        subset=[feature, target]
    ).copy()

    # Select input feature
    X = clean_data[[feature]].values

    # Select target
    y = clean_data[target].values

    # Split into training and testing data
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    # Standardize the input feature
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    print("preprocessing.py is working.")