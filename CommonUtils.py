import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = './mushroom_dataset/agaricus-lepiota.data'
data = pd.read_csv(file_path, header=None)


def preprocess_data(data):
    # Map the first column ('e', 'p') to binary values
    data[0] = data[0].apply(lambda x: 1 if x == 'p' else 0)  # 1 for poisonous, 0 for edible
    X = pd.get_dummies(data.iloc[:, 1:])
    y = data[0]

    return X, y


def k_fold_cross_validation(gradient_descent, classify, learning_rate):
    # Apply preprocessing
    X, y = preprocess_data(data)
    # K-Fold cross-validation
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []

    # K-Fold Loop
    for train_index, test_index in kf.split(X):
        # Split the data
        X_train, X_test = X.iloc[train_index].values, X.iloc[test_index].values
        y_train, y_test = y.iloc[train_index].values, y.iloc[test_index].values

        # Initialize weights and bias for each fold
        weights = np.random.rand(X_train.shape[1])
        bias = 0

        # Train the model using gradient descent
        weights, bias = gradient_descent(X_train, y_train, weights, bias, learning_rate, epochs)

        # Make predictions on the test set
        y_pred = classify(X_test, weights, bias)

        # Calculate accuracy and append to the list
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
