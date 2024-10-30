import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = '../mushroom_dataset/agaricus-lepiota.data'
data = pd.read_csv(file_path, header=None)


def preprocess_data(data):
    # Map the first column ('e', 'p') to binary values
    data[0] = data[0].apply(lambda x: 1 if x == 'p' else 0)  # 1 for poisonous, 0 for edible
    X = pd.get_dummies(data.iloc[:, 1:])  # one hot encoding
    y = data[0]

    return X, y


# Apply preprocessing
X, y = preprocess_data(data)
NUM_OF_FEATURES = X.shape[1]
weights = np.random.random(NUM_OF_FEATURES)
bias = 0  # I randomly init as zero
learning_rate = 0.01
epochs = 2 * (10 ** 5)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# prediction function
def pred(x, p_weights, p_bias):
    return sigmoid(np.dot(x, p_weights) + p_bias)


def classify(x, cl_weights, cl_bias, threshold=0.5):
    y_pred = pred(x, cl_weights, cl_bias)
    return [1 if p >= threshold else 0 for p in y_pred]


def compute_loss(y, y_pred):
    return - np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


def gradient_descent(x, y, gd_weights, gd_bias, gd_learning_rate, gd_epochs):
    n = len(y)
    for eee in range(gd_epochs):
        y_pred = pred(x, gd_weights, gd_bias)

        dw = (1 / n) * np.dot(x.T, (y_pred - y))
        db = (1 / n) * np.sum(y_pred - y)

        gd_weights -= gd_learning_rate * dw
        gd_bias -= gd_learning_rate * db

        if eee % 100 == 0:
            loss = compute_loss(y, y_pred)
            print(f"Epoch {eee}/{gd_epochs}, Loss: {loss}")

    return gd_weights, gd_bias


# K-Fold cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
accuracies = []

# K-Fold Loop
for train_index, test_index in kf.split(X):
    # Split the data
    X_train, X_test = X.iloc[train_index].values, X.iloc[test_index].values
    y_train, y_test = y.iloc[train_index].values, y.iloc[test_index].values

    # Train the model using gradient descent
    weights, bias = gradient_descent(X_train, y_train, weights, bias, learning_rate, epochs)

    # Make predictions on the test set
    y_pred = classify(X_test, weights, bias)

    # Calculate accuracy and append to the list
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
