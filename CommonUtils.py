import pandas as pd


def get_X_y(file_path):
    # Load the dataset
    data = pd.read_csv(file_path, header=None)
    # Map the first column ('e', 'p') to binary values
    data[0] = data[0].apply(lambda x: 1 if x == 'p' else 0)  # 1 for poisonous, 0 for edible
    X = pd.get_dummies(data.iloc[:, 1:])  # one hot encoding
    y = data[0]
    return X, y
