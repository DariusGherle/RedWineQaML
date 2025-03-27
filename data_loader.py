import pandas as pd


def load_and_prepare_data(filepath):
    train_data = pd.read_csv(filepath)

    features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                'pH', 'sulphates', 'alcohol']

    X = train_data[features]
    y = train_data['quality']

    return train_data, X, y
