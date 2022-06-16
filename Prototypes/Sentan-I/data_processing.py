from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np


def load_data(path):
    data = pd.read_pickle(path)
    features = to_numpy(data['Composite_Vector'])
    data['Ordinal_Emotion'] = data['Ordinal_Emotion'].astype(int) - 1
    ordinal_labels = data['Ordinal_Emotion']
    label_encoding = data.groupby('Ordinal_Emotion')['Emotion'].unique()
    ordinal_labels = ordinal_labels.to_numpy().astype(np.int64)

    return features, ordinal_labels, label_encoding


def split_and_tensor(x, y, t_size=0.2):
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    y = F.one_hot(y, 8)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=t_size)

    return x_train, x_test, y_train, y_test


def to_numpy(x):
    data = np.array([])
    for v in x:
        data = np.concatenate((data, v), axis=0)
    data = data.reshape(-1, 193)
    return data
