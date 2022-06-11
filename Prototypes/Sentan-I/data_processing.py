from sklearn.model_selection import train_test_split
import torch
import numpy as np
import pandas as pd

# Divide los conjuntos de datos
# Y los transforma a su representación vectorial


def split_and_transform(x, y, t_size=0.2):
    x = x_flat(x)
    y = encode(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=t_size)

    x_train = x_to_tensor(x_train)
    x_test = x_to_tensor(x_test)

    y_train = torch.from_numpy(y_train).type(torch.float32)
    y_test = torch.from_numpy(y_test).type(torch.float32)
    return x_train, x_test, y_train, y_test


# x entra como objeto de numpy y se transforma
# En su version de matriz caracteristica
def x_flat(x):
    x = x.to_numpy()
    new_x = []
    for n_rows in x:
        base = np.array([])
        for row in n_rows:
            base = np.concatenate((base, row), axis=None)
        new_x.append(base)
    return np.array(new_x)


# Transforma X a su version tensorial
# Pasa de ser NxL a  NxCxL ya que torch necesita
# Una dimension para sus canales
def x_to_tensor(x):
    return torch.unsqueeze(torch.tensor(x), dim=1).type(torch.float32)


# Convierte las entradas categoricas del modelo en
# Entradas codificadas como one hot encoding
def encode(y):
    y = pd.get_dummies(y)
    return np.array(y.values.tolist())
