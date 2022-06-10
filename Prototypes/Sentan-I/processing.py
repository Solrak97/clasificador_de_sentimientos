from torch import tensor
import numpy as np
import pandas as pd

def pipeline(x, y):
    pass


# x entra como objeto de numpy
def x_flat(x):
    x = x.to_numpy()
    new_x = []
    for n_rows in x:
        base = np.array([])
        for row in n_rows:
            base = np.concatenate((base, row), axis=None)
        new_x.append(base)
    return np.array(new_x)


def encode(y):
    y = pd.get_dummies(y)
    return np.array(y.values.tolist())


# x entra como objeto de numpy
def x_transform(x, shape):
    x = x.to_numpy()

    pass


# y entra como serie de numpy
def y_transform(y, shape):
    pass