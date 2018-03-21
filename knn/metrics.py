import numpy as np


def euclidian_distance(x, y):
    x = x[:, np.newaxis, :]
    y = y[np.newaxis, :, :]
    return (x**2 - 2*x*y + y**2).sum(axis=2) ** (1/2)

def cosine_distance(x, y):
    x = x[:, np.newaxis, :]
    y = y[np.newaxis, :, :]
    return 1 - (x*y).sum(axis=2) / (((x*x).sum(axis=2) ** (1/2)) * ((y*y).sum(axis=2) ** (1/2)))
