import numpy as np


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def softmax(x, kaxis=0):
    e = np.exp(x)
    return e / np.sum(e, axis=kaxis)