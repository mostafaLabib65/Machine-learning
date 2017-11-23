import mnist
import numpy as np
import helper as hl
import matplotlib.pyplot as plt
import os

w_path = './weights'
p_path = './plots'
report_interval = 50

epochs = 1000
alpha_initial = 2
alpha_final = 0.5
k_classes = 10
m = 10000
batch_size = 128
iterations = int(m / batch_size)
activation = hl.softmax

images, labels = mnist.load_mnist()
print(images.shape)

# Load and ravel the images
X = np.array([k.ravel() for k in images])[0:m, :].T

# Insert 1 at the beginning of each image
X = np.insert(X, 0, 1, axis=0)
print(X.shape)
n_features = X.shape[0]

# Normalize the data
X = X / 255

# Initialize the weights
w1 = np.random.randn(20, n_features)
w2 = np.random.randn(10, 20)
print(w1.shape)

z1 = w1.dot(X)
a1 = hl.sigmoid(z1)
z2 = w2.dot(a1)
a2 = hl.sigmoid(z2)






