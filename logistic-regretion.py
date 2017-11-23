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
w = np.random.randn(n_features, k_classes)
print(w.shape)




# Create a 0-1 probability matrix

##############################
y = np.zeros((k_classes, m))##
y[labels[0:m], range(m)] = 1##
##############################
js = []
errors = []


def calculate_error(portion=1):
    wrong = 0.
    r = int(m*portion)
    for i in range(r):
        if r == m:
            ex = X[:, i]
        else:
            ex = X[:, np.random.randint(0, m)]
        prediction = np.argmax(activation(np.dot(ex, w)))
        if prediction != labels[i]:
            wrong += 1
    return wrong / r * 100


def batch_gd(alpha):
    # Batch Gradient Descent
    global w
    score = np.dot(w.T, X[:, 0:m])
    h = activation(score)
    delta = (h-y[:,0:m])
    g = alpha / m * np.dot(X[:, 0:m], delta.T)
    w -= g


def sgd(alpha):
    global w
    for j in range(m):
        sample_idx = np.random.randint(0, m - batch_size)
        sample = X[:, sample_idx]
        score = np.dot(w.T, sample)
        h = activation(score)
        delta = (h - y[:, sample_idx])
        g = alpha / m * np.dot(sample.reshape(n_features, 1), delta.reshape(1, k_classes))
        w -= g


def mini_batch_gd(alpha):
    # Mini-Batch Gradient Descent
    global w
    # At each epoch, we calculate statistics to how the model improves or declines
    for j in range(iterations):
        # At each iteration, we refine the model
        batch_idx = np.random.randint(0, m - batch_size)
        batch = X[:, batch_idx:batch_idx+batch_size]
        score = np.dot(w.T, batch)
        h = activation(score)
        delta = (h-y[:, batch_idx:batch_idx+batch_size])
        g = alpha / m * np.dot(batch, delta.T)
        w -= g


for i in range(epochs):
    ##print('Epoch', i)
    if i % report_interval == 0:
        score = np.dot(w.T, X)
        h = hl.sigmoid(score)
        j = -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h), axis=0) / m
        js.append(j)
        error = calculate_error()
        errors.append(error)
        print('Error', error, '%')

    # Annealing alpha over time
    alpha = alpha_initial - i/epochs * (alpha_initial - alpha_final)

    # You can comment out two to test the third
    # batch_gd(alpha)
    # sgd(alpha)
    batch_gd(alpha)

js = np.array(js)

print('Final Error', calculate_error(), '%')

rpt = int(epochs / report_interval)
plt.plot(range(rpt), js[:, 0], color='black', label='J0')
plt.plot(range(rpt), js[:, 1], color='cyan', label='J1')
plt.plot(range(rpt), js[:, 2], color='blue', label='J2')
plt.plot(range(rpt), js[:, 3], color='gray', label='J3')
plt.plot(range(rpt), js[:, 4], color='orange', label='J4')
plt.plot(range(rpt), js[:, 5], color='yellow', label='J5')
plt.plot(range(rpt), js[:, 6], color='green', label='J6')
plt.plot(range(rpt), js[:, 7], color='purple', label='J7')
plt.plot(range(rpt), js[:, 8], color='brown', label='J8')
plt.plot(range(rpt), js[:, 9], color='magenta', label='J9')
################################
if not os.path.exists(p_path):##
    os.makedirs(p_path)       ##
################################
plt.legend()
plt.savefig(os.path.join(p_path, 'JPlot.png'))
plt.show()

plt.plot(range(rpt), errors, color='red', label='Total Error')

plt.savefig(os.path.join(p_path, 'ErrorPlot.png'))
plt.legend()
plt.show()

if not os.path.exists(w_path):
    os.makedirs(w_path)

np.save(os.path.join(w_path, 'linreg_mnist'), w)

images, labels = mnist.load_mnist(dataset='testing')
while True:
    n = np.random.randint(0, labels.shape[0])
    print('\n\nGround Truth:', labels[n])
    ex = images[n].ravel()
    ex = np.insert(ex, 0, 1) / 255
    h = activation(np.dot(ex, w))
    best_match = np.argmax(h)
    print('Model Prediction:', best_match)
    print('Confidence: %.2f %%' % (h[best_match]*100))
    plt.imshow(images[n, :, :],cmap="gray")
    plt.show()