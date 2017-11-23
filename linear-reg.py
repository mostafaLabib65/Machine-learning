import numpy as np
import matplotlib.pyplot as plt
import mnist
#some constants
m = 10
alpha = 0.01
n = 2
itr = 1500
theta = np.random.randn(n,1)
j = []

#data
X = np.random.randint(-10,10,[m,n])
Y = np.sum(X,axis=1).reshape(m,1)


#linear-reg (grediant descent)
for i in range(itr):
    predections = np.array(X.dot(theta)).reshape(m, 1)
    error = (predections - Y)
    J = (1/(2*m)) * np.sum((error * error),axis=0)
    j.append(J)
    theta = theta - (alpha/m) * np.sum(error * X, axis=0).reshape(n,1)

print(theta)

plt.plot(j)
plt.ylabel('error function')
plt.show()