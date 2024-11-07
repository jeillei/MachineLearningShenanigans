import numpy as np
import pandas as pd
import time
import random
from matplotlib import pyplot as plt

data = pd.read_csv('data/mnist_train.csv')
m, n = data.shape
data = np.array(data)

np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape


def init_params():
    W1 = np.random.rand(200,784) - 0.5
    b1 = np.random.rand(200, 1) - 0.5
    W2 = np.random.rand(10,200) - 0.5
    b2 = np.random.rand (10, 1) - 0.5
    return W1, b1, W2, b2

def relu (Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z)/sum(np.exp(Z))
    return A

def forward_prop(W1, b1,W2,b2,X):
    
    Z1 = np.dot(W1,X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2   
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def relu_deriv (Z):
    return Z>0

def one_hot(Y):
    ohy = np.zeros((Y.size, Y.max()+1))
    ohy[np.arange(Y.size),Y] = 1
    ohy = ohy.T
    return ohy

def backward_prop(Z1, A1, Z2, A2, W1,W2,X,Y):
    ohy = one_hot(Y)
    dZ2 = A2 - ohy
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2)
    dZ1 = np.dot(W2.T, dZ2) * relu_deriv(Z1)
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1)
    return db1, db2, dW1, dW2

def update_param(W1, dW1,b1,db1,W2,dW2,b2,db2,alpha):
    W1 = W1- alpha * dW1
    b1 = b1- alpha *db1
    W2 = W2- alpha * dW2
    b2 = b2 - alpha * db2
    return W1, W2, b1, b2

def predict(A2):
    # print("Predicted probabilities (distribution):")
    return np.argmax(A2,0)

def accuracy (prd, Y):
    print(prd, Y)
    return np.sum(prd == Y)/ Y.size

def grad (X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2, = forward_prop(W1, b1, W2, b2, X)
        db1, db2, dW1, dW2 = backward_prop(Z1, A1, Z2, A2, W1, W2,X,Y)
        W1, W2, b1, b2 = update_param(W1, dW1,b1, db1, W2, dW2, b2,db2, alpha)
        if i%10 == 0:
            print("iteratiion: ", i)
            predictions = predict(A2)
            print(accuracy(predictions, Y))
    return W1, b1, W2, b2

def save_weights(W1, b1, W2, b2):
    np.savetxt('W1.csv', W1, delimiter=', ')
    np.savetxt('b1.csv', b1, delimiter=', ')
    np.savetxt('W2.csv', W2, delimiter=', ')
    np.savetxt('b2.csv', b2, delimiter=', ')


def make_prdc(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1,W2,b2,X)
    predictions = predict(A2)
    return predictions

if __name__ == "__main__":
    W1, b1, W2, b2 = grad(X_train, Y_train, 0.1, 1500)

    save_weights(W1, b1, W2, b2)

    W1 = np.loadtxt('W1.csv', delimiter=',')
    b1 = np.loadtxt('b1.csv', delimiter=',')
    W2 = np.loadtxt('W2.csv', delimiter=',')
    b2 = np.loadtxt('b2.csv', delimiter=',')

    b1 = b1.reshape(-1, 1)
    b2 = b2.reshape(-1, 1)



# def test_prediction(index, W1, b1, W2, b2):
#     img = X_train[:, index, None]
#     prediction = make_prdc(X_train[:, index, None], W1, b1, W2, b2)
#     label = Y_train[index]
#     print("Prediction: ", prediction)
#     print("Label: ", label)

#     img = img.reshape((28, 28)) * 255
#     plt.gray()
#     plt.imshow(img, interpolation='nearest')
#     plt.axis('off')  # Hide axis
#     plt.show(block=False)  # Show the image
#     plt.pause(2)
#     plt.close()


# def loop_pred():
#     i =random.randint(0,1000)
#     test_prediction(i, W1, b1, W2, b2)

# try:
#     while True:
#         loop_pred()
#         time.sleep(1)
# except KeyboardInterrupt:
#     pass