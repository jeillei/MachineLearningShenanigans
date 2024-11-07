import numpy as np
import pandas as pd
import time
import random
from matplotlib import pyplot as plt

data = pd.read_csv('/home/jelly/Downloads/mnist_train.csv')
m, n = data.shape
data = np.array(data)


np.random.shuffle(data)

data_dev = data[0:100].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[100:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

print(X_train.shape)
def xinit(fin,fout):
    limit = np.sqrt(6/float(fin+fout))
    W = np.random.uniform(low = -limit, high=limit, size=(fin,fout))
    return W

def init_params():
    W1 = xinit(500,784)
    b1 = np.zeros((500, 1))
    W2 = xinit(300,500)
    b2 = np.zeros((300,1))
    W3 = xinit(10,300)
    b3 = np.zeros((10, 1))
    return W1, b1, W2, b2, W3, b3

def leaky_relu (Z):
    return np.maximum(Z, 0.01 * Z)

def softmax(Z):
    expz = np.exp(Z - np.max(Z, axis=0, keepdims=True)) 
    return expz / np.sum(expz, axis=0, keepdims=True)


def forward_prop(W1, b1,W2,b2,W3, b3, X):
    
    Z1 = np.dot(W1,X) + b1
    A1 = leaky_relu(Z1)
    Z2 = np.dot(W2, A1) + b2   
    A2 = leaky_relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def leaky_relu_deriv (Z):
    return np.where(Z>0, 1, 0.01)

def one_hot(Y):
    ohy = np.zeros((Y.size, Y.max()+1))
    ohy[np.arange(Y.size),Y] = 1
    ohy = ohy.T
    return ohy

def clip_grad (max, Z):
    norm = np.linalg.norm(Z)
    if norm > max:  
        Z *= max/norm
    return Z

def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    ohy = one_hot(Y)
    dZ3 = A3 - ohy
    # dZ3 = clip_grad(20, dZ3)
    dW3 = 1/m * np.dot(dZ3, A2.T)
    db3 = 1/m * np.sum(dZ3)

    dZ2 = np.dot(W3.T, dZ3) * leaky_relu_deriv(Z2)
    # dZ2 = clip_grad(20, dZ2)
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2)

    dZ1 = np.dot(W2.T, dZ2) * leaky_relu_deriv(Z1)
    # dZ1 = clip_grad(20, dZ1)
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1)
    return db1, db2, db3, dW1, dW2, dW3

def update_parameters(W1, dW1,mW1, vW1, beta1, beta2, i, alpha):
    mW1 = beta1*mW1 + (1-beta1)*dW1
    vW1 = beta2*vW1 + (1-beta2)*(dW1**2)
    mW1_hat = mW1/(1-beta1**i)
    vW1_hat = vW1/(1-beta2**i)
    W1 = W1 - alpha*mW1_hat/(np.sqrt(vW1_hat)+1e-8)
    return W1 ,mW1, vW1

def predict(A2):
    return np.argmax(A2,0)

def accuracy (prd, Y):
    # print(Y)
    return np.sum(prd == Y)/ Y.size


def descent (X, Y, alpha, iterations):
    W1, b1, W2, b2, W3, b3 = init_params()
    beta1 = 0.9
    beta2 = 0.999
    m, v = [np.zeros_like(w) for w in [W1, W2, W3]], [np.zeros_like(w) for w in [W1, W2, W3]]
    mb, vb = [np.zeros_like(b) for b in [b1, b2, b3]], [np.zeros_like(b) for b in [b1, b2, b3]]
    for i in range(1, iterations):
        Z1, A1, Z2, A2, Z3, A3= forward_prop(W1, b1, W2, b2, W3, b3, X)
        db1,db2,db3,dW1,dW2,dW3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)
        W1, m[0], v[0] = update_parameters(W1, dW1, m[0], v[0], beta1, beta2, i, alpha)
        b1, mb[0], vb[0] = update_parameters(b1, db1, mb[0], vb[0], beta1, beta2, i, alpha)
        W2, m[1], v[1] = update_parameters(W2, dW2, m[1], v[1], beta1, beta2, i, alpha)
        b2, mb[1], vb[1] = update_parameters(b2, db2, mb[1], vb[1], beta1, beta2, i, alpha)
        W3, m[2], v[2] = update_parameters(W3, dW3, m[2], v[2], beta1, beta2, i, alpha)
        b3, mb[2], vb[2] = update_parameters(b3, db3, mb[2], vb[2], beta1, beta2, i, alpha)
        if i%5 == 0:
            predictions = predict(A3)
            print("iteratiion:", i)
            # acc = np.trunc(accuracy(predictions, Y))
            acc = int(accuracy(predictions, Y) *100)
            print(f'{acc}% accurate')
        
    return W1, b1, W2, b2, W3, b3



W1, b1, W2, b2, W3, b3= descent(X_train, Y_train, 0.0005, 20)

def make_prdc(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_prop(W1, b1,W2,b2,W3,b3,X)
    predictions = predict(A3)
    return predictions


# def test_prediction(index, W1, b1, W2, b2, W3, b3):
#     img = X_train[:,index,None]
#     prediction = make_prdc(img, W1, b1, W2, b2, W3, b3)
#     label = Y_train[index]
#     print("Prediction: ", prediction)
#     print("Label: ", label)

import pygame as pg
from scipy.ndimage import zoom

fps = 10
clock = pg.time.Clock()

pg.init()
screen = pg.display.set_mode((560, 560))
draw_surface = pg.Surface((560, 560))

running = True
drawing = False
array = [0] *784
array = np.reshape(array, (28,28))
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_t:
                draw_surface.fill((0, 0, 0))
        elif event.type == pg.MOUSEBUTTONDOWN:
            drawing = True
        elif event.type == pg.MOUSEBUTTONUP:
            drawing = False
            drawdata = pg.surfarray.array2d(draw_surface)
            drawdata = drawdata.T
            drawdata = zoom(drawdata,(28/drawdata.shape[0], 28/drawdata.shape[1]), order =2)

            drawdata = drawdata / 255.0  
            drawdata = drawdata.reshape(-1, 1) 
            drawdata = drawdata
            prediction = make_prdc(drawdata, W1, b1, W2, b2, W3, b3)
            print("Prediction: ", prediction)
            # plt.gray()
            # plt.imshow(drawdata.reshape(28, 28), interpolation='nearest')  # Reshape for displaying
            # plt.axis('off')  # Hide axes
            # plt.show(block=False)  # Show the image without blocking
            # plt.pause(1)  # Pause to view the image for 1 second
            # plt.close()  # Close the figure after viewing

    if drawing:
        pos = pg.mouse.get_pos()
        pg.draw.circle(draw_surface, (255, 255, 255), pos, 20)  

    screen.blit(draw_surface, (0, 0)) 
    pg.display.flip()

pg.quit()


#     img = img.reshape((28, 28)) * 255
#     plt.gray()
#     plt.imshow(img, interpolation='nearest')
#     plt.axis('on')  # Hide axis
#     plt.show(block=False)  # Show the image
#     plt.pause(2)
#     plt.close()


# def loop_pred():
#     i =random.randint(0,1000)
#     test_prediction(i, W1, b1, W2, b2, W3, b3)

# try:
#     while True:
#         loop_pred()
#         time.sleep(1)
# except KeyboardInterrupt:
#     plt.close()
#     pass



