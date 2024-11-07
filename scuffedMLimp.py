import numpy as np
import pandas as pd
import scuffedML

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

W1 = np.loadtxt('ModelData/W1.csv', delimiter=',')
b1 = np.loadtxt('ModelData/b1.csv', delimiter=',')
W2 = np.loadtxt('ModelData/W2.csv', delimiter=',')
b2 = np.loadtxt('ModelData/b2.csv', delimiter=',')

b1 = b1.reshape(-1, 1)
b2 = b2.reshape(-1, 1)

def make_prdc(X, W1, b1, W2, b2):
    _, _, _, A2 = scuffedML.forward_prop(W1, b1,W2,b2,X)
    predictions = scuffedML.predict(A2)
    return predictions

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
            drawdata = zoom(drawdata,(28/drawdata.shape[0], 28/drawdata.shape[1]), order =1)

            drawdata = drawdata / 255.0  
            drawdata = drawdata.reshape(-1, 1) 
            drawdata = drawdata
            prediction = make_prdc(drawdata, W1, b1, W2, b2)
            print("Prediction: ", prediction)
            # plt.gray()
            # plt.imshow(drawdata.reshape(28, 28), interpolation='nearest')  # Reshape for displaying
            # plt.axis('off')  # Hide axes
            # plt.show(block=False)  # Show the image without blocking
            # plt.pause(1)  # Pause to view the image for 1 second
            # plt.close()  # Close the figure after viewing

    if drawing:
        pos = pg.mouse.get_pos()
        pg.draw.circle(draw_surface, (255, 255, 255), pos, 10)  

    screen.blit(draw_surface, (0, 0)) 
    pg.display.flip()

pg.quit()