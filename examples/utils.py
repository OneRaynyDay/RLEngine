import numpy as np
import matplotlib.pyplot as plt

def render_cliffwalking(q, figname):
    X = 12
    Y = 4
    Fx = np.zeros((Y, X))
    Fy = np.zeros((Y, X))
    for y in range(Y):
        for x in range(X):
            amax = np.argmax(q[x+y*12])
            if amax == 0: # UP
                Fy[y, x] = -1
            elif amax == 1: # RIGHT
                Fx[y, x] = 1
            elif amax == 2: # DOWN
                Fy[y, x] = 1
            elif amax == 3: # LEFT
                Fx[y, x] = -1
    plt.quiver(Fx,Fy)
    plt.savefig(figname)
