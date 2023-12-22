import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def KF_prediction(mu_prev, sigma_prev):
    mu_bar = A_t.dot(mu_prev)
    sigma_bar = A_t.dot(sigma_prev).dot(A_t.transpose()) + R_t
    mu = mu_bar
    sigma = sigma_bar
    return mu, sigma


A_t = np.array([[1, 1], [0, 1]])
G = np.array([[0.5], [1]])
R_t = G.dot(G.transpose())
mu_0 = np.array([[0], [0]])
sigma_0 = np.array([[0, 0], [0, 0]])

for t in range(5):
    if t == 0:
        mu_t = mu_0
        sigma_t = sigma_0
    else:
        mu_t, sigma_t = KF_prediction(mu_t, sigma_t)
        print('\u03BC' + chr(8320 + t) + ' = ')
        print(mu_t)
        print('\u03A3' + chr(8320 + t) + ' = ')
        print(sigma_t)
        print()
        a = sigma_t[0][0]
        b = sigma_t[0][1]
        c = sigma_t[1][1]
        w = (a+c)/2 + math.sqrt( ((a-c)/2)**2 + b**2 )
        h = (a+c)/2 - math.sqrt( ((a-c)/2)**2 + b**2 )
        t = math.atan2(w-a, b)
        ellipse = Ellipse(xy=(0, 0), width=math.sqrt(w), height=math.sqrt(h), angle=math.degrees(t), fc='none', ec='black')
        ax = plt.subplot(111)
        ax.add_patch(ellipse)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.axhline(y=0, color='k', lw=0.5)
        ax.axvline(x=0, color='k', lw=0.5)
        plt.show()
