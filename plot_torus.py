"""
@author: Mindy Ross
python version 3.7.4
numpy version: 1.19.2
"""
# plot torus

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

theta = np.linspace(0, 2.*np.pi, 30)
phi = np.linspace(0, 2.*np.pi, 24)
theta, phi = np.meshgrid(theta, phi)
c, a = 1,0.7
x = (c + a*np.cos(theta)) * np.cos(phi)
y = (c + a*np.cos(theta)) * np.sin(phi)
z = a * np.sin(theta)

fig = plt.figure(figsize = (50,40))
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_zlim(-3,3)
ax1.plot_surface(x, y, z,
                 color='#E6E6FA', edgecolors='#CBC3E3')
ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax1.grid(False)
ax1.view_init(10, 0)
plt.show()