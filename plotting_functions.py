from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def plot_sphere(disp = False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    u, v = np.mgrid[0:2*np.pi:150j, 0:np.pi:150j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    # alpha controls opacity
    ax.plot_wireframe(x, y, z, linewidth=0.2, antialiased=True)

    if disp:
        plt.show()
    
    return ax
    

def plot_points(data, disp = False):
    fig_obj = plot_sphere()
    fig_obj.scatter(data[:, 0], data[:, 1], data[:, 2], color="r",s=20)
    if disp:
       plt.show()

    return fig_obj
    
#def plot_contours():   


    