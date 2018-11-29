from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec


def GGG_plots(X,Y,B_mag,u,v,p,t):

    fig = plt.figure(figsize=(7, 9))
    gs = gridspec.GridSpec(nrows=2, ncols=2)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.contourf(X,Y,B_mag,cmap=cm.Blues)
    fig.colorbar(label = '$|B|$')
    ax0.quiver(X,Y,u[t],v[t])
    ax0.xlabel('$x$')
    ax0.ylabel('$y$')
    ax0.title('$t$ = ' + str(t))

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.contourf(X,Y,B_mag,cmap=cm.Blues)
    fig.colorbar(label = '$|B|$')
    ax1.streamplot(X,Y,u[t],v[t],color=u[t],cmap='autumn')
    ax1.xlabel('$x$')
    ax1.ylabel('$y$')
    ax1.title('$t$ = ' + str(t))

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.contourf(X,Y,p[t],cmap=cm.seismic)
    fig.colorbar(label = '$p$')
    ax2.xlabel('$x$')
    ax2.ylabel('$y$')
    ax2.title('$t$ = ' + str(t))

    plt.show()
