from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec


def GGG_plots(X,Y,u,v,p,t):

    plt.figure(figsize=(12, 8))
    #gs = gridspec.GridSpec(nrows=2, ncols=2)

    plt.subplot(2,2,1)
    #ax0.contourf(X,Y,B_mag,cmap=cm.Blues)
    #fig.colorbar(label = '$|B|$')
    plt.quiver(X,Y,u[t],v[t])
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Velocity plot of fluid elements at $t$ = ' + str(t/1500))

    plt.subplot(2,2,2)
    #ax1.contourf(X,Y,B_mag,cmap=cm.Blues)
    #fig.colorbar(label = '$|B|$')
    plt.streamplot(X,Y,u[t],v[t],color=u[t],cmap='autumn')
    plt.xlabel('$x$')
    #plt.ylabel('$y$')
    plt.title('Plot of streamlines at $t$ = ' + str(t/1500))


    plt.subplot(2,2,3)
    plt.contourf(X,Y,p[t],cmap=cm.seismic)
    plt.colorbar(label = '$p$')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Contour of pressure at $t$ = ' + str(t/1500))

    fn = 'Double_vortex_t' + str(t) + '.png'

    #flt.savefig(fn)

    plt.show()
