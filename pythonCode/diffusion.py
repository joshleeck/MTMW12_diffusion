#!/usr/bin/python

# Outer code for setting up the diffusion problem on a uniform
# grid and calling the function to perform the diffusion and plot.

from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np

# read in all the linear advection schemes, initial conditions and other
# code associated with this application
execfile("diffusionSchemes.py")
execfile("diagnostics.py")
execfile("initialConditions.py")

def main():
    '''
    Diffuse a squareWave between squareWaveMin and squareWaveMax on a domain
    between x = xmin and x = xmax split over nx spatial steps with diffusion
    coefficient K, time step dt for nt time steps
    '''
    
    #Parameters
    xmin = 0
    xmax = 1
    nx = 41
    nt = 500
    dt = 0.1
    K = 1e-3
    squareWaveMin = 0.4
    squareWaveMax = 0.6
    
    #Derived Parameters
    dx = (xmax - xmin)/(nx - 1)
    d = K*dt/dx**2 #Non-dimensional diffusion coefficent
    print("non-dimensional diffusion coefficient = ", d)
    print("dx = ", dx, " dt = ", dt, " nt = ", nt)
    print("end time = ", nt*dt)
    
    #spatial points for plotting and for defining initial conditions
    x = np.zeros(nx)
    for j in xrange(nx):
        x[j] = xmin + j*dx
    print("x=", x)
    
    #Initial conditions
    phiOld = squareWave(x, squareWaveMin, squareWaveMax)
    #Analytic solution (of square wave profile in an infinite domain)
    phiAnalytic = analyticErf(x, K*dt*nt, squareWaveMin, squareWaveMax)
    
    #Diffusion using FTCS and BTCS
    phiFTCS = FTCS(phiOld.copy(), d, nt)
    phiBTCS = BTCS(phiOld.copy(), d, nt)
    
    #Calculate and print out error norms
    print("FTCS L2 error norm = ", L2ErrorNorm(phiFTCS, phiAnalytic))
    print("BTCS L2 error norm = ", L2ErrorNorm(phiBTCS, phiAnalytic))
    
    #Plot the solutions
    font = {'size':16}
    plt.rc('font', **font)
    plt.figure(1)
    plt.clf()
    plt.ion()
    plt.plot(x, phiOld, label='Initial', color='black')
    plt.plot(x, phiAnalytic, label='Analytic', color='black', linestyle='--', 
             linewidth=2)
    plt.plot(x, phiFTCS, label='FTCS', color='blue')
    plt.plot(x, phiBTCS, label='BTCS', color='red')
    plt.axhline(0, linestyle=':', color='black')
    plt.ylim([0, 1])
    #plt.legend(bbox_to_anchor=(1.1, 1))
    plt.legend(loc='upper left', prop={'size': 14})
    plt.xlabel('$x$')
    plt.tight_layout()
    #plt.savefig("C:\Users\Joshua\Desktop\FTCS_BTCS.png", 
    #            format='png', dpi=600)
    
    plt.figure(2)
    plt.plot(x, phiFTCS-phiAnalytic, 'r', label='FTCS error')
    plt.plot(x, phiBTCS-phiAnalytic, 'b', label='BTCS error')
    plt.legend(loc='upper left', prop={'size': 14})
    plt.xlabel('$x$')
    plt.tight_layout()
    #plt.savefig("C:\Users\Joshua\Desktop\FTCS_BTCS_error.png", 
    #            format='png', dpi=600)
    
main()