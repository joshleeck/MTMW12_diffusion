#!/usr/bin/python

# Outer code for setting up the diffusion problem on a uniform
# grid and calling the function to perform the diffusion and plot.

from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
from math import log

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
    nt = 100
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
    #phiOld = naivesquareWave(x, squareWaveMin, squareWaveMax)
    phiOld = squareWave(x, squareWaveMin, squareWaveMax)
    #Analytic solution (of square wave profile in an infinite domain)
    phiAnalytic = analyticErf(x, K*dt*nt, squareWaveMin, squareWaveMax)
    
    #Diffusion using FTCS and BTCS
    phiFTCS = FTCS(phiOld.copy(), d, nt)
    phiBTCS = BTCS(phiOld.copy(), d, nt)
    
    #Calculate and print out error norms
    print("FTCS L2 error norm = ", L2ErrorNorm(phiFTCS, phiAnalytic))
    print("BTCS L2 error norm = ", L2ErrorNorm(phiBTCS, phiAnalytic))
    
    #Classify stability of BTCS/FTCS solution using error < 1 as a threshold
    if (L2ErrorNorm(phiFTCS, phiAnalytic) < 1):
        print("FTCS numerical solution is stable")
    else:
        print("FTCS numerical solution is unstable")
    if (L2ErrorNorm(phiBTCS, phiAnalytic) < 1):
        print("BTCS numerical solution is stable")
    else:
        print("BTCS numerical solution is unstable")    
        
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
    #plt.savefig("C:\Users\Joshua\Desktop\initial_conditions_square.pdf")
    
    plt.figure(2)
    plt.plot(x, phiFTCS-phiAnalytic, 'b', label='FTCS error')
    plt.plot(x, phiBTCS-phiAnalytic, 'r', label='BTCS error')
    plt.legend(loc='upper left', prop={'size': 14})
    plt.xlabel('$x$')
    plt.tight_layout()
    #plt.savefig("C:\Users\Joshua\Desktop\FTCS_unstable_error.pdf")

    
def FTCS_qn5(nx, dt):
    """
    This function performs the FTCS scheme without plotting.
    parameters nx: number of points in space
               dt: timestep
    """
    #Parameters
    xmin = 0
    xmax = 1
    nt = 4/dt     #keep end time fixed at 4 seconds
    K = 1e-3
    squareWaveMin = 0.4
    squareWaveMax = 0.6
    
    #Derived Parameters
    dx = (xmax - xmin)/(nx - 1)
    d = K*dt/dx**2 #Non-dimensional diffusion coefficent
    
    #spatial points for plotting and for defining initial conditions
    x = np.zeros(nx)
    for j in xrange(nx):
        x[j] = xmin + j*dx
    
    #Initial conditions
    phiOld = squareWave(x, squareWaveMin, squareWaveMax)
    #Analytic solution (of square wave profile in an infinite domain)
    phiAnalytic = analyticErf(x, K*dt*nt, squareWaveMin, squareWaveMax)
    
    #Diffusion using FTCS
    phiFTCS = FTCS(phiOld.copy(), d, nt)

    return phiFTCS, phiAnalytic, dx, d    
    
def calc_order_of_convergence():
    """
    This function calculates the order of convergence of the FTCS scheme.
    """
    #calculate values for arrays
    eps1 = L2ErrorNorm(FTCS_qn5(51, 0.2)[0], FTCS_qn5(51, 0.2)[1])
    eps2 = L2ErrorNorm(FTCS_qn5(201, 0.0125)[0], FTCS_qn5(201, 0.125)[1])
    eps3 = L2ErrorNorm(FTCS_qn5(801, 0.00078125)[0], \
                       FTCS_qn5(801, 0.00078125)[1])
    
    dx1  = FTCS_qn5(51, 0.2)[2]
    dx2  = FTCS_qn5(201, 0.0125)[2]
    dx3  = FTCS_qn5(801, 0.00078125)[2]
    
    #check that d is fixed
    d1   = FTCS_qn5(51, 0.2)[3]
    d2   = FTCS_qn5(201, 0.0125)[3]
    d3   = FTCS_qn5(801, 0.00078125)[3]

    print ("dx are", dx1, dx2, dx3)
    print ("d  are", d1, d2, d3)
    
    #initialise arrays
    eps_array = [abs(eps1), abs(eps2), abs(eps3)]
    dx_array  = [dx1, dx2, dx3]

    #formula for calculating order of convergence 
    n1 = (log(abs(eps2)) - log(abs(eps1)))/        \
        (log(dx2) - log(dx1))
        
    n2 = (log(abs(eps3)) - log(abs(eps2)))/        \
        (log(dx3) - log(dx2))
    
    n = (n1 + n2)/2

    print ("Order of convergence is", n)
    
    #plot order of convergence
    plt.plot(dx_array,eps_array)
    plt.ylabel("Normalised RMS Error")
    plt.yscale('log')
    plt.ylim(1e-5,1e-1)
    plt.xscale('log')
    plt.tight_layout()
    plt.xlabel(r'$\Delta$x (m)')
    #plt.savefig("C:\Users\Joshua\Desktop\FTCS_order_of_convergence4.pdf")
    plt.show()

#calc_order_of_convergence()    
main()