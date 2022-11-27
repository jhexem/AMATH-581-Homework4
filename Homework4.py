import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.sparse
import scipy.optimize
import time

alpha = 2   #define parameters
L = 10
n = 128   #size of the D matrix is nxn

xvals = np.linspace(-L, L, n, endpoint=False)   #generate x points and t points
tvals = np.linspace(0, 2, 501)
trange = [0, 2]

dx = xvals[1] - xvals[0]   #calculate dx and dt
dt = tvals[1] - tvals[0]

CFL = (alpha * dt) / (dx ** 2)   #calculate the CFL number lambda

g14 = lambda z, CFL: 1 - ((5 * CFL) / 2) + ((8 * CFL) / 3) * np.cos(z) - (CFL / 6) * np.cos(2 * z)   #define functions for g and abs(g)
absg14 = lambda z, CFL: np.abs(g14(z, CFL))

A1 = np.abs(g14(1, CFL))

zvals = np.linspace(-np.pi, np.pi, 1000)
CFLvals = np.linspace(0, 3, 1000)
Z, Lambda = np.meshgrid(zvals, CFLvals)

'''solg14 = absg14(Z, Lambda)   #makes contour plot for z vs lambda
fig,ax = plt.subplots(1, 1,figsize=(7,7))
ax.contourf(Z, Lambda, solg14)
plt.xlabel('z')
plt.ylabel('lambda')
plt.show()   #'''

maxval14 = scipy.optimize.fminbound(lambda z: -absg14(z, CFL), -np.pi, np.pi)#find the maximum value of |g(z;CFL)|
maximum14 = absg14(maxval14, CFL)

A2 = maximum14

def Dmatrix(n):   #function that created the D  matrix for the 1,4 method
   main_diag = np.ones(n) * (-5/2)
   next_diag = np.ones(n) * (4/3)
   outer_diag = np.ones(n) * (-1/12)

   index_list = np.array([-(n-1), -(n-2), -2, -1, 0, 1, 2, n-2, n-1])
   data = np.array([next_diag, outer_diag, outer_diag, next_diag, main_diag, next_diag, outer_diag, outer_diag, next_diag])

   D = scipy.sparse.spdiags(data, index_list, n, n, format='csc')
   return D

D = Dmatrix(n)   #call D matrix function
A3 = D.todense()

def rhsfunc14(t, u, CFL, D):   #deifne the ODE
   return u + (CFL * (D @ u))

u0func = lambda x: 10 * np.cos((2 * np.pi * x) / L) + 30 * np.cos((8 * np.pi * x) / L)   #define the initial condition
u0 = u0func(xvals)

sol14 = scipy.integrate.solve_ivp(lambda t, u: rhsfunc14(t, u, CFL, D), trange, u0, t_eval=tvals)   #solve the ODE

A5 = np.array([sol14.y[:, -1]]).T

'''X, T = np.meshgrid(xvals, tvals)   #plot the surface of the solutions of the ODE
fig,ax = plt.subplots(subplot_kw = {"projection":"3d"},figsize=(7,7))
surf = ax.plot_surface(X,T,sol14.y.T,cmap='magma')
plt.xlabel('x')
plt.ylabel('time')
plt.show()   #'''

gCN = lambda z, CFL: (1 - CFL + CFL * np.cos(z)) / (1 + CFL - CFL * np.cos(z))
absgCN = lambda z, CFL: np.abs(gCN(z, CFL))

maxvalCN = scipy.optimize.fminbound(lambda z: -absgCN(z, CFL), -np.pi, np.pi)   #find the maximum value of |g(z;CFL)|
maximumCN = absgCN(maxvalCN, CFL)

A6 = maximumCN

'''solgCN = absgCN(Z, Lambda)   #makes contour plot for z vs lambda
fig,ax = plt.subplots(1, 1,figsize=(7,7))
ax.contourf(Z, Lambda, solgCN)
plt.xlabel('z')
plt.ylabel('lambda')
plt.show()   #'''

def Bmatrix(n, CFL):   #define function that creates the B matrix for the Crank Nicholson method
   main_diag = np.ones(n) * (1 + CFL)
   other_diag = np.ones(n) * (-CFL / 2)

   index_list = np.array([-(n-1), -1, 0, 1, n-1])
   data = np.array([other_diag, other_diag, main_diag, other_diag, other_diag])

   B = scipy.sparse.spdiags(data, index_list, n, n, format='csc')
   return B

def Cmatrix(n, CFL):   #define function that creates the C matrix for the Crank Nicholson method
   main_diag = np.ones(n) * (1 - CFL)
   other_diag = np.ones(n) * (CFL / 2)

   index_list = np.array([-(n-1), -1, 0, 1, n-1])
   data = np.array([other_diag, other_diag, main_diag, other_diag, other_diag])

   B = scipy.sparse.spdiags(data, index_list, n, n, format='csc')
   return B

B = Bmatrix(n, CFL)   #calls the function to get the B and C matrix
C = Cmatrix(n, CFL)

A7 = B.todense()
A8 = C.todense()

def RHSfuncLU(t, u, C, LUdecompB):
   u1 = LUdecompB.solve(C @ u)
   return u1

LUdecompB = scipy.sparse.linalg.splu(B)
solLU = scipy.integrate.solve_ivp(lambda t, u: RHSfuncLU(t, u, C, LUdecompB), trange, u0, t_eval=tvals)

A9 = np.array([solLU.y[:, -1]]).T

'''X, T = np.meshgrid(xvals, tvals)   #plot the surface of the solutions of the ODE
fig,ax = plt.subplots(subplot_kw = {"projection":"3d"},figsize=(7,7))
surf = ax.plot_surface(X,T,solLU.y.T,cmap='magma')
plt.xlabel('x')
plt.ylabel('time')
plt.show()   #'''

def RHSfuncBC(t, u, B, C):
   (u1, info) = scipy.sparse.linalg.bicgstab(B, C @ u)
   return u1

timestart = time.time()
solBC = scipy.integrate.solve_ivp(lambda t, u: RHSfuncBC(t, u, B, C), trange, u0, t_eval=tvals)
timeend = time.time()
elapsedtime = timeend - timestart

A10 = np.array([solBC.y[:, -1]]).T

X, T = np.meshgrid(xvals, tvals)   #plot the surface of the solutions of the ODE
fig,ax = plt.subplots(subplot_kw = {"projection":"3d"},figsize=(7,7))
surf = ax.plot_surface(X,T,solBC.y.T,cmap='magma')
plt.xlabel('x')
plt.ylabel('time')
plt.show()   #'''



'''My values for the maximum of |g(z;lambda)| are both 1 and I believe that they should be different. Am
I doing the maximization correctly?

My surface plots for all of my solutions are growing in time instead of dissipating and I am not sure why.
Is my CFL number correct? Is there an incorrect negative sign somewhere?'''