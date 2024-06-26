import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.sparse
import scipy.optimize
import time

alpha = 2   #define parameters
L = 10
n = 128   #size of the matrices is nxn

xvals = np.linspace(-L, L, n, endpoint=False)   #generate x points and t points
trange = [0, 2]
tvals = np.linspace(trange[0], trange[1], 501)

dx = xvals[1] - xvals[0]   #calculate dx and dt
dt = tvals[1] - tvals[0]

CFL = (alpha * dt) / (dx ** 2)   #calculate the CFL number lambda

g14 = lambda z, CFL: 1 - ((5 * CFL) / 2) + ((8 * CFL) / 3) * np.cos(z) - (CFL / 6) * np.cos(2 * z)   #define functions for g for the 14 method
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

maxval14 = scipy.optimize.fminbound(lambda z: -absg14(z, CFL), -np.pi, np.pi)   #find the maximum value of |g(z;CFL)|
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

u0func = lambda x: 10 * np.cos((2 * np.pi * x) / L) + 30 * np.cos((8 * np.pi * x) / L)   #define the initial condition with n=128
u0 = u0func(xvals)

def solve14(u0, D, CFL, tvals, xvals):
   u = u0
   sollist = np.zeros((len(tvals), len(xvals)))
   sollist[0, :] = u0
   for i in range(len(tvals)-1):
      u = u + CFL * (D @ u)
      sollist[i+1, :] = u
   return sollist

sol14 = solve14(u0, D, CFL, tvals, xvals)

A5 = np.array([sol14[-1, :]]).T

'''X, T = np.meshgrid(xvals, tvals)   #plot the surface of the solutions of the ODE
fig,ax = plt.subplots(subplot_kw = {"projection":"3d"},figsize=(7,7))
surf = ax.plot_surface(X,T,sol14,cmap='magma')
plt.xlabel('x')
plt.ylabel('time')
plt.show()   #'''

gCN = lambda z, CFL: (1 - CFL + CFL * np.cos(z)) / (1 + CFL - CFL * np.cos(z))   #defines the g function for the CN method
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

A7 = B.todense()   #correctly formats the B and C matrices to submit
A8 = C.todense()

def solveLU(u0, LUdecompB, C, tvals, xvals):
   u = u0
   sollist = np.zeros((len(tvals), len(xvals)))
   sollist[0, :] = u0
   for i in range(len(tvals)-1):
      u = LUdecompB.solve(C @ u)
      sollist[i+1, :] = u
   return sollist

LUdecompB = scipy.sparse.linalg.splu(B)
solLU = solveLU(u0, LUdecompB, C, tvals, xvals)

A9 = np.array([solLU[-1, :]]).T

'''X, T = np.meshgrid(xvals, tvals)   #plot the surface of the solutions of the ODE
fig,ax = plt.subplots(subplot_kw = {"projection":"3d"},figsize=(7,7))
surf = ax.plot_surface(X,T,solLU,cmap='magma')
plt.xlabel('x')
plt.ylabel('time')
plt.show()   #'''

def solveBC(u0, B, C, tvals, xvals):
   u = u0
   sollist = np.zeros((len(tvals), len(xvals)))
   sollist[0, :] = u0
   for i in range(len(tvals)-1):
      (u, info) = scipy.sparse.linalg.bicgstab(B, C @ u)
      sollist[i+1, :] = u
   return sollist

timestart = time.time()
solBC = solveBC(u0, B, C, tvals, xvals)
timeend = time.time()
elapsedtime = timeend - timestart   #gives the total runtime of the solve

A10 = np.array([solBC[-1, :]]).T

'''X, T = np.meshgrid(xvals, tvals)   #plot the surface of the solutions of the ODE
fig,ax = plt.subplots(subplot_kw = {"projection":"3d"},figsize=(7,7))
surf = ax.plot_surface(X,T,solBC,cmap='magma')
plt.xlabel('x')
plt.ylabel('time')
plt.show()   #'''

file128 = open('exact_128.csv')   #get data from the csv file to obtain the exact solution with 128 points
exact128 = np.zeros(128)
i = 0
for line in file128:
   exact128[i] = float(line)
   i += 1
file128.close()

file256 = open('exact_256.csv')   #get data from the csv file to obtain the exact solution with 256 points
exact256 = np.zeros(256)
i = 0
for line in file256:
   exact256[i] = float(line)
   i += 1
file256.close()

file512 = open('exact_512.csv')   #get data from the csv file to obtain the exact solution with 512 points
exact512 = np.zeros(512)
i = 0
for line in file512:
   exact512[i] = float(line)
   i += 1
file512.close()

file1024 = open('exact_1024.csv')   #get data from the csv file to obtain the exact solution with 1024 points
exact1024 = np.zeros(1024)
i = 0
for line in file1024:
   exact1024[i] = float(line)
   i += 1
file1024.close()

A11 = np.linalg.norm(exact128 - np.ndarray.flatten(A5))   #calculate the norm differences for the 14 and CN methods
A12 = np.linalg.norm(exact128 - np.ndarray.flatten(A9))

n = 256
xvals_256 = np.linspace(-L, L, n, endpoint=False)

dx = xvals_256[1] - xvals_256[0]   #calculate dx and dt
dt = (CFL * (dx ** 2)) / alpha
tvals_256 = np.arange(0, 2 + dt, dt)
CFL = (alpha * dt) / (dx ** 2)   #calculate the CFL number lambda'''

D256 = Dmatrix(n)   #call functions that create the D, B, and C matrices of size 256x256
B256 = Bmatrix(n, CFL)
C256 = Cmatrix(n, CFL)

u0_256 = u0func(xvals_256)   #get initial condition with 256 values

sol14_256 = solve14(u0_256, D256, CFL, tvals_256, xvals_256)

LUdecompB256 = scipy.sparse.linalg.splu(B256)
solLU_256 = solveLU(u0_256, LUdecompB256, C256, tvals_256, xvals_256)

A13 = np.linalg.norm(exact256 - sol14_256[-1, :])   #calculate the norm differences for the 14 and CN methods for n=256
A14 = np.linalg.norm(exact256 - solLU_256[-1, :])

#------Presentation Problem------#

from matplotlib import cm

x = np.linspace(-L, L, 20 * L + 1)
t = np.array([np.linspace(0, 5, 100)])

fig1 = plt.figure()
ax1 = plt.axes(projection = '3d')
X, T = np.meshgrid(xvals_256, tvals_256)

ax1.plot_surface(X, T, solLU_256, cmap = cm.inferno)
ax1.set_xlabel('x axis')
ax1.set_ylabel('t axis')
ax1.set_zlabel('$u(x, t)$')
ax1.set_title('Solution to the Heat Equation Using the Crank-Nicolson Method')
ax1.view_init(elev=30, azim=135)
plt.savefig('Homework4Plot2.png')
plt.show()

#------Presentation Problem------#

'''times = np.arange(0, 2 + (dt * 5), dt* 5)

def frame(t):
   time = np.where(times == t)
   plt.plot(xvals_256, np.ndarray.flatten(solLU_256[time, :]))
   plt.ylim([-40, 40])
   plt.title('Heat Equation Solution'.format(t))
   plt.savefig('image{0}.png'.format(t), transparent=False, facecolor='white')
   plt.close()

import imageio

for t in times:
   frame(t)
   
frames = []
for t in times:
   pic = imageio.imread('image{0}.png'.format(t))
   frames.append(pic)
   
imageio.mimsave('Homework4.gif', # output gif
                frames,          # array of input frames
                fps = 30)         # optional: frames per second'''