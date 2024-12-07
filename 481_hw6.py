import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fft2, ifft2
from scipy.linalg import kron
from scipy.integrate import odeint
from scipy.integrate import solve_ivp


# Hyperbolic functions
def sech(x): return 1 / np.cosh(x)
def tanh(x): return np.sinh(x) / np.cosh(x)

# Parameters
N, beta, D1, D2 = 64, 1, 0.1, 0.1
Nx, Ny = 64, 64
N1 = Nx * Ny
Lx, Ly, tspan, t_eval = 20, 20, (0, 4), np.arange(0, 4.5, 0.5)
x = np.linspace(-Lx/2, Lx/2, N+1)[:N]
y = np.linspace(-Ly/2, Ly/2, N+1)[:N]
X, Y = np.meshgrid(x, y)
kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, N/2), np.arange(-N/2, 0)))
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, N/2), np.arange(-N/2, 0)))
kx[0], ky[0] = 1e-6, 1e-6
KX, KY = np.meshgrid(kx, ky)
lap_fft = KX**2 + KY**2

# Initial conditions
m = 1
U = tanh(np.sqrt(X**2 + Y**2)) * np.cos(m * np.angle(X + Y*1j) - np.sqrt(X**2 + Y**2))
V = tanh(np.sqrt(X**2 + Y**2)) * np.sin(m * np.angle(X + Y*1j) - np.sqrt(X**2 + Y**2))
ut, vt = fft2(U), fft2(V)
UV0 = np.hstack([ut.reshape(N1),vt.reshape(N1)])

# Reaction-diffusion equations (periodic)
def RD_periodic(t,uvt):
	utc=uvt[0:N1]
	vtc=uvt[N1:]
	ut=utc.reshape(Nx,Ny)
	vt=vtc.reshape(Nx,Ny)
	u,v=ifft2(ut),ifft2(vt)
	A1=u*u+v*v
	Lambda=1-A1
	Omega=-beta*A1
	rhsu=(-D1*lap_fft*ut+fft2(Lambda*u-Omega*v)).reshape(N1)
	rhsv=(-D2*lap_fft*vt+fft2(Omega*u+Lambda*v)).reshape(N1)
	rhs=np.hstack([rhsu,rhsv])
	return rhs

# Solve periodic case
sol_per = solve_ivp(RD_periodic, tspan, UV0, t_eval=t_eval, method="RK45")
A1 = np.real(sol_per.y)
print(A1)
print(np.shape(A1))

# Chebyshev differentiation matrix
def cheb(n):
    if n == 0: return 0., 1.
    c = np.hstack(([2.], np.ones(n-1), [2.])) * (-1)**np.arange(n+1)
    x = np.cos(np.pi * np.arange(n+1) / n)
    X = np.tile(x, (n+1, 1))
    dX = X - X.T
    D = np.outer(c, 1/c) / (dX + np.eye(n+1))
    D -= np.diag(D.sum(axis=1))
    return D, x

# Chebyshev parameters
Nc = 30
D, x = cheb(Nc)
D[0, :], D[-1, :] = 0, 0
Dxx = D @ D / 100
Xc, Yc = np.meshgrid(x, x)
Xc, Yc = 10 * Xc, 10 * Yc
Nc2 = (Nc + 1)**2
U = tanh(np.sqrt(Xc**2 + Yc**2)) * np.cos(m * np.angle(Xc + Yc*1j) - np.sqrt(Xc**2 + Yc**2))
V = tanh(np.sqrt(Xc**2 + Yc**2)) * np.sin(m * np.angle(Xc + Yc*1j) - np.sqrt(Xc**2 + Yc**2))
UV0 = np.hstack([U.flatten(), V.flatten()])
L = kron(np.eye(len(Dxx)), Dxx) + kron(Dxx, np.eye(len(Dxx)))

# Reaction-diffusion equations (no-flux)
def RD_noflux(t, UV):
    u, v = UV[:Nc2], UV[Nc2:]
    A = u**2 + v**2
    lam, omg = 1 - A, -beta * A
    rhs_u = D1 * (L @ u) + lam * u - omg * v
    rhs_v = D2 * (L @ v) + omg * u + lam * v
    return np.hstack([rhs_u, rhs_v])

# Solve no-flux case
sol_nf = solve_ivp(RD_noflux, tspan, UV0, t_eval=t_eval)
A2 = sol_nf.y
print(A2, A2.shape)
