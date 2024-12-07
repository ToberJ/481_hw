import numpy as np
from scipy.fft import fft2, ifft2
from scipy.integrate import solve_ivp
from scipy.sparse import spdiags, diags
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.sparse.linalg import splu, bicgstab, gmres, spsolve
from scipy.linalg import lu, solve_triangular
import time
from matplotlib import animation


m = 64  # N value in x and y directions
n = m * m  # total size of matrix
dn = 20 / m

e0 = np.zeros(n)
e1 = np.ones(n)

e2 = e1.copy()
e4 = e0.copy()

for j in range(1, m + 1):
    e2[m * j - 1] = 0  # zero out every m-th value
    e4[m * j - 1] = 1  # set every m-th value to one

e3 = np.roll(e2, 1)
e5 = np.roll(e4, 1)

diagonals = [e1, e1, e5, e2, -4 * e1, e3, e4, e1, e1]
offsets = [-(n - m), -m, -m + 1, -1, 0, 1, m - 1, m, (n - m)]
matA = spdiags(diagonals, offsets, n, n).toarray()
A = matA / (dn ** 2)

adds = (1 / (2 * dn)) * np.ones(n)
subs = -adds
B = spdiags([adds, adds, subs, subs], [m, -(n - m), n - m, -m], n, n).toarray()

e0 = np.zeros(n)
e1 = np.ones(n)
e2 = e0.copy()
e3 = e1.copy()

for i in range(n):
    if i % m == 0:
        e0[i] = 1
        e1[i] = 0
    if (i + 1) % m == 0:
        e2[i] = 1
        e3[i] = 0

e2 = -e0
e3 = -e3
C = spdiags([e1, e3, e0], [1, -1, -m+1], n, n).toarray()

for i in range(m):
    row_index = i * m      # Start of each row
    col_index = (i + 1) * m - 1  # End of each row
    if row_index < n and col_index < n:
        C[row_index, col_index] = -1
C = C * (1 / (2 * dn))


n = 64
tp = [0, 4]
tspan = np.arange(tp[0], tp[1] + 0.5, 0.5)
nu = 0.001
Lx, Ly = 20, 20
nx, ny = 64, 64
N = nx * ny
t_span = [0, 4]
t_eval = np.arange(0, 4.5, 0.5)
x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
y = y2[:ny]
X, Y = np.meshgrid(x, y)
w = np.exp(-X**2 - Y**2 / 20).flatten()  
kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2
A[0 , 0] = 2 / (20/64) ** 2 



def rhs(t, omega):
    omega_matrix = omega.reshape(n, n)

    psi_hat = -fft2(omega_matrix) / K
    psi = np.real(ifft2(psi_hat))
    psi = psi.flatten()
    
    psi_x = B.dot(psi)
    psi_y = C.dot(psi)
    omega_x = B.dot(omega)
    omega_y = C.dot(omega)
    
    jacobian = psi_x * omega_y - psi_y * omega_x
    return -jacobian + nu * (A.dot(omega))


start_time = time.time()
print(f"t_span: {t_span}")
print(f"t_eval: {t_eval}")

solution = solve_ivp(rhs, t_span, w, t_eval=t_eval, method='RK45')

end_time = time.time() # Record the end time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

A1 = solution.y  
from scipy.sparse.linalg import splu, bicgstab, gmres, spsolve
from scipy.linalg import lu, solve_triangular, lu_factor, lu_solve
import time

def rhs_solver(t, omega_flat, solver_type="direct"):
    omega = omega_flat

    if solver_type == "direct":  # A\b
        psi = np.linalg.solve(A, omega).flatten() # Sparse direct solver


    elif solver_type == "lu":  # LU decomposition
        # P, L, U = lu(A) # LU decomposition
        # y = solve_triangular(L, np.dot(P, w), lower=True)  # Forward substitution
        LU, piv = lu_factor(A)
        psi = lu_solve((LU, piv), omega)  # Backward substitution


    elif solver_type == "bicgstab":  # BICGSTAB
        psi, info = bicgstab(A, omega, tol=1e-6)

    elif solver_type == "gmres":  # GMRES
        psi, info = gmres(A, omega, tol=1e-6)


    else:
        raise ValueError("Invalid solver type specified.")

    # psi = psi.reshape((n, n))  
    psi_x = B.dot(psi)
    psi_y = C.dot(psi)
    omega_x = B.dot(omega)
    omega_y = C.dot(omega)
    jacobian = psi_x * omega_y - psi_y * omega_x
    laplacian_omega = A.dot(omega)
    omega_t = -jacobian + nu * laplacian_omega
    return omega_t.flatten()

solvers = ["direct", "lu"]
results = {}

for solver in solvers:
    print(f"Using solver: {solver}")
    print(f"t_span: {t_span}")
    print(f"t_eval: {t_eval}")

    sol = solve_ivp(rhs_solver, t_span, w, args=(solver,), t_eval=t_eval, method='RK45')
    results[solver] = sol.y.T  # Store the result
    end_time = time.time() # Record the end time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

A2 = results["direct"]
A3 = results["lu"]
A2 = A2.T
A3 = A3.T

def initial_condition(X, Y, type='single'):
    if type == 'opposite_pair':
        return (np.exp(-((X - 2) ** 2 + Y ** 2) / 1) -
                np.exp(-((X + 2) ** 2 + Y ** 2) / 1))
    elif type == 'same_pair':
        return (np.exp(-((X - 2) ** 2 + Y ** 2) / 1) +
                np.exp(-((X + 2) ** 2 + Y ** 2) / 1))
    elif type == 'collision':
        return (np.exp(-((X - 3) ** 2 + (Y - 3) ** 2) / 1) -
                np.exp(-((X - 3) ** 2 + (Y + 3) ** 2) / 1) +
                np.exp(-((X + 3) ** 2 + (Y - 3) ** 2) / 1) -
                np.exp(-((X + 3) ** 2 + (Y + 3) ** 2) / 1))
    elif type == 'random':
        vortices = np.zeros_like(X)
        num_vortices = np.random.randint(10, 16)
        for _ in range(num_vortices):
            x0 = np.random.uniform(-Lx / 2, Lx / 2)
            y0 = np.random.uniform(-Ly / 2, Ly / 2)
            amp = np.random.uniform(-1, 1)
            ecc = np.random.uniform(1, 5)
            vortices += amp * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / ecc)
        return vortices
    else:
        # Default to single vortex
        return np.exp(-X ** 2 - Y ** 2 / 20)

# Function to display the animation
def display_animation(X, Y, omega_time_series, t_eval, ic_type):
    D = omega_time_series.reshape(len(t_eval), m, m)
    vmax = np.max(np.abs(D))
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(X, Y, D[0], shading='auto', cmap='RdBu', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)
    ax.set_title(f'Vorticity at t = {t_eval[0]:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    def update(frame):
        ax.clear()
        im = ax.pcolormesh(X, Y, D[frame], shading='auto', cmap='RdBu', vmin=vmin, vmax=vmax)
        ax.set_title(f'Vorticity at t = {t_eval[frame]:.2f}, IC: {ic_type}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        return [im]

    anim = animation.FuncAnimation(fig, update, frames=len(t_eval), interval=200, blit=False)

    plt.show()  # Display the animation window

# Run simulations for each initial condition
ic_types = ['opposite_pair', 'same_pair', 'collision', 'random']
for ic_type in ic_types:
    print(f"\nSimulating for initial condition: {ic_type}")
    omega0 = initial_condition(X, Y, type=ic_type).flatten()
    start_time = time.time()
    solution = solve_ivp(rhs, t_span, omega0, t_eval=t_eval, method='RK45')
    elapsed_time = time.time() - start_time
    print(f"Simulation completed in {elapsed_time:.2f} seconds")

    # Extract the solution
    omega_time_series = solution.y.T  # Shape: (time_steps, n)

    # Display the animation
    display_animation(X, Y, omega_time_series, t_eval, ic_type)
def initial_condition(X, Y, type='single'):
    if type == 'opposite_pair':
        return (np.exp(-((X - 2) ** 2 + Y ** 2) / 1) -
                np.exp(-((X + 2) ** 2 + Y ** 2) / 1))
    elif type == 'same_pair':
        return (np.exp(-((X - 2) ** 2 + Y ** 2) / 1) +
                np.exp(-((X + 2) ** 2 + Y ** 2) / 1))
    elif type == 'collision':
        return (np.exp(-((X - 3) ** 2 + (Y - 3) ** 2) / 1) -
                np.exp(-((X - 3) ** 2 + (Y + 3) ** 2) / 1) +
                np.exp(-((X + 3) ** 2 + (Y - 3) ** 2) / 1) -
                np.exp(-((X + 3) ** 2 + (Y + 3) ** 2) / 1))
    elif type == 'random':
        vortices = np.zeros_like(X)
        num_vortices = np.random.randint(10, 16)
        for _ in range(num_vortices):
            x0 = np.random.uniform(-Lx / 2, Lx / 2)
            y0 = np.random.uniform(-Ly / 2, Ly / 2)
            amp = np.random.uniform(-1, 1)
            ecc = np.random.uniform(1, 5)
            vortices += amp * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / ecc)
        return vortices
    else:
        # Default to single vortex
        return np.exp(-X ** 2 - Y ** 2 / 20)

# Function to display the animation
def display_animation(X, Y, omega_time_series, t_eval, ic_type):
    D = omega_time_series.reshape(len(t_eval), m, m)
    vmax = np.max(np.abs(D))
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(X, Y, D[0], shading='auto', cmap='RdBu', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)
    ax.set_title(f'Vorticity at t = {t_eval[0]:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    def update(frame):
        ax.clear()
        im = ax.pcolormesh(X, Y, D[frame], shading='auto', cmap='RdBu', vmin=vmin, vmax=vmax)
        ax.set_title(f'Vorticity at t = {t_eval[frame]:.2f}, IC: {ic_type}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        return [im]

    anim = animation.FuncAnimation(fig, update, frames=len(t_eval), interval=200, blit=False)

    plt.show()  # Display the animation window

# Run simulations for each initial condition
ic_types = ['opposite_pair', 'same_pair', 'collision', 'random']
for ic_type in ic_types:
    print(f"\nSimulating for initial condition: {ic_type}")
    omega0 = initial_condition(X, Y, type=ic_type).flatten()
    start_time = time.time()
    solution = solve_ivp(rhs, t_span, omega0, t_eval=t_eval, method='RK45')
    elapsed_time = time.time() - start_time
    print(f"Simulation completed in {elapsed_time:.2f} seconds")

    # Extract the solution
    omega_time_series = solution.y.T  # Shape: (time_steps, n)

    # Display the animation
display_animation(X, Y, omega_time_series, t_eval, ic_type)