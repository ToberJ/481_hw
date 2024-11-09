import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt 
from scipy.integrate import simpson
from scipy.sparse.linalg import eigs
def shoot2(x,y, beta):
    return [y[1], (x**2 -beta) * y[0]]
tol = 1e-4 
col = ['r', 'b', 'g', 'c', 'm', 'k'] 
L = 4 
xshoot = np.arange(-L, L + 0.1, 0.1) 
beta_start = 1
A1 = [] 
A2 = [] 
for modes in range(1, 6): 
    beta = beta_start 
    dbeta = 1
    for _ in range(1000): 
        y0 = [1, np.sqrt(L**2 - beta)]
        sol = solve_ivp(shoot2, [xshoot[0],xshoot[-1]],y0, args=(beta,),t_eval=xshoot)
        ys=sol.y.T
        if abs(ys[-1, 1] + np.sqrt(L**2 -beta) * ys[-1, 0]) < tol: 
            A2.append(beta) 
            break

        if ((-1) ** (modes + 1) * (ys[-1,1]+ np.sqrt((L**2 -beta)) * ys[-1, 0])) > 0:
            beta += dbeta
        else:
            beta -= dbeta
            dbeta /= 2
    beta_start = beta + 0.1 
    norm = np.trapz(ys[:, 0] * ys[:,0], x=xshoot) 
    normalized_eigenfunction = abs(ys[:, 0] / np.sqrt(norm))
    A1.append(normalized_eigenfunction)
    plt.plot(xshoot, normalized_eigenfunction, col[modes - 1], label=f'Mode {modes}')
plt.show() 
A1 = np.transpose(A1)
print(np.shape(A1))
print(np.shape(A2))
print(A1)
print(A2) 


from scipy.linalg import eigh
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigs

L=4
x=np.arange(-L,L+0.1,0.1)
n=len(x)
dx=x[1]-x[0]
H = np.zeros((n, n))

for i in range(1, n - 1):
    H[i, i] = -2 - (x[i]**2) * dx**2  
    H[i, i - 1] = 1  
    H[i, i + 1] = 1  

H[0, 0] = -2 -  (x[0]**2) * dx**2 + 4/3
H[0, 1] = 2/3  

H[-1, -1] = -2 -  (x[-1]**2) * dx**2 + 4/3  
H[-1, -2] = 2/3 

A=np.zeros((n-2,n-2))
for j in range(n-2):
    A[j,j]=-2-(dx**2)*x[j+1]**2
    if j < n-3:
        A[j+1,j]=1
        A[j,j+1]=1
A[0,0]+=4/3
A[0,1]+=-1/3
A[-1,-1]+=4/3
A[-1,-2]+=-1/3
eigval,eigfunc=eigs(-A,k=5,which='SM')
V2=np.vstack([4/3*eigfunc[0,:]-1/3*eigfunc[1,:],eigfunc,4/3*eigfunc[-1,:]-1/3*eigfunc[-2,:]])
phi_sol=np.zeros((n,5))
beta_sol=np.zeros(5)
for j in range(5):
    norm = np.trapz(V2[:, j] ** 2, x=x)
    phi_sol[:,j]=abs(V2[:,j]/np.sqrt(norm))
beta_sol = abs(np.sqrt(eigval[:5]**2))/dx**2

colors = ['r', 'b', 'g', 'c', 'm']  
for i in range(5):
    plt.plot(x, phi_sol[:, i], colors[i], label=f'Mode {i+1}')
plt.title("Normalized Eigenfunctions")
plt.xlabel("x")
plt.ylabel("φ(x)")
plt.legend()
plt.grid()
plt.show()
A4=np.array(beta_sol)
A3=np.array(phi_sol)
print(A4)




import numpy as np
from scipy.integrate import solve_ivp

def shoot(x, phi, gamma, epsilon):
    return [phi[1], (gamma * phi[0]**2 + x**2 - epsilon) * phi[0]]

L = 2
tol = 1e-4
gammas = [0.05, -0.05]
xp = [-L, L]
xshoot = np.linspace(xp[0], xp[1], 41)
A5, A6 = [], []  
A7, A8 = [], [] 

first_2_epsilons = np.array([])
first_2_phis = np.zeros((len(xshoot), 4))
idx = 0
for gamma in gammas:
    A0, e0 = 1e-6, 0.1 

    for modes in range(1, 3):
        dA = 0.01  
        phi = []
        for _ in range(100):
            epsilon, depsilon = e0, 0.2  
            for itr in range(100):
                phi0 = [A0, np.sqrt(L**2 - epsilon) * A0] 
                sol = solve_ivp(shoot, xp, phi0, args=(gamma, epsilon), t_eval=xshoot)
                phi = sol.y
                xs = sol.t
                difference = phi[1, -1] + np.sqrt(L**2 - epsilon) * phi[0, -1]
                if abs(difference) < tol:
                    break
                if (-1) ** (modes + 1) * difference > 0:
                    epsilon += depsilon
                else:
                    epsilon -= depsilon
                    depsilon /= 2 
            A_temp = np.trapz(phi[0, :]**2, xs)
            if abs(abs(A_temp) - 1) < tol:
                break
            elif A_temp < 1:
                A0 += dA
            else:
                A0 -= dA / 2
                dA /= 2  
        e0+=2
        first_2_epsilons = np.append(first_2_epsilons, epsilon)
        normalized_phi = phi[0, :] / np.sqrt(np.trapz(phi[0, :] * phi[0, :], xshoot))
        first_2_phis[:, idx] = np.abs(normalized_phi)
        if gamma == 0.05:
            A5.append(np.abs(normalized_phi))  
            A6.append(epsilon)                 
        else:
            A7.append(np.abs(normalized_phi))  
            A8.append(epsilon)  
        idx += 1

A5 = np.transpose(A5)
A7 = np.transpose(A7)
print("Eigenfunctions for γ = 0.05 (A5):", A5)
print("Eigenvalues for γ = 0.05 (A6):", A6)
print("Eigenfunctions for γ = -0.05 (A7):", A7)
print("Eigenvalues for γ = -0.05 (A8):", A8)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

K = 1
L = 2
E = 1  
x_span = [-L, L]
y0 = [1, np.sqrt(K * L**2 - E)] 

def hw1_rhs_a(x, y, energy):
    phi, dphi_dx = y
    d2phi_dx2 = (K * x**2 - energy) * phi
    return [dphi_dx, d2phi_dx2]

tolerances = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

methods = ['RK45', 'RK23', 'Radau', 'BDF']
A9 = []

for method in methods:
    avg_step_sizes = []

    for tol in tolerances:
        options = {'rtol': tol, 'atol': tol}
        sol = solve_ivp(hw1_rhs_a, x_span, y0, method=method, args=(E,), **options)        
        step_sizes = np.diff(sol.t)
        avg_step_size = np.mean(step_sizes)
        avg_step_sizes.append(avg_step_size)

    log_tolerances = np.log10(tolerances)
    log_avg_step_sizes = np.log10(avg_step_sizes)

    slope, _ = np.polyfit(log_avg_step_sizes, log_tolerances, 1)
    A9.append(slope)

    plt.plot(log_avg_step_sizes, log_tolerances, label=f'{method} (slope={slope:.2f})')

plt.xlabel('Log10(Average Step Size)')
plt.ylabel('Log10(Tolerance)')
plt.title('Convergence Study for ODE Solvers')
plt.legend()
plt.grid(True)
plt.show()

print("Computed slopes for methods (RK45, RK23, Radau, BDF):")
print(np.array(A9).reshape(4, 1))

def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
L=4
x=np.arange(-L,L+0.1,0.1)
h=np.array([(np.ones_like(x)),(2*x),(4*x**2-2),(8*x**3-12*x),(16*x**4-48*x**2+12)])
phi=np.zeros((len(x),5))
for m in range(5):
    phi[:,m]=(np.exp((-x**2)/2)*h[m,:]/np.sqrt(factorial(m)*(2**m)*np.sqrt(np.pi))).T
print(phi)
erpsi_a=np.zeros(5)
erpsi_b=np.zeros(5)
er_a=np.zeros(5)
er_b=np.zeros(5)
ysola,Esola = A1,A2
ysolb,Esolb=A3,A4


for j in range(5):
    erpsi_a[j]=np.trapz((abs(ysola[:,j])-abs(phi[:,j]))**2,x=x)
    erpsi_b[j]=np.trapz((abs(ysolb[:,j])-abs(phi[:,j]))**2,x=x)
    er_a[j]=100*abs(Esola[j]-(2*(j+1)-1))/(2*(j+1)-1)
    er_b[j]=100*abs(Esolb[j]-(2*(j+1)-1))/(2*(j+1)-1)
A10=erpsi_a
A12=erpsi_b
A11=er_a
A13=er_b
