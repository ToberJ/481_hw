import numpy as np 
from scipy.integrate import odeint 
import matplotlib.pyplot as plt 

def shoot2(y, x,  beta):
    return [y[1], (x**2 -beta) * y[0]]  
tol = 10e-6  
col = ['r', 'b', 'g', 'c', 'm', 'k']  
L = 4 
xshoot = np.arange(-L, L + 0.1, 0.1)  
beta_start = 0.1
A2 = []  
A1 = []  

for modes in range(1, 6):  
    beta = beta_start  
    dbeta = 0.2

    for _ in range(1000): 
        y0 = [1, np.sqrt(L**2 - beta)]
        y = odeint(shoot2, y0, xshoot, args=(beta,))  
        if abs(y[-1, 1] + np.sqrt(L**2 -beta) * y[-1, 0]) < tol:  
            A2.append(beta)  
            break  

        if ((-1) ** (modes + 1) * (y[-1,1]+ np.sqrt((L**2 -beta)) * y[-1, 0])) > 0:
            beta += dbeta
        else:
            beta -= dbeta
            dbeta /= 2
    beta_start = beta + 0.1  
    norm = np.trapz(y[:, 0] * y[:,0], xshoot)  

    normalized_eigenfunction = abs(y[:, 0] / np.sqrt(norm))

    A1.append(normalized_eigenfunction)
    plt.plot(xshoot, normalized_eigenfunction, col[modes - 1], label=f'Mode {modes}')
plt.show()  

A1 = np.transpose(A1)
# rows, cols = A1.shape
# print("Rows:", rows)  # Output: Rows: 5
# print("Columns:", cols)  
print(A1)
print(A2) 

