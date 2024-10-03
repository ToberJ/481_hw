import numpy as np


def newton_raphson(x0, tol=1e-6, max_iter=1000):
    x = [x0]
    for j in range(max_iter):
        
        fx = x[j] * np.sin(3 * x[j]) - np.exp(x[j])
        f_prime_x = np.sin(3 * x[j]) + 3 * x[j] * np.cos(3 * x[j]) - np.exp(x[j])
        x_next = x[j] - fx / f_prime_x
        x.append(x_next)
        fc = x[j+1] * np.sin(3 * x[j+1]) - np.exp(x[j+1])
    
        if abs(fc) < tol:
            x.append(x_next)
            break
    return x

# Bisection Method
def bisection(a, b, tol=1e-6, max_iter=1000):
    mid_points = []  
    for i in range(max_iter):
        c = (a + b) / 2.0
        fc = c*np.sin(3*c) - np.exp(c)
        
        mid_points.append(c)
        
        if abs(fc) < tol:
            break

        if abs(b - a) < tol:
            break
        
        if fc > 0:
            a = c
        else:
            b = c
        

    return mid_points

# Initial parameters
x0_newton = -1.6
a_bisect = -0.7
b_bisect = -0.4
tolerance = 1e-6

# Newton-Raphson Method
A1 = newton_raphson(x0_newton, tol=tolerance)
print("A1 = ")
for x in A1:
    print(x)

# Bisection Method
A2 = bisection(a_bisect, b_bisect, tol=tolerance)
print("A2 = ")
for x in A2:
    print(x)

# Number of iterations
A3 = np.array([len(A1), len(A2)])
print("A3 =", A3)

A = np.array([[1, 2], [-1, 1]])  
B = np.array([[2, 0], [0, 2]])   
C = np.array([[2, 0, -3], [0, 0, -1]])  
D = np.array([[1, 2], [2, 3], [-1, 0]])  

x = np.array([1, 0])  
y = np.array([0, 1])  
z = np.array([1, 2, -1])  

A4 = A + B

A5 = 3 * x - 4 * y

A6 = A @ x

A7 = B @ (x - y)

A8 = D @ x

A9 = D @ y + z

A10 = A @ B

A11 = B @ C

A12 = C @ D

print("A4 = ", A4)
print("A5 = ", A5)
print("A6 = ", A6)
print("A7 = ", A7)
print("A8 = ", A8)
print("A9 = ", A9)
print("A10 = ", A10)
print("A11= ", A11)
print("A12 = ", A12)