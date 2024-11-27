import numpy as np
from scipy.sparse import spdiags, diags
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt

# Initialize variables
m = 8  # N value in x and y directions
n = m * m  # total size of matrix
dn = 20 / m

# Create zero and one vectors
e0 = np.zeros(n)
e1 = np.ones(n)

# Create copies of vectors
e2 = e1.copy()
e4 = e0.copy()

# Modify every m-th element
for j in range(1, m + 1):
    e2[m * j - 1] = 0  # zero out every m-th value
    e4[m * j - 1] = 1  # set every m-th value to one

# Shift elements for e3 and e5
e3 = np.roll(e2, 1)
e5 = np.roll(e4, 1)

# Place diagonal elements
diagonals = [e1, e1, e5, e2, -4 * e1, e3, e4, e1, e1]
offsets = [-(n - m), -m, -m + 1, -1, 0, 1, m - 1, m, (n - m)]
matA = spdiags(diagonals, offsets, n, n).toarray()
A = matA / (dn ** 2)

# Matrix B
adds = (1 / (2 * dn)) * np.ones(n)
subs = -adds
B = spdiags([adds, adds, subs, subs], [m, -(n - m), n - m, -m], n, n).toarray()

# Matrix C
e0 = np.zeros(n)
e1 = np.ones(n)
e2 = e0.copy()
e3 = e1.copy()

# Modify based on conditions
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

# Additional modification to matrix C
for i in range(m):
    row_index = i * m      # Start of each row
    col_index = (i + 1) * m - 1  # End of each row
    if row_index < n and col_index < n:
        C[row_index, col_index] = -1
A3 = C * (1 / (2 * dn))

def plot_matrix(matrix, title):
    plt.figure(figsize=(6, 6))
    plt.imshow(matrix, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.title(title)
    plt.show()

# Plot each matrix
plot_matrix(A1, "Matrix A")
plot_matrix(A2, "Matrix B")
plot_matrix(A3, "Matrix C")