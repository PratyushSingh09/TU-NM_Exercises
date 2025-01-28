import numpy as np
import matplotlib.pyplot as plt

def matrixmeth():
    x = np.linspace(0, 1, 100)
    dx = x[1] - x[0]
    n = len(x) # add more n for more steps

    A = np.zeros((n,n))
    b = np.zeros(n)

    # set the boundary conditions
    A[0, 0] = 1 # 1*y = y = 1.2
    A[-1, -1] = 1 # same argument but at the very end
    b[0] = 1.2  # y(0)
    b[-1] = 0.9  # y(1)

    for i in range(1,n-1): # populate the rest of A matrix
        A[i,i-1] = 1/(dx**2)
        A[i,i+1] = 1/(dx**2)
        A[i,i] = -2/(dx**2) - 2

    y = np.linalg.solve(A, b) # python magic -> no iteration required if problem is approached from linear algebraic perspective

    gradient_0 = (y[1]-y[0])/dx

    print("y'(0)=", gradient_0)

    return x,y

x,y = matrixmeth()

plt.plot(x,y, label="Matrix Method Solution")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.title("Solution to Linear BVP using Matrix Method")
plt.legend()
plt.grid()
plt.show()


