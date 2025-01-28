# Assignment 7.1 (Proshmit Dasgupta)

# Solving a Nonlinear BVP using the Shooting Method (Problem IV)

import matplotlib.pyplot as plt

# Below, I have defined a function to compute y(x) at x=2, for an initial slope named 'yp'.
# The goal is to find the value of yp for which y(2) = 3.

def compute_y2(yp_initial):
    y = 0  
    yp = yp_initial  
    x = 0
    dx = 0.01

    y_values = [y]  
    x_values = [x]  

    while x < 2:
        ypp = 1 - ((2 + y**2) * y) / (1 + y**2)  
        yp = yp + ypp * dx  
        y = y + yp * dx  
        x = x + dx  

        x_values.append(x)
        y_values.append(y)

    return y, x_values, y_values

# We again use the bisection method as the root-finding algorithm
def shooting_method(target_y2, tol=1e-6):
    low, high = -10, 10         
    while high - low > tol:
        mid = (low + high) / 2
        y2, _, _ = compute_y2(mid)
        if y2 > target_y2:
            high = mid
        else:
            low = mid

    return (low + high) / 2

# Target boundary condition at x = 2, as per given conditions in question
target_y2 = 3

# Here, we find the optimal initial slope yp
optimal_yp = shooting_method(target_y2)
print("Optimal initial slope, y'(0):", optimal_yp)

# Thus, the final solution using the optimal initial slope is as follows:
y_final, x_values, y_values = compute_y2(optimal_yp)
print("y(x) at x=2 is:", y_final)

# Plot of the solution
plt.plot(x_values, y_values, label="y(x)")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.title("Solution of Nonlinear BVP using Shooting Method")
plt.axhline(target_y2, color='m', linestyle='--', label="Target y(2) = 3")
plt.legend()
plt.grid()
plt.show()

