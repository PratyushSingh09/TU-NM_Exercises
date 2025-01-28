# Assignment 7.1 (Proshmit Dasgupta)

# Problem II (Solving given linear BVP using Shooting method)

import matplotlib.pyplot as plt

# I am defining a function in order to compute y(x) at x=1, for an initial slope named 'yp'. The goal is to find the value of yp for which y(1) = 0.9

# In order to integrate the system, I have used Euler's method

def compute_y1(yp_initial):
    y = 1.2  
    yp = yp_initial  
    x = 0
    dx = 0.01

    y_values = [y]  
    x_values = [x]  

    while x < 1:
        ypp = 2 * y  
        yp = yp + ypp * dx  
        y = y + yp * dx  
        x = x + dx  

        x_values.append(x)
        y_values.append(y)

    return y, x_values, y_values

# Here, bisection method has been used by me as the root-finding algorithm

def shooting_method(target_y1, tol=1e-6):
    low, high = -10, 10  # Initial bounds for yp
    while high - low > tol:
        mid = (low + high) / 2
        y1, _, _ = compute_y1(mid)
        if y1 > target_y1:
            high = mid
        else:
            low = mid

    return (low + high) / 2

# Target boundary condition at x = 1, as given in the question
target_y1 = 0.9

optimal_yp = shooting_method(target_y1)
print("Optimal initial slope, that is y'(0):", optimal_yp)
# The above output is the best possible value for the initial slope - such that the target boundary condition is satisfied.


y_final, x_values, y_values = compute_y1(optimal_yp)
print ("y(x) at x=1 is:", y_final)


# Plot of the solution
plt.plot(x_values, y_values, label="y(x)")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.title("Solution of BVP using Shooting Method")
plt.axhline(target_y1, color='r', linestyle='--', label="Target y(1) = 0.9")
plt.legend()
plt.grid()
plt.show()


