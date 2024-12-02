# Assignment 06 (PROSHMIT DASGUPTA)
# Problem III
# Part (iii)


import sympy as smp

# Integration symbolically using sympy
def symbolic_double_integral(f, x, y, x_lower, x_upper, y_lower, y_upper):
    inner_integral = smp.integrate(f, (y, y_lower, y_upper))
    result = smp.integrate(inner_integral, (x, x_lower, x_upper))
    return result

# Define the function and symbols
x, y = smp.symbols('x y', real=True, positive=True)
f = x * y**2

# Symbolic limits
x_lower = 0      # Lower limit for x
x_upper = 2      # Upper limit for x
y_lower = 0      # Lower limit for y
y_upper = x / 2  # Upper limit for y

# Analytical solution
analytical_result = symbolic_double_integral(f, x, y, x_lower, x_upper, y_lower, y_upper)

# Numerical approximation (using mid-point method)
def num_double_int(f, y_lower_func, y_upper_func, x_lower, x_upper, nx, ny):
    hx = float((x_upper - x_lower) / nx)
    I = 0
    for j in range(nx):
        xj = x_lower + hx / 2 + j * hx
        y_lower = y_lower_func(xj)
        y_upper = y_upper_func(xj)
        hy = float((y_upper - y_lower) / ny)
        for i in range(ny):
            yi = y_lower + hy / 2 + i * hy
            I += hx * hy * f(xj, yi)  
    return I

numerical_result = num_double_int(
    lambda x, y: x * y**2,  # Function
    lambda x: 0,            # Lower limit for y
    lambda x: x / 2,        # Upper limit for y
    0,                      # Lower limit for x
    2,                      # Upper limit for x
    100, 100                # Grid resolution
)

# Output results
print(f"Analytical result: {analytical_result}")
print(f"Numerical result: {numerical_result}")

