# Assignment 06 (PROSHMIT DASGUPTA)
# Problem III
# Part (ii)


import sympy as smp

# Symbolic integration using sympy
def symbolic_double_integral(f, x, y, x_lower, x_upper, y_lower, y_upper):
    inner_integral = smp.integrate(f, (x, x_lower, x_upper))
    result = smp.integrate(inner_integral, (y, y_lower, y_upper))
    return result

# Function and symbols
x, y = smp.symbols('x y', real=True, positive=True)
f = x * y**2

# Symbolic limits
x_lower = 2 * y  # Lower limit for x
x_upper = 2      # Upper limit for x
y_lower = 0      # Lower limit for y
y_upper = 1      # Upper limit for y

# Analytical solution for the given double integral
analytical_result = symbolic_double_integral(f, x, y, x_lower, x_upper, y_lower, y_upper)

# My Numerical approximation using mid-point method
def num_double_int(f, x_lower_func, x_upper_func, y_lower, y_upper, nx, ny):
    hy = float((y_upper - y_lower) / ny)
    I = 0
    for j in range(ny):
        yj = y_lower + hy / 2 + j * hy
        x_lower = x_lower_func(yj)
        x_upper = x_upper_func(yj)
        hx = float((x_upper - x_lower) / nx)
        for i in range(nx):
            xi = x_lower + hx / 2 + i * hx
            I += hx * hy * f(xi, yj)
    return I

numerical_result = num_double_int(
    lambda x, y: x * y**2,
    lambda y: 2 * y,       # Lower limit for x
    lambda y: 2,           # Upper limit for x
    0,                     # Lower limit for y
    1,                     # Upper limit for y
    100, 100               # Grid resolution
)

# Output results
print(f"Analytical result: {analytical_result}")
print(f"Numerical result: {numerical_result}")



