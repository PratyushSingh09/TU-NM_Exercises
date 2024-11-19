from sympy import symbols, expand, Eq, solve


# part 1: verification of P(n,m)
def polygen(degree, x, prefix):
    if prefix == 'b':
        coeffs = [1] + [symbols(f"{prefix}{i}") for i in range(1, degree + 1)]
    else:
        coeffs = [symbols(f"{prefix}{i}") for i in range(degree + 1)]
    poly = sum(coeff * x ** i for i, coeff in enumerate(coeffs))
    return poly


x = symbols('x')

num_degree = 2
denom_degree = 5
degree = num_degree + denom_degree

numerator = polygen(num_degree, x, 'a')
denominator = polygen(denom_degree, x, 'b')

#print(numerator, denominator)
#taylor = x - (1/3)*x**3 + (1/5)*x**5 - (1/7)*x**7 + (1/9)*x**9
taylor = 1 + x + (1 / 2) * x ** 2 + (1 / 6) * x ** 3 + (1 / 24) * x ** 4 + (1 / 120) * x ** 5 + (1 / 720) * x ** 6 + (
            1 / 5040) * x ** 7
coefficients = [taylor.coeff(x, i) for i in range(degree + 1)]
num_coeff = [numerator.coeff(x, i) for i in range(num_degree + 1)]
denom_coeff = [denominator.coeff(x, i) for i in range(denom_degree + 1)]
#print(coefficients)
#print(num_coeff)
#print(denom_coeff)

mix = expand(taylor * denominator)
test = [mix.coeff(x, i) for i in range(num_degree + denom_degree + 1)]
#print('a_n solutions:',test)

#print(denom_coeff)
#print(coefficients)

large = []

for j in range(1, degree - num_degree + 1):
    small = []
    for i in range(denom_degree + 1):
        down = num_degree + j - i
        c_n = coefficients[down]
        b_n = denom_coeff[i]
        mix2 = b_n * c_n
        small.append(mix2)
    large.append(small)

#print(large)

b_coeffs = [symbols(f"b{i}") for i in range(1, len(large) + 1)]

equations = []

for row_idx, row in enumerate(large):
    equation = sum(row)
    equations.append(Eq(equation, 0))

solutions = solve(equations, b_coeffs)
updated_a_n = [mix.subs(solutions).coeff(x, i) for i in range(num_degree + denom_degree + 1)]

print('---------------------------')
for i, value in enumerate(updated_a_n[:num_degree + 1]):
    print(f"a{i} = {value}")
print('---------------------------')
for b, value in solutions.items():
    print(f"{b} = {value}")
print('---------------------------')
# part 2: comparisons
import numpy as np


def taylor(x):
    taylor = 1 + x + (1 / 2) * x ** 2 + (1 / 6) * x ** 3 + (1 / 24) * x ** 4 + (1 / 120) * x ** 5 + (
                1 / 720) * x ** 6 + (1 / 5040) * x ** 7
    return taylor


def P34(x):
    num = 1 + (3 / 7) * x + (1 / 14) * x ** 2 + (1 / 210) * x ** 3
    denom = 1 - (4 / 7) * x + (1 / 7) * x ** 2 - (2 / 105) * x ** 3 + (1 / 840) * x ** 4
    P = num / denom
    return P


def P25(x):
    num = 1 + (2 / 7) * x + (1 / 42) * x ** 2
    denom = 1 - (5 / 7) * x + (5 / 21) * x ** 2 - (1 / 21) * x ** 3 + (1 / 168) * x ** 4 - (1 / 2520) * x ** 5
    P = num / denom
    return P


def e(x):
    e = np.exp(x)
    return e


print('-----------------------------------------------------------------------------------------------------------------')
print('x=0.5 | e:', e(0.5), '| Taylor:', taylor(0.5), '| P(3,4):', P34(0.5), '| P(2,5):', P25(0.5))
print('x=0.5 errors | Taylor:', taylor(0.5)-e(0.5),'| P(3,4):', P34(0.5)-e(0.5), '| P(2,5):', P25(0.5)-e(0.5))
print('-----------------------------------------------------------------------------------------------------------------')
print('x=1 | e:', e(1), '| Taylor:', taylor(1), '| P(3,4):', P34(1), '| P(2,5):', P25(1))
print('x=1 errors | Taylor:', taylor(1)-e(1),'| P(3,4):', P34(1)-e(1), '| P(2,5):', P25(1)-e(1))
print('-----------------------------------------------------------------------------------------------------------------')
print('x=2 | e:', e(2), '| Taylor:', taylor(2), '| P(3,4):', P34(2), '| P(2,5):', P25(2))
print('x=2 errors | Taylor:', taylor(2)-e(2),'| P(3,4):', P34(2)-e(2), '| P(2,5):', P25(2)-e(2))
print('-----------------------------------------------------------------------------------------------------------------')
print('x=5 | e:', e(5), '| Taylor:', taylor(5), '| P(3,4):', P34(5), '| P(2,5):', P25(5))
print('x=5 errors | Taylor:', taylor(5)-e(5),'| P(3,4):', P34(5)-e(5), '| P(2,5):', P25(5)-e(5))
print('-----------------------------------------------------------------------------------------------------------------')
# P(3,4) and P(2,5) perform well when x is small, as seen in x=0.5, with a greater accuracy than the Taylor series
# This continues to x=2, with P(3,4) performing better than P(2,5)
# Pattern breaks between x=2 and x=5, with accuracy diverging significantly as x increases, particularly for P(2,5).
# As seen in the graph plotted below, P(2,5) exhibits high degree of divergence from original e^x function

import matplotlib.pyplot as plt

x_values = np.array([0.5, 1, 2, 5])
x_plot = np.linspace(0, 5, 500)

plt.figure(figsize=(10, 6))
plt.plot(x_plot, e(x_plot), label="e^x")
plt.plot(x_plot, [taylor(x) for x in x_plot], label="Taylor P(7,0)")
plt.plot(x_plot, [P34(x) for x in x_plot], label="Padé P(3,4)")
plt.plot(x_plot, [P25(x) for x in x_plot], label="Padé P(2,5)")
plt.ylim(-50, 150)
plt.title("Approximations of e^x")
plt.xlabel("x")
plt.ylabel("Output")
plt.legend()
plt.grid()
plt.show()

