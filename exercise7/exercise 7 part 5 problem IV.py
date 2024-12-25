import numpy as np
import matplotlib.pyplot as plt


# array filling function
def fill_array(array, min, max, n, func):
    h = (max-min)/(n-1)
    for i in range(n):
        x = min+(i*h)
        array[i] = func(x)
    #print('Array filled for', min, 'to', max, 'with', n, 'points at', h, 'interval steps.')
    return array

# Newton-Cotes 2nd Order Polynomial - AKA Simpson's Rule
def NC2P(x, f, h):
    # x requires input of x-array
    # f is the array of f(x) over the x-axis

    n = len(x)
    if n%2 == 0:
        print('n is not an odd number.')
        return None

    y = 0 # initial value
    for i in range(0, n-2, 2):
        simps = (h/3) * (f[i] + 4*f[i+1] + f[i+2])
        y += simps

    if n >= 5:  # check for central differencing feasibility
        center = n // 2
        f4 = (f[center+2] - 4*f[center+1] + 6*f[center] - 4*f[center-1] + f[center-2]) / (h**4)
    else:
        print('Not feasible to compute via central differencing.')
        f4 = 0

    error = abs((h**4 / 90) * f4)
    return y, error

# (a) polynomial integration

#min = 0
#max = 1
#n = 5
#h = (max-min)/(n-1)
#func1 = lambda x: x + x**3
#x = np.zeros(n)
#f = fill_array(x, min, max, n, func1)

#integral, error = NC2P(x,f,h)
#print('Integral:', integral)
#print('Error:', error)

#exit()

# results for (a):
# Integral: 0.75
# Error: 0.0

# Error term in 2nd order Newton-Cotes method is proportional to the fourth derivative of f(x), which
# is something a cubic polynomial does not have. Hence, there is an exact result when a cubic polynomial
# is integrated using this method due to the absence of an error term.

# (b) potential well
# L = 2
# w1 = 3
# w2 = 4.5
# domega = w2 - w1
# timestamps = [0, np.pi/domega]
# min = 3/4 * L
# max = L
#
# def probdis(x,t):
#     wavefunc = np.sqrt(1/L) * (np.sin(np.pi*x/L)*np.exp(-1j*w1*t) + np.sin(2*np.pi*x/L)*np.exp(-1j*w2*t))
#     probability_distribution = np.abs(wavefunc)**2
#     return probability_distribution
#
# results_0 = []
# results_t = []
# for n in range(5,502,2):
#     x = np.linspace(min, max, n)
#     h = (max - min) / (n - 1)
#     for t in timestamps:
#         func = lambda x: probdis(x, t)
#         f = fill_array(x, min, max, n, func)
#
#         integral, error = NC2P(x,f,h)
#
#
#         if t == timestamps[0]:
#             results_0.append((np.log(h), np.log(error)))
#         elif t == timestamps[1]:
#             results_t.append((np.log(h), np.log(error)))
#
#      #print(f"At Time: {t}, for n: {n}, Integral: {integral:.6f}, Error: {error:.6f}")
#
# log_h = [result[0] for result in results_0]
# log_error_0 = [result[1] for result in results_0]
# log_error_t = [result[1] for result in results_t]
#
# plt.figure(figsize=(10, 6))
# plt.plot(log_h, log_error_0, label="t = 0", marker='o')
# plt.plot(log_h, log_error_t, label="t = π/w2-w1", marker='s')
#
#
# plt.xlabel("log(h)", fontsize=14)
# plt.ylabel("log(E)", fontsize=14)
# plt.title("log(E) vs. log(h) for Both Times", fontsize=16)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend(fontsize=12)
# plt.show()

# plots have constant slope around 4, which is compatible with the expected convergence rate given by h^4

# (c) extra application

a0 = 0.0529

def psi_2s(r):
    psi_2s = (1/(4 * np.sqrt(2*np.pi* a0**3))) * (2-(r/a0)) * np.exp(-r/(2*a0))
    return psi_2s

def integrand(r,n):
    integrand = 4*np.pi*(psi_2s(r)**2)*(r**n)
    return integrand

min = 0
max = 15*a0
n = 751

h = (max-min)/(n-1)
x1 = np.zeros(n)
x2 = np.zeros(n)
func2 = lambda r: integrand(r,1)
func3 = lambda r: integrand(r,2)
f_mean = fill_array(x1, min, max, n, func2)
f_mean_sq = fill_array(x2, min, max, n, func3)

integral_mean, _ = NC2P(x1,f_mean,h)
integral_mean_sq, _ = NC2P(x2,f_mean_sq,h)
stdev = np.sqrt(integral_mean_sq)

print('Mean Radius:', integral_mean, '* a0')
print('Standard Deviation:', stdev)

exit()