import numpy as np

def f(x):
    f = 1 / (1 + np.exp(x)) # Here we use the same function as the previous case: 1/(1+e^(x))
    #print(f)
    return f


def fprime(x):  # computation of true value
    fprime = (np.exp(x) - 1) * np.exp(x) / ((1 + np.exp(x)) ** 3)
    #print(fprime)
    return fprime


def y6(x):  # error term for CDR
    y6 = (np.exp(x) * (np.exp(5 * x) - 57 * np.exp(4 * x) + 302 * np.exp(3 * x) - 302 * np.exp(2 * x) + 57 * np.exp(
        x) - 1)) / (np.exp(x) + 1) ** 7
    return y6


def CDR(x, h):
    cdiff = ((-f(x - 2 * h) + 16 * f(x - h) - 30 * f(x) + 16 * f(x + h) - f(x + 2 * h)) / (12 * h ** 2)) + (
                h ** 4 / 90) * y6(x)
    #print(cdiff)
    return cdiff


def error(x, h):
    error = fprime(x) - CDR(x, h)
    print('True Value:', fprime(x))
    print('Central Differences Value:', CDR(x, h))
    print('Error:', error)
    return error


error(1, 0.1)

# We obtain an error on the order of -10, a much closer fit than that of Problem 1's
# CDR with the same parameters. A smaller error may be obtained if h is made smaller
# (but only up until magnitude -13 due to computational limitations)
