#Assignment-04
#Problem II

#Solution:

import numpy as np

#Setting data values as per the difference matrix table given,

x=np.array([4,6,8,10])
y=np.array([1,3,8,20])  # f(x)

n=np.size(x)-1          #n is the degree of the polynomial
f=np.zeros((n+1,n+1))
f[:,0] = y

#Divided differences table algorithm:

for j in range(1,n+1):                                 # 'j' represents the columns in the divided differences table, ranging from 1,2...n ( In our case, n=3 )
    for i in range(0,(n-j)+1):                         # 'i' represents the rows, ranging from 0 to (n-j)
        f[i,j]=(f[i+1,j-1]-f[i,j-1])/(x[i+j]-x[i])     

#This prints the divided differences table for the construction of the Newton interpolated polynomial:

print("\nThe divided differences table is:")
print(f)

#Algorithm for the formation of the Newton Polynomial:

def newton_polynomial(x, f, n):
    from sympy import symbols, expand    # This module helps in expansion and simplification of the constructed Newton polynomial
    X = symbols('x')  
    poly = f[0, 0]                       # Constant term outside the summation
    term = 1

    for i in range(1, n + 1):
        term *= (X - x[i - 1])  
        poly += f[0, i] * term

    
    simplify = expand(poly)
    return simplify

result = newton_polynomial(x, f, n)
print("\nThe simplified third degree Newton polynomial is as follows:")
print(result)



