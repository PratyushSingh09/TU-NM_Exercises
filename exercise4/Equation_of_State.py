import numpy as np
import matplotlib.pyplot as plt

def interp(xs,ys,n,x):
    result=0
    for i in range(n):          #loop for Summation
        t = 1
        for j in range(n):      #Loop for multiplication
            if i==j: 
                continue
            else:
                t = (x - xs[j])/(xs[i] - xs[j])*t

        result = t*ys[i] + result

    return result

# Defined paramters
gamma_1 = 4/3
gamma_2 = 5/3
k_1 = 20
k_2 = 1
rho_T = 5     # Transition Density
alpha = 5     # Transition speed parameter

#Heaviside function
def H(alpha,x):
    return 0.5*(1 + np.tanh(alpha*x))
#EOS Function
def P(rho,alpha,rho_T,k_1,k_2,gamma_1,gamma_2):
    return H(alpha,rho_T - rho)*k_1*(rho**(gamma_1)) + H(alpha,rho - rho_T)*k_2*(rho**(gamma_2))

N = [5,10,20]
# Array that describes the uniform density distribution
xs = np.linspace(3, 7, N[2])
# Pressure for corresponding density values
ys = [P(rho,alpha,rho_T,k_1,k_2,gamma_1,gamma_2) for rho in xs] 

m = 200
xk = np.linspace(4.5,5.5,m)    # Uniform density interval with m = 200 points
yk_1 = [interp(xs,ys,N[0],rho) for rho in xk]
yk_2 = [interp(xs,ys,N[1],rho) for rho in xk]
yk_3 = [interp(xs,ys,N[2],rho) for rho in xk]

plt.figure('Pressure vs Density')
plt.title("For interval I = [4.4,5.5]")
plt.plot(xs,ys,color="cyan",label = 'Analytical Function')
plt.plot(xk,yk_1, color="red",label = 'n = 5')
plt.plot(xk,yk_2, color="magenta",label = 'n = 10')
plt.plot(xk,yk_3, color="green",label = 'n = 20')
plt.xlabel(r"$Density (\rho)$")
plt.ylabel("Pressure P")
plt.legend()
plt.show()
# Change of interval I to [0,10]:
xk_new = np.linspace(0,10,m)
yk_1_new = [interp(xs,ys,N[0],rho) for rho in xk_new]
yk_2_new = [interp(xs,ys,N[1],rho) for rho in xk_new]
yk_3_new = [interp(xs,ys,N[2],rho) for rho in xk_new]

plt.figure('Pressure vs Density')
plt.title("For interval I = [0,10]")
plt.plot(xs,ys,color="cyan",label = 'Analytical Function')
plt.plot(xk_new,yk_1_new, color="red",label = 'n = 5')
plt.plot(xk_new,yk_2_new, color="magenta",label = 'n = 10')
plt.plot(xk_new,yk_3_new, color="green",label = 'n = 20')
plt.xlabel(r"$Density (\rho)$")
plt.ylabel("Pressure P")
plt.legend()
plt.show()
# Below, the chi-squared error for the function has been defined
f_xk = [P(rho,alpha,rho_T,k_1,k_2,gamma_1,gamma_2) for rho in xk]
E1 = 0
for j in range(m):
    t1 = (f_xk[j] - yk_3[j])**2
    E1 += t1

E1 = np.sqrt((1/m)*E1)
print("chi-squared error for f(xk) and y(k) for points within the interval I = [4.4,5.5] and N=20 points",E1)
#  Plotting log(E) vs n for a range n = 3 to n = 40 for three separate intervals
n = np.arange(3,41,1)
# i) For I=[4.5,5.5]
xk_1 = np.linspace(4.5,5.5,m)
yk_1 = []
E_1 = []
f_xk = [P(rho, alpha, rho_T, k_1, k_2, gamma_1, gamma_2) for rho in xk_1]
for i in range(len(n)):
    xs = np.linspace(2, 9, n[i])
    ys = [P(rho, alpha, rho_T, k_1, k_2, gamma_1, gamma_2) for rho in xs]
    
    yk__ = [interp(xs, ys, n[i], rho) for rho in xk]
    yk_1.append(yk__)
    E1 = 0
    for j in range(m):
        t1 = (f_xk[j] - yk__[j]) ** 2
        E1 += t1
    
    E_1.append(np.log(np.sqrt((1 / m) * E1)))

plt.figure(r"$log(E) vs. n$")
plt.plot(n, E_1, marker="o", color="tab:orange", label = "For interval [4.4,5.5]")
plt.xlabel("n")
plt.ylabel("Log(E)")
plt.legend()
plt.show()
# ii) For I=[0,10]
xk_2 = np.linspace(0,10,m)
yk_2 = []
E_2 = []
f_xk = [P(rho, alpha, rho_T, k_1, k_2, gamma_1, gamma_2) for rho in xk_2]
for i in range(len(n)):
    xs = np.linspace(2, 9, n[i])
    ys = [P(rho, alpha, rho_T, k_1, k_2, gamma_1, gamma_2) for rho in xs]
    
    yk__ = [interp(xs, ys, n[i], rho) for rho in xk]
    yk_2.append(yk__)
    E1 = 0
    for j in range(m):
        t1 = (f_xk[j] - yk__[j]) ** 2
        E1 += t1
    
    E_2.append(np.log(np.sqrt((1 / m) * E1)))

plt.figure(r"$log(E) vs. n$")
plt.plot(n, E_2, marker="o", color="tab:pink", label = "For interval [0,10]")
plt.xlabel("n")
plt.ylabel("Log(E)")
plt.legend()
plt.show()
# iii) For interval I = [0:30]
xk_3 = np.linspace(0,30,m)
yk_3 = []
E_3 = []
f_xk = [P(rho, alpha, rho_T, k_1, k_2, gamma_1, gamma_2) for rho in xk_3]
for i in range(len(n)):
    xs = np.linspace(2, 9, n[i])
    ys = [P(rho, alpha, rho_T, k_1, k_2, gamma_1, gamma_2) for rho in xs]
    
    yk__ = [interp(xs, ys, n[i], rho) for rho in xk]
    yk_3.append(yk__)
    E1 = 0
    for j in range(m):
        t1 = (f_xk[j] - yk__[j]) ** 2
        E1 += t1
    
    E_3.append(np.log(np.sqrt((1 / m) * E1)))

plt.figure(r"$log(E) vs. n$")
plt.plot(n, E_3, marker="o", color="tab:olive", label = "For interval [0,30]")
plt.xlabel("n")
plt.ylabel("Log(E)")
plt.legend()
plt.show()

