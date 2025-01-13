import numpy as np
import matplotlib.pyplot as plt

def f_y(x,y,z):
    return np.sin(y)+np.cos(z*x)

def f_z(x,y,z):
    return np.exp(-y*x)+np.where(x == 0, 0, np.sin(z*x)/x)

def runge_kutta_4(fy,fz,y0,z0,h):
    n = int((5/h)+1)
    x = np.linspace(-1, 4, n)
    y = np.zeros(n)
    z = np.zeros(n)
    y[0] = y0
    z[0] = z0

    for i in range(n-1):
        k1y = h*fy(x[i],y[i],z[i])
        k1z = h*fz(x[i],y[i],z[i])
        k2y = h*fy(x[i]+h/2,y[i]+k1y/2,z[i]+k1z/2)
        k2z = h*fz(x[i]+h/2,y[i]+k1y/2,z[i]+k1z/2)
        k3y = h*fy(x[i]+h/2,y[i]+k2y/2,z[i]+k2z/2)
        k3z = h*fz(x[i]+h/2,y[i]+k2y/2,z[i]+k2z/2)
        k4y = h*fy(x[i]+h,y[i]+k3y,z[i]+k3z)
        k4z = h*fz(x[i]+h,y[i]+k3y,z[i]+k3z)
        y[i+1] = y[i]+k1y/6+k2y/3+k3y/3+k4y/6
        z[i+1] = z[i]+k1z/6+k2z/3+k3z/3+k4z/6
    return x,y,z

y0 = 2.37
z0 = -3.48
h = 0.25 #step size

x,y,z = runge_kutta_4(f_y,f_z,y0,z0,h)

plt.figure(figsize=(12,6))

# y(t) and z(t) against t
plt.subplot(1, 2, 1)
plt.plot(x, y, label="y'")
plt.plot(x, z, label="z'")
plt.title("y(x) and z(x) as functions of x")
plt.xlabel("x")
plt.ylabel("y(x) and z(x)")
plt.legend()
plt.grid()

# 1(b) np.sin(x*y)*np.cos(x+y)
plt.subplot(1, 2, 2)
plt.plot(y, z)
plt.title("Parametric Plot y(x) vs z(x)")
plt.xlabel("y(x)")
plt.ylabel("z(x)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
