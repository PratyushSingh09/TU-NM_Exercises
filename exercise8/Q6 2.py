import numpy as np
import matplotlib.pyplot as plt

def f_a(x, y):
    return y*np.cos(x+y)

def f_b(x, y):
    return np.sin(x*y)*np.cos(x+y)

def heun(f,y0,h):
    n = int(30/h+1)
    x = np.linspace(0, 30, n)
    y = np.zeros(n)
    y[0] = y0

    for i in range(n-1):
        k1 = f(x[i],y[i])
        predictor = y[i]+h*k1
        k2 = f(x[i+1],predictor)
        y[i+1] = y[i]+(h/2)*(k1+k2)
    return x,y

def runge_kutta_4(f,y0,h):
    n = int(30/h+1)
    x = np.linspace(0, 30, n)
    y = np.zeros(n)
    y[0] = y0

    for i in range(n-1):
        k1 = h*f(x[i],y[i])
        k2 = h*f(x[i]+h/2,y[i]+k1/2)
        k3 = h*f(x[i]+h/2,y[i]+k2/2)
        k4 = h*f(x[i]+h,y[i]+k3)
        y[i+1] = y[i]+k1/6+k2/3+k3/3+k4/6
    return x,y

def adams_4(f,y0,h):
    n = int(30/h+1)
    x = np.linspace(0, 30, n)
    y = np.zeros(n)
    y[0] = y0

    for i in range(3): #copied from Runge-Kutta method
        k1 = h*f(x[i],y[i])
        k2 = h*f(x[i]+h/2,y[i]+k1/2)
        k3 = h*f(x[i]+h/2,y[i]+k2/2)
        k4 = h*f(x[i]+h,y[i]+k3)
        y[i+1] = y[i]+k1/6+k2/3+k3/3+k4/6

    for i in range(3, n-1): #everything else
         predictor = y[i]+(h/24)*((55*f(x[i],y[i]))-(59*f(x[i-1],y[i-1]))+(37*f(x[i-2],y[i-2]))-(9*f(x[i-3],y[i-3])))
         y[i+1] = y[i]+(h/24)*((9*f(x[i+1],predictor))+(19*f(x[i],y[i]))-(5*f(x[i-1],y[i-1]))+f(x[i-2],y[i-2]))
    return x,y

y0 = 1
h = 0.1 #step size

x_a_heun,y_a_heun = heun(f_a,y0,h)
x_a_rk4,y_a_rk4 = runge_kutta_4(f_a,y0,h)
x_a_adams4,y_a_adams4 = adams_4(f_a,y0,h)

x_b_heun,y_b_heun = heun(f_b,y0,h)
x_b_rk4,y_b_rk4 = runge_kutta_4(f_b,y0,h)
x_b_adams4,y_b_adams4 = adams_4(f_b,y0,h)

plt.figure(figsize=(14,9))

# 1(a) y*np.cos(x+y)
plt.subplot(2, 1, 1)
plt.plot(x_a_heun, y_a_heun, label="Heun's Method")
plt.plot(x_a_rk4, y_a_rk4, label="RK4 Method")
plt.plot(x_a_adams4, y_a_adams4, label="Adams4 Method")
plt.title("y' = ycos(x+y), y(0) = 1")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()

# 1(b) np.sin(x*y)*np.cos(x+y)
plt.subplot(2, 1, 2)
plt.plot(x_b_heun, y_b_heun, label="Heun's Method")
plt.plot(x_b_rk4, y_b_rk4, label="RK4 Method")
plt.plot(x_b_adams4, y_b_adams4, label="Adams4 Method")
plt.title("y' = sin(xy)cos(x+y), y(0) = 1")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
