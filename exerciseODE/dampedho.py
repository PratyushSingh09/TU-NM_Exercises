import numpy as np
import matplotlib.pyplot as plt

m = 1
k = 1
b = 0.2
def function(t, y):
    x, v = y
    xdot = v
    xddot = -(b / m) * v - (k / m) * x
    return [xdot, xddot]

x0 = 1
v0 = 0
y0 = [x0, v0]

t0, tf = 0, 20
t_eval = np.linspace(t0, tf, 1001)


def runge_kutta_4(f, y0, t_eval):
    n = len(t_eval)
    h = 20 / (n - 1)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    x = t_eval

    for i in range(n - 1):
        k1 = h * np.array(f(x[i], y[i]))
        k2 = h * np.array(f(x[i] + h / 2, y[i] + k1 / 2))
        k3 = h * np.array(f(x[i] + h / 2, y[i] + k2 / 2))
        k4 = h * np.array(f(x[i] + h, y[i] + k3))
        y[i + 1] = y[i] + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
        print(i)
    return x, y


t_vals, y_vals = runge_kutta_4(function, y0, t_eval)

x_vals = y_vals[:, 0]
v_vals = y_vals[:, 1]


plt.figure(figsize=(10, 4))
plt.plot(t_vals, x_vals, label="x(t)")
plt.xlabel("Time t")
plt.ylabel("Displacement x")
plt.title("Damped Harmonic Oscillator, x over t")
plt.grid()
plt.legend()
plt.show()


plt.figure(figsize=(6, 6))
plt.plot(x_vals, v_vals)
plt.xlabel("x (Displacement)")
plt.ylabel("v (Velocity)")
plt.title("Phase Diagram")
plt.grid()
plt.show()

# Q2: Assuming the previous problem refers to the pendulum, the phase diagram here indicates a gradual decrease in the
#     magnitude of displacement and velocity over time. This is not seen in the prior pendulum case, as no dampening
#     mechanic was coded into the simulation. The pendulum's phase diagram would be a circle.

