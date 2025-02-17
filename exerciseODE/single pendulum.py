import numpy as np
import matplotlib.pyplot as plt

g = 9.8
m = 1
l = 1*m

def function(x,y):
    theta, w = y
    theta_dot = w
    theta_ddot = -(g/l)*np.sin(theta)

    return [theta_dot, theta_ddot]

def runge_kutta_4(f, y0, t_eval):
    n = len(t_eval)
    h = 15 / (n - 1)
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

t0, tf = 0, 15 # delta t
n = 1501
y0 = [np.pi / 4, 0]
t_eval = np.linspace(t0, tf, n)

t_vals, y_vals = runge_kutta_4(function, y0, t_eval)
theta_vals = y_vals[:, 0]


import matplotlib.animation as animation

fig, axs = plt.subplots(2, 1, figsize=(14, 6), gridspec_kw={"height_ratios": [1, 1]})
ax = axs[0]

ax.set_aspect(1)
ax.set_xlim(-1.5*l, 1.5*l)
ax.set_ylim(-1.5*l, 1.5*l)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Pendulum Motion")
ax.set_aspect('equal')
ax.grid(False)
line, = ax.plot([], [], 'o-', lw=2)

def setup():
    line.set_data([], [])
    return line,

def update(frame):
    theta = theta_vals[frame]
    x = l * np.sin(theta)
    y = -l * np.cos(theta)
    line.set_data([0, x], [0, y])
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(t_vals), init_func=setup, blit=True, interval=30)


ax2 = axs[1]
ax2.set_aspect(1)
ax2.plot(t_eval, theta_vals, label=r'$\theta(t)$')
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("θ(t)")
ax2.legend(bbox_to_anchor=(1, 1))
ax2.grid()


plt.show()