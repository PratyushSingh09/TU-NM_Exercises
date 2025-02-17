import numpy as np
import matplotlib.pyplot as plt

g = 9.8
m1 = 1
m2 = 1
l1 = 1
l2 = 1


# tweak as desired

def equations(x, y):
    theta1, z1, theta2, z2 = y  # theta1, theta1_dot, theta2, theta2_dot

    delta = theta1 - theta2
    #mu = 1+ m1 / m2
    #F = mu - (np.cos(delta)** 2)

    # equations of motion taken from https://web.mit.edu/jorloff/www/chaosTalk/double-pendulum/double-pendulum-en.html
    theta1_ddot = (-g * (2 * m1 + m2) * np.sin(theta1) - m2 * g * np.sin(theta1 - 2 * theta2) - 2 * np.sin(
        delta) * m2 * (z2 ** 2 * l2 + z1 ** 2 * l1 * np.cos(delta))) / (l1 * (2 * m1 + m2 - m2 * np.cos(2 * (delta))))
    theta2_ddot = (2 * np.sin(delta) * (
                z1 ** 2 * l1 * (m1 + m2) + g * (m1 + m2) * np.cos(theta1) + z2 ** 2 * l2 * m2 * np.cos(delta))) / (
                              l2 * (2 * m1 + m2 - m2 * np.cos(2 * (delta))))

    # equations of motion from uploaded file (will explode)
    #theta1_ddot = (1 / (l1 * F)) * ((g * (np.sin(theta2) * np.cos(delta) - mu * np.sin(theta1))) - ((((z2 ** 2) * l2) + ((z1 ** 2) * l1 * np.cos(delta))) * np.sin(delta)))
    #theta2_ddot = (1 / (l2 * F)) * ((g * mu * (np.sin(theta1) * np.cos(delta) - np.sin(theta2))) - (((mu * (z1 ** 2) * l1) + ((z2 ** 2) * l2 * np.cos(delta))) * np.sin(delta)))

    # equations of motion (cross reference)
    #theta1_ddot = (1 / F) * (-g * (mu * np.sin(theta1) + np.sin(delta) * np.cos(delta)) - np.sin(delta) * (z2 ** 2 + mu * (z1 ** 2) * np.cos(delta)))
    #theta2_ddot = (1 / F) * (mu * np.sin(delta) * (z1 ** 2 + g * np.cos(theta1)) + mu * g * np.sin(theta2) * np.cos(delta) + (z2 ** 2) * np.sin(delta) * np.cos(delta))

    return [z1, theta1_ddot, z2, theta2_ddot]


def runge_kutta_4(f, y0, t_eval):
    n = len(t_eval)
    h = 40 / (n - 1)
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
        y[i + 1, 0] = (y[i + 1, 0] + np.pi) % (2 * np.pi) - np.pi
        y[i + 1, 2] = (y[i + 1, 2] + np.pi) % (
                    2 * np.pi) - np.pi  # normalize the thetas so they fit in -2pi to 2pi range

    return x, y


t0, tf = 0, 40  # delta t
n = 4001
#y0 = [np.pi/2, 0, np.pi/2, 0]  # initial angles, angle velocities
y0 = [2, 0, 1, 0.1]

epsilon = 1e-8  # lyapunov exp stuff
#y0_ptb = np.array(y0)
#y0_ptb[0] += epsilon
y0_ptb = [2 + epsilon, 0, 1, 0.1]

t_eval = np.linspace(t0, tf, n)

t_vals, y_vals = runge_kutta_4(equations, y0, t_eval)  # ignore t_vals, was in use for non-animated timescale
t_vals, y_ptb_vals = runge_kutta_4(equations, y0_ptb, t_eval)

dt = np.linalg.norm(y_vals - y_ptb_vals, axis=1)
lambda_vals = np.log(dt / epsilon) / t_vals
lambda_vals[0] = 0

theta1_vals, theta2_vals = y_vals[:, 0], y_vals[:, 2]
theta1_ptb_vals, theta2_ptb_vals = y_ptb_vals[:, 0], y_ptb_vals[:, 2]

x1 = l1 * np.sin(theta1_vals)
y1 = -l1 * np.cos(theta1_vals)
x2 = x1 + l2 * np.sin(theta2_vals)
y2 = y1 - l2 * np.cos(theta2_vals)

x3 = l1 * np.sin(theta1_ptb_vals)
y3 = -l1 * np.cos(theta1_ptb_vals)
x4 = x3 + l2 * np.sin(theta2_ptb_vals)
y4 = y3 - l2 * np.cos(theta2_ptb_vals)


# plot animation section
def jumper(theta_vals):  # this one just makes the angle values jump whenever the pendulum does a flip
    theta_jump = np.copy(theta_vals)
    jump_indices = np.where(np.abs(np.diff(theta_vals)) > np.pi)[0]

    for idx in jump_indices:
        theta_jump[idx] = np.nan

    return theta_jump


fig, axs = plt.subplots(3, 2, figsize=(14, 8), gridspec_kw={"height_ratios": [1, 3, 1], "width_ratios": [1, 1]})
ax1 = axs[1, 0]
ax1.set_aspect(1)
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax1.set_aspect('equal')
ax1.set_xlabel("X Position (m)")
ax1.set_ylabel("Y Position (m)")
ax1.set_title("Double Pendulum Motion", loc='left')

time = ax1.text(0.05, 0.95, '', horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes)
line, = ax1.plot([], [], 'o-', lw=2)
trail, = ax1.plot([], [], '-', lw=1, alpha=0.5)
trail_x, trail_y = [], []

ax3 = axs[1, 1]
ax3.set_aspect(1)
ax3.set_xlim(-2, 2)
ax3.set_ylim(-2, 2)
ax3.set_aspect('equal')
ax3.set_xlabel("X Position (m)")
ax3.set_ylabel("Y Position (m)")
ax3.set_title("Perturbed Pendulum Motion", loc='left')

time_ptb = ax3.text(0.05, 0.95, '', horizontalalignment='left', verticalalignment='top', transform=ax3.transAxes)
line_ptb, = ax3.plot([], [], 'o-', lw=2)
trail_ptb, = ax3.plot([], [], '-', lw=1, alpha=0.5)
trail_x_ptb, trail_y_ptb = [], []

ax2 = axs[0, 0]
ax2.set_box_aspect(1 / 3)
#ax2.plot(t_vals, theta1_jump, label=r'$\theta_1(t)$', color='red')
#ax2.plot(t_vals, theta2_jump, label=r'$\theta_2(t)$', color='black')

ax2.axhline(np.pi, color='gray', linestyle='--', alpha=0.5)
ax2.axhline(-np.pi, color='gray', linestyle='--', alpha=0.5)
ax2.set_ylim(-4, 4)
ax2.set_xlim(0, 40)
ax2.set_xlabel("Time (s)", loc='right')
ax2.set_ylabel("Angle (radian)")
ax2.set_title("Double Pendulum Angles Over Time")
ax2.grid()

theta1_line, = ax2.plot([], [], label=r'$\theta_1(t)$', color='red')
theta2_line, = ax2.plot([], [], label=r'$\theta_2(t)$', color='black')
ax2.legend(bbox_to_anchor=(1, 1))

ax4 = axs[0, 1]
ax4.set_box_aspect(1 / 3)
ax4.axhline(np.pi, color='gray', linestyle='--', alpha=0.5)
ax4.axhline(-np.pi, color='gray', linestyle='--', alpha=0.5)
ax4.set_ylim(-4, 4)
ax4.set_xlim(0, 40)
ax4.set_xlabel("Time (s)", loc='right')
ax4.set_ylabel("Angle (radian)")
ax4.set_title("Perturbed Pendulum Angles Over Time")
ax4.grid()

theta1_ptb_line, = ax4.plot([], [], label=r'$\theta_1(t)$', color='red')
theta2_ptb_line, = ax4.plot([], [], label=r'$\theta_2(t)$', color='black')
ax4.legend(bbox_to_anchor=(1, 1))

for ax in axs[2, :]:
    ax.remove()
gs = fig.add_gridspec(3, 2, height_ratios=[1, 3, 1], width_ratios=[1, 1])
ax5 = fig.add_subplot(gs[2, :])

#ax5.set_box_aspect(1/18)
ax5.plot(t_eval, lambda_vals, label=r'$\lambda(t)$')
ax5.set_xlabel("Time (s)")
ax5.set_ylabel("Lyapunov Exponent")
ax5.set_title("Lyapunov Exponent for Double Pendulum")
ax5.legend()
ax5.grid()


def update(frame):
    trail_x.append(x2[frame])
    trail_y.append(y2[frame])

    trail.set_data(trail_x, trail_y)
    line.set_data([0, x1[frame], x2[frame]], [0, y1[frame], y2[frame]])
    time.set_text('time = {:.2f} seconds'.format(frame / 100))

    trail_x_ptb.append(x4[frame])
    trail_y_ptb.append(y4[frame])

    trail_ptb.set_data(trail_x_ptb, trail_y_ptb)
    line_ptb.set_data([0, x3[frame], x4[frame]], [0, y3[frame], y4[frame]])
    time_ptb.set_text('time = {:.2f} seconds'.format(frame / 100))

    theta1_jump = jumper(theta1_vals)
    theta2_jump = jumper(theta2_vals)
    theta1_line.set_data(t_eval[:frame], theta1_jump[:frame])
    theta2_line.set_data(t_eval[:frame], theta2_jump[:frame])

    theta1_ptb_jump = jumper(theta1_ptb_vals)
    theta2_ptb_jump = jumper(theta2_ptb_vals)
    theta1_ptb_line.set_data(t_eval[:frame], theta1_ptb_jump[:frame])
    theta2_ptb_line.set_data(t_eval[:frame], theta2_ptb_jump[:frame])

    return line, time, trail, line_ptb, time_ptb, trail_ptb, theta1_line, theta2_line, theta1_ptb_line, theta2_ptb_line


from matplotlib.animation import FuncAnimation

fig.subplots_adjust(hspace=0.5, wspace=0)

ani = FuncAnimation(fig, update, frames=len(t_eval), interval=1, blit=True, repeat=False)

plt.show()

# Q1: Shown in plot.
# Q2: Shown in plot.
# Q3: Yes, as shown in plot with a perturbation of 1e-8 from initial values: [2+epsilon, 0, 1, 0.1]
# Q4: Lyapunov exponent is positive at all t.

# Q5: The double pendulum system will become more chaotic if the mass ratio increases, since the heavier m1 will have
#     more momentum to drag m2 into behaving more erratically. Conversely, a smaller mass ratio with a heavier m2 will
#     result in m2 leading the movement of the system, dragging m1 along and paradoxically reducing chaotic motion, but
#     not erasing it completely. In extreme cases, a small mass ratio may behave similarly to a single pendulum system.

# Q6: Higher stepsize results in a more accurate simulation, but a lower stepsize may induce chaos as a result of lower
#     resolution. The pendulum system may diverge from simulations of the same initial values at an earlier time as a
#     result.
