# Lorenz Equations and Attractor Problem ( Problems V and X in ODE Assignment, done by Proshmit Dasgupta )

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


# The system parameters sigma, R and beta - as given in the question
sig=10
R=28
beta=8/3

# Below, I have defined the required system of Lorenz equations

def lorenz_eq_sys(r):
    x, y, z = r
    dxdt = sig*(y-x)
    dydt = x*(R-z) - y
    dzdt = x*y - beta*z
    return np.array([dxdt, dydt, dzdt])

# Some other initial conditions, as given
r0 = [5,5,5]
t_start = 0
t_end = 50
h = 1e-3  
  
# The function below performs one step of the RK4
def rk4_step(eqn, r, h):
    k1 = h * eqn(r)
    k2 = h * eqn(r + k1/2)
    k3 = h * eqn(r + k2/2)
    k4 = h * eqn(r + k3)
    return r + (k1 + 2*k2 + 2*k3 + k4) / 6

# This function defined below performs all the RK4 steps
def solve_Lorenz_eq(r0, t_start, t_end, h):
    n_points = np.arange(t_start, t_end, h)    # Gives number of time steps
    path = np.zeros((len(n_points), 3))        # Gives a matrix of nX3 type
    path[0] = r0
    r=np.array(r0)

    for i in range(1, len(n_points)):
        r = rk4_step(lorenz_eq_sys, r, h)
        path[i]=r
    return n_points, path 

# Definition of the plot function
def lorenz_att_plot(n_points, path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(path[:, 0], path[:, 1], path[:, 2], lw=0.5)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-30, 30)
    ax.set_zlim(0, 50)
    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_zlabel('Z-Axis')
    ax.set_title('Lorenz Attractor')
    plt.show()

# Solving and plotting
n_points, path = solve_Lorenz_eq(r0, t_start, t_end, h)
lorenz_att_plot(n_points, path)

#Solving for trajectory
r0=[5,5,5]
t_start, t_end, h = 0, 50, 1e-3
n_points,path = solve_Lorenz_eq(r0, t_start, t_end, h)
# Below, I have initialized the plot elements (with the help of ChatGPT for animated plot)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

#Setting Axis Limits
ax.set_xlim(np.min(path[:, 0]), np.max(path[:, 0]))
ax.set_ylim(np.min(path[:, 1]), np.max(path[:, 1]))
ax.set_zlim(np.min(path[:, 2]), np.max(path[:, 2]))

#Label and title setting
ax.set_xlabel('X-Axis')
ax.set_ylabel('Y-Axis')
ax.set_zlabel('Z-Axis')
ax.set_title('Lorenz Attractor with Animation')

# Setting up the plot
line, = ax.plot([], [], [], color='m', lw=1)
trail, = ax.plot([], [], [], 'ro', lw=1, alpha=0.5)

# Update function for animation
def update(frame):
    line.set_data(path[:frame+1, 0], path[:frame+1, 1])
    line.set_3d_properties(path[:frame+1, 2])
    trail.set_data([path[frame, 0]], [path[frame, 1]])
    trail.set_3d_properties([path[frame, 2]])
    return line, trail

# Animation Plot
animate = animation.FuncAnimation(fig, update, frames=range(0,len(path),5), interval=10, blit=False)
plt.show()


# Study of the Chaotic behaviour of the plot

#Initial conditions
r0_1 = [5,5,5]
r0_2 = [5 + 1e-5,5,5]
h=1e-3
t_end=20

#Solving the system for the two paths
def solve_double_paths(r0_1, r0_2, h, t_end):
    n_points = np.arange(0, t_end, h)
    path1 = np.zeros((len(n_points), 3))
    path2 = np.zeros((len(n_points), 3))
    distances = np.zeros(len(n_points))
    r1=np.array(r0_1)
    r2=np.array(r0_2)

    for i in range(1, len(n_points)):
        r1 = rk4_step(lorenz_eq_sys, r1, h)
        r2 = rk4_step(lorenz_eq_sys, r2, h)
        path1[i]=r1
        path2[i]=r2
        distances[i]=np.linalg.norm(r1-r2)
    return path1, path2, distances, n_points

path1, path2, distances, n_points = solve_double_paths(r0_1, r0_2, h, t_end)

print("Distance at t=20 for h= 1e-3:",distances[-1])

d=np.log(distances + 1e-10)

#Plot of Log(d) vs x-values
fig = plt.figure()
plt.plot(path1[:,0], d, lw=1, color='r', label='Without perturbation')
plt.plot(path2[:,0], d, lw=1, color='b', label = 'With small perturbation')
plt.xlabel('X Values')
plt.ylabel('log(d)')
plt.legend()
plt.grid()
plt.show()

# Plot of the distance between the two trajectories over time
t=np.linspace(0, t_end, len(n_points))
fig = plt.figure()
plt.plot(t,d)
plt.xlabel("Time [s]")
plt.ylabel("log(d)")
plt.title("Distance between the two trajectories over time (stepsize= 1e-3)")
plt.grid()
plt.show()

#Comparison with different step size value
h1 = 5e-4
_, _, distances, _ = solve_double_paths(r0_1, r0_2, h1, t_end)
print("Distance at t=20 for h= 5e-4:",distances[-1])
print(" We observe that d(20) values for different step size are similar ")

#Plot of the distance between the two trajectories over time (h=5e-4)
fig = plt.figure()
plt.plot(t,d)
plt.xlabel("Time [s]")
plt.ylabel("log(d)")
plt.title("Distance between the two trajectories over time (step-size=5e-4)")
plt.grid()
plt.show()

# New initial conditions with 5e-15 perturbation
h=5e-4
t_end=50
r0_3 = [5 + 5e-15, 5, 5]
_, _, distances, _ = solve_double_paths(r0_1, r0_3, h, t_end)
log_d=np.log(np.maximum(distances, 1e-10))
lambda_empirical= (log_d[-1] - log_d[0])/t_end  # Lyapunov Exponent
print("Lyapunov exponent:", lambda_empirical)
print("Positive value of this exponent confirms that we are dealing with a chaotic system")
print (" d(20) for path r0_3:", distances[-1])

# Lorenz Attractor plots for different initial conditions:

#Initial conditions
r0_1 = [1,1,1]
r0_2 = [1 + 1e-9,1,1]
h=1e-3
t_end=50

#Solving the system for the two paths
def solve_double_paths(r0_1, r0_2, h, t_end):
    n_points = np.arange(0, t_end, h)
    path1 = np.zeros((len(n_points), 3))
    path2 = np.zeros((len(n_points), 3))
    r1=np.array(r0_1)
    r2=np.array(r0_2)

    for i in range(1, len(n_points)):
        r1 = rk4_step(lorenz_eq_sys, r1, h)
        r2 = rk4_step(lorenz_eq_sys, r2, h)
        path1[i]=r1
        path2[i]=r2
    return path1, path2, n_points

# Definition of the plot function
def lorenz_att_plots(n_points, path1, path2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(path1[:, 0], path1[:, 1], path1[:, 2], lw=0.5, label='Unperturbed')
    ax.plot(path2[:, 0], path2[:, 1], path2[:, 2], lw=0.5, label='Perturbed trajectory')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-30, 30)
    ax.set_zlim(0, 50)
    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_zlabel('Z-Axis')
    ax.set_title('Lorenz Attractor Plots')
    plt.legend()
    plt.show()

# Solving and plotting
path1, path2, n_points = solve_double_paths(r0_1, r0_2, h, t_end)
lorenz_att_plots(n_points, path1, path2)

      





    


