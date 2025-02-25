import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Settings
function_str = '6*sin(x) + 3*cos(y)**2 + x**2 + y**2'  # Objective function
x_vals = np.linspace(-3, 6, 200)
y_vals = np.linspace(-3, 5, 200)
x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)

# Calculate the value of the objective function at each point
from nesterov import nesterov_accelerated_gradient  # Assuming the main code is stored in the file nesterov.py
history, function = nesterov_accelerated_gradient(function_str)

# Display optimization path in 3D space
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

z_vals = function(x_mesh, y_mesh)
ax.plot_surface(x_mesh, y_mesh, z_vals, cmap='viridis', alpha=0.7)

history_x, history_y = zip(*history)
history_z = function(np.array(history_x), np.array(history_y))

point, = ax.plot([], [], [], 'ro', markersize=5, label='NAG Update')
path, = ax.plot([], [], [], linestyle='dashed', color='red', alpha=0.6)

def init():
    point.set_data([], [])
    point.set_3d_properties([])
    path.set_data([], [])
    path.set_3d_properties([])
    return point, path

def update(frame):
    point.set_data([history_x[frame]], [history_y[frame]])
    point.set_3d_properties([history_z[frame]])
    path.set_data(history_x[:frame + 1], history_y[:frame + 1])
    path.set_3d_properties(history_z[:frame + 1])
    return point, path

ani = FuncAnimation(fig, update, frames=len(history), init_func=init, blit=False, interval=200)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$f(x, y)$')
ax.set_title('Nesterov Accelerated Gradient Descent (NAG) in 3D')
plt.legend()
plt.show()
