import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def polynomial_3rd_order(theta, a, b, c, d):
    """ Computes f(θ) = aθ^3 + bθ^2 + cθ + d """
    return a * theta ** 3 + b * theta ** 2 + c * theta + d

def gradient(theta, a, b, c):
    """ Computes the gradient: ∇f(θ) = 3aθ^2 + 2bθ + c """
    return 3 * a * theta ** 2 + 2 * b * theta + c

def stochastic_gradient_descent(a, b, c, d, theta_init, alpha, iterations, grad_clip_value=10):
    """ Performs SGD to minimize f(θ) with gradient clipping """
    theta = theta_init
    theta_values = []  # Store theta updates
    loss_values = []  # Store function values

    for _ in range(iterations):
        grad = gradient(theta, a, b, c)

        # Clip the gradient to avoid overflow
        grad = np.clip(grad, -grad_clip_value, grad_clip_value)

        theta -= alpha * grad  # Update theta
        theta_values.append(theta)
        loss_values.append(polynomial_3rd_order(theta, a, b, c, d))

    return theta, theta_values, loss_values

# Define polynomial coefficients
a, b, c, d = 3, -2, -4, 1
initial_theta = 0.0
learning_rate = 0.01
num_iterations = 100

# Run SGD
optimized_theta, theta_vals, loss_vals = stochastic_gradient_descent(a, b, c, d, initial_theta, learning_rate,
                                                                     num_iterations)

# Set up the figure and axis for the animation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot the polynomial function
theta_range = np.linspace(-5, 5, 400)
loss_range = polynomial_3rd_order(theta_range, a, b, c, d)
ax1.plot(theta_range, loss_range, label="f(θ) = 3θ^3 - 2θ^2 - 4θ + 1", color='b')
ax1.set_title('Polynomial Function f(θ)', fontsize=14)
ax1.set_xlabel('θ', fontsize=12)
ax1.set_ylabel('f(θ)', fontsize=12)
ax1.grid(True)
ax1.legend()

line1, = ax1.plot([], [], 'ro', markersize=8)  # Point for current θ

# Plot the loss function over iterations
ax2.plot(range(num_iterations), loss_vals, marker='o', linestyle='-', color='r', label="Loss")
ax2.set_title('Loss Function Over Iterations', fontsize=14)
ax2.set_xlabel('Iterations', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.grid(True)
ax2.legend()

# Adding the updated gradient formula as a text annotation on the plot
ax2.text(0.5, 0.8, r'$\nabla f(\theta) = 9\theta^2 - 4\theta - 4$', horizontalalignment='center',
         fontsize=12, transform=ax2.transAxes)

line2, = ax2.plot([], [], 'go', markersize=6)  # Point for current loss

def update(frame):
    """ Update function for the animation """
    # Update θ plot (pass as sequences even if single point)
    line1.set_data([theta_vals[frame]], [loss_vals[frame]])  # Set as list for single data point

    # Update loss plot
    line2.set_data(range(frame + 1), loss_vals[:frame + 1])  # Plot from start to current frame

    return line1, line2,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=num_iterations, interval=50, blit=True)

plt.tight_layout()
plt.show()
