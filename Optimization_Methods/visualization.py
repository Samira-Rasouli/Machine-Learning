import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Gradient_Descent import compute_error_for_line_given_points, gradient_descent_runner  # وارد کردن کد اصلی

def run():
    points = np.genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0  # initial y-intercept guess
    initial_m = 0  # initial slope guess
    num_iterations = 100

    print(f"Starting gradient descent at b = {initial_b}, m = {initial_m}, error = {compute_error_for_line_given_points(initial_b, initial_m, points)}")
    print("Running...")

    history = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

    # Setup the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(points[:, 0], points[:, 1], color='red', label="Data Points")
    line, = ax.plot([], [], color='green', label="Fitted Line", lw=2)
    ax.set_xlim(np.min(points[:, 0]) - 1, np.max(points[:, 0]) + 1)
    ax.set_ylim(np.min(points[:, 1]) - 1, np.max(points[:, 1]) + 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()

    # Create additional axis for showing error and parameters
    ax2 = ax.twinx()
    ax2.set_ylabel("Error", color="blue")
    ax2.tick_params(axis='y', labelcolor="blue")

    # Initialize the line and error text
    error_text = ax.text(0.7, 0.9, '', transform=ax.transAxes, fontsize=12)
    params_text = ax.text(0.7, 0.85, '', transform=ax.transAxes, fontsize=12)

    def init():
        line.set_data([], [])
        error_text.set_text('')
        params_text.set_text('')
        return line, error_text, params_text

    def update(frame):
        b, m = history[frame]
        # Update the line
        line.set_data(points[:, 0], m * points[:, 0] + b)

        # Update error and parameters
        error = compute_error_for_line_given_points(b, m, points)
        error_text.set_text(f"Error: {error:.4f}")
        params_text.set_text(f"b = {b:.4f}, m = {m:.4f}")

        # Update error plot
        ax2.clear()
        ax2.set_ylabel("Error", color="blue")
        ax2.plot(range(frame+1), [compute_error_for_line_given_points(history[i][0], history[i][1], points) for i in range(frame+1)], color='blue')

        return line, error_text, params_text

    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_iterations, init_func=init, blit=True, interval=50)

    # Show the animation
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run()
