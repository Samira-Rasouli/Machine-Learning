import numpy as np
import sympy as sp

# Function to compute the objective function and gradient from the input
def compute_gradient(function_str):
    x, y = sp.symbols('x y')  # Define variables
    function = sp.sympify(function_str)  # Convert input string to an algebraic expression
    grad_x = sp.diff(function, x)  # Derivative with respect to x
    grad_y = sp.diff(function, y)  # Derivative with respect to y

    # Convert algebraic expression to usable functions
    function_lambda = sp.lambdify((x, y), function, 'numpy')
    grad_x_lambda = sp.lambdify((x, y), grad_x, 'numpy')
    grad_y_lambda = sp.lambdify((x, y), grad_y, 'numpy')

    return function_lambda, grad_x_lambda, grad_y_lambda


# Nesterov accelerated gradient descent algorithm
def nesterov_accelerated_gradient(function_str, lr=0.1, momentum=0.9, epochs=50, x_init=2.5, y_init=2.5):
    # Compute the objective function and gradient
    function, grad_x, grad_y = compute_gradient(function_str)

    # Starting from initial values
    x, y = x_init, y_init
    v_x, v_y = 0, 0  # Initial velocity in both directions
    history = [(x, y)]  # Store history to display the optimization path

    for _ in range(epochs):
        # Predict the future position
        x_lookahead, y_lookahead = x + momentum * v_x, y + momentum * v_y
        grad_x_val, grad_y_val = grad_x(x_lookahead, y_lookahead), grad_y(x_lookahead, y_lookahead)

        # Update velocities
        v_x = momentum * v_x - lr * grad_x_val
        v_y = momentum * v_y - lr * grad_y_val

        # Update values of x and y
        x += v_x
        y += v_y
        history.append((x, y))

    return history, function
