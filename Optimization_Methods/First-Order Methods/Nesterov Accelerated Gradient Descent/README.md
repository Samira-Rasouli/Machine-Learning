#Nesterov Accelerated Gradient Descent (NAG)

Nesterov Accelerated Gradient Descent (NAG) is an advanced optimization technique designed to improve the performance of standard Gradient Descent algorithms. It was introduced by Yurii Nesterov and aims to accelerate the convergence of gradient descent methods.

In a typical Gradient Descent algorithm, the gradient of the objective function at the current point is used to update the parameters. However, this approach can converge slowly, especially in cases where the function has a very flat region or where the gradients change slowly.

Nesterov Accelerated Gradient Descent uses a technique called Momentum to address this issue. The main idea is to make a prediction about the future position of the parameters, rather than just using the gradient at the current position. This allows for faster corrections and less oscillation during the optimization process.

**NAG Algorithm Steps**

The Nesterov Accelerated Gradient Descent algorithm consists of three main steps:

**1. Predicting the future position: x_lookahead  = x + μv_x**

   - μ is the Momentum parameter.
     
   - v_x is the velocity (momentum) term that accumulates the past gradients.
     
**2. Calculating the gradient at the predicted position: ∇f(x_lookahead)**

   - The gradient of the objective function is computed at the predicted (lookahead) position.
     
**3. Updating the velocity and parameters:**

![image](https://github.com/user-attachments/assets/b27d2419-1980-481e-86a5-b90b18cb009c)


**Nesterov Accelerated Gradient Descent Algorithm:**
The NAG algorithm is implemented here. In each epoch, we predict the future position of the parameters, compute the gradient at that position, and update the parameters.

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
