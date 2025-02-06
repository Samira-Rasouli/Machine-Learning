**Stochastic Gradient Descent (SGD)**
![figure](https://github.com/user-attachments/assets/95074187-9458-4ab7-a8f0-b0163fc37f93)


**1. Introduction**

Stochastic Gradient Descent (SGD) is a variant of Gradient Descent that updates model parameters using only a single randomly chosen data point (or a small subset) at each iteration. This makes SGD much faster and more scalable for large datasets compared to traditional Batch Gradient Descent.

**2. How SGD Works**

Instead of computing the gradient over the entire dataset, SGD updates the parameters using one data sample at a time:


                                                θ(t+1) = θ(t) - α ∇f_i(θ(t))


where:
- θ represents the model parameters,

- α is the learning rate,

- f_i(θ) is the loss function evaluated at a single randomly chosen data point i,

- ∇f_i(θ) is the gradient computed using that single data point.

**3. Advantages of SGD**

✅ Faster Computation – Computationally efficient, especially for large datasets.

✅ Online Learning – Can be used in streaming data scenarios where the dataset is too large to be stored in memory.

✅ Better Generalization – The randomness of updates introduces noise, helping escape local minima and improving generalization.

**4. Disadvantages of SGD**

❌ High Variance in Updates – Updates can be noisy and cause erratic convergence.

❌ Difficult to Tune the Learning Rate – Choosing the right learning rate is crucial; too high leads to divergence, and too low causes slow convergence.

❌ Does Not Always Converge Smoothly – The updates fluctuate due to random sampling, making convergence less stable than batch gradient descent.

**5. Techniques to Improve SGD**

Several methods help improve the performance and stability of SGD:

**A. Mini-Batch SGD**

Instead of updating the model using a single data point, we use a small batch (e.g., 32 or 128 samples) to compute the gradient. This balances the efficiency of SGD with the stability of batch gradient descent.

**B. Momentum-based SGD**

Instead of relying only on the current gradient, Momentum SGD accumulates past gradients to smooth out updates.

Update rule:

                                            v_t = β v_{t-1} + (1 - β) ∇f_i(θ)

                                                      θ = θ - α v_t



where β is a momentum coefficient (usually between 0.9 and 0.99).

**C. Adaptive Learning Rate Methods**

Adagrad, RMSProp, and Adam are optimization algorithms that adjust the learning rate dynamically for each parameter to stabilize learning. Adam (Adaptive Moment Estimation) is the most commonly used optimization algorithm in deep learning, combining both momentum and adaptive learning rates.

**6. Applications of SGD**

SGD is widely used in:

- Deep Learning – Training neural networks in frameworks like TensorFlow and PyTorch.
- Logistic Regression & Linear Regression – Optimizing cost functions in large datasets.
- Support Vector Machines (SVMs) – Efficient training of SVMs using large-scale data.
- Recommendation Systems – Optimizing matrix factorization models.

**7. Conclusion**

Stochastic Gradient Descent is a powerful optimization technique widely used in machine learning and deep learning due to its speed and scalability. However, it requires proper tuning and often benefits from enhancements like mini-batches, momentum, or adaptive learning rate methods to achieve stable and efficient convergence.






