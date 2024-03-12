import numpy as np
import matplotlib.pyplot as plt

# Define the initial weights and learning rate
W = np.array([10, 0.2, -0.75])
learning_rate = 0.05

# Define the AND gate input data and targets
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Define activation functions
def step_activation(x):
    return np.where(x >= 0, 1, 0)

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def relu_activation(x):
    return np.maximum(0, x)

# Calculate the sum-square-error
def calculate_error(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)

# Train the perceptron with different activation functions
activation_functions = [step_activation, sigmoid_activation, relu_activation]

for activation_function in activation_functions:
    W = np.array([10, 0.2, -0.75])  # Reset weights for each activation function
    errors = []
    epochs = 0
    convergence_error = 0.002
    max_epochs = 1000

    while True:
        epochs += 1
        total_error = 0

        for i in range(len(X)):
            # Forward pass
            z = np.dot(np.insert(X[i], 0, 1), W)
            y_pred = activation_function(z)

            # Backward pass (update weights)
            error = y[i] - y_pred
            W += learning_rate * error * np.insert(X[i], 0, 1)

            total_error += calculate_error(y[i], y_pred)

        errors.append(total_error)

        # Check for convergence
        if total_error <= convergence_error or epochs >= max_epochs:
            break

    # Plot epochs against error values for each activation function
    plt.plot(range(1, epochs + 1), errors, label=activation_function._name_)

plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs Epochs for Different Activation Functions')
plt.legend()
plt.show()