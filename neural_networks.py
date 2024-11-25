import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import os
from functools import partial
import matplotlib
import logging
logging.basicConfig(level=logging.DEBUG)

matplotlib.use('Agg')  # Use a non-interactive backend

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # learning rate
        self.activation_fn = activation  # activation function

        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.zeros((1, output_dim))

    def activation(self, x):
        if self.activation_fn == 'tanh':
            return np.tanh(x)
        elif self.activation_fn == 'relu':
            return np.maximum(0, x)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError("Unsupported activation function")

    def activation_derivative(self, x):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_fn == 'relu':
            return (x > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.activation(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.Z2  # Linear activation for the output layer
        return self.A2

    def backward(self, X, y):
        m = X.shape[0]

        # Compute gradients
        dZ2 = (self.A2 - y) / m
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.activation_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update weights and biases
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

def generate_data(n_samples=100):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    # Hidden Space Visualization (3D Plot)
    hidden_features = mlp.A1
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title("Hidden Space at Step {}".format(frame * 10))
    ax_hidden.set_xlabel("Neuron 1")
    ax_hidden.set_ylabel("Neuron 2")
    ax_hidden.set_zlabel("Neuron 3")

    # Input Space Decision Boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = mlp.forward(grid)
    preds = preds.reshape(xx.shape)
    ax_input.contourf(xx, yy, preds, levels=50, cmap='bwr', alpha=0.5)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_title("Input Space at Step {}".format(frame * 10))
    ax_input.set_xlabel("X1")
    ax_input.set_ylabel("X2")

    # Gradients Visualization
        # Gradients Visualization with Connections and Thickness
    ax_gradient.set_xlim(-1, mlp.W1.shape[1])
    ax_gradient.set_ylim(-1, mlp.W1.shape[0])
    ax_gradient.set_title("Gradients at Step {}".format(frame * 10))

    for i in range(mlp.W1.shape[1]):  # Loop through hidden neurons
        for j in range(mlp.W1.shape[0]):  # Loop through input neurons
            # Thickness of the line represents the gradient magnitude
            gradient_magnitude = abs(mlp.W1[j, i]) * 5  # Scaled for visibility
            ax_gradient.plot(
                [j, i], [0, 1], color='blue', linewidth=gradient_magnitude, alpha=0.7
            )

    # Draw circles to represent neurons
    for i in range(mlp.W1.shape[1]):  # Hidden layer neurons
        ax_gradient.add_patch(Circle((i, 1), radius=0.15, color='green', alpha=0.8))
    for j in range(mlp.W1.shape[0]):  # Input layer neurons
        ax_gradient.add_patch(Circle((j, 0), radius=0.15, color='red', alpha=0.8))

    ax_gradient.set_xlabel("Hidden Layer Neurons")
    ax_gradient.set_ylabel("Input Layer Neurons")



def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)
    

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num // 10, repeat=False)

    # Save the animation as a GIF
    output_file = os.path.join(result_dir, f"visualize_{activation}.gif")
    ani.save(output_file, writer='pillow', fps=10)
    plt.close()
    print(f"Saved visualization as {output_file}")


if __name__ == "__main__":
    activations = ["tanh", "relu", "sigmoid"]
    lr = 0.1
    step_num = 1000
    for activation in activations:
        print(f"Generating visualization for activation: {activation}")
        visualize(activation, lr, step_num)
