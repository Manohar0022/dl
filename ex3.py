import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# XOR input and output datasets
inputs = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
outputs = np.array([[1], [1], [0], [0]])

# Initialize weights and biases
np.random.seed(42)
input_layer_neurons = 2
hidden_layer_neurons = 2
output_layer_neurons = 1

# Random initialization of weights and biases
hidden_weights = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
hidden_bias = np.random.uniform(size=(1, hidden_layer_neurons))
output_weights = np.random.uniform(size=(hidden_layer_neurons, output_layer_neurons))
output_bias = np.random.uniform(size=(1, output_layer_neurons))

# Set learning rate
learning_rate = 0.1

# Training the MLP
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    hidden_layer_input = np.dot(inputs, hidden_weights) + hidden_bias
    hidden_layer_activation = tanh(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_activation, output_weights) + output_bias
    predicted_output = tanh(output_layer_input)

    # Compute error
    error = outputs - predicted_output

    # Backpropagation
    d_predicted_output = error * tanh_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * tanh_derivative(hidden_layer_activation)

    # Update weights and biases
    output_weights += hidden_layer_activation.T.dot(d_predicted_output) * learning_rate
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    hidden_weights += inputs.T.dot(d_hidden_layer) * learning_rate
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

# Test the model
print("Trained Weights (Hidden Layer):\n", hidden_weights)
print("Trained Weights (Output Layer):\n", output_weights)
print("Output after training:\n", predicted_output)
