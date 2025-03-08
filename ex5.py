import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x) ** 2
def  cne_derivative(y_true, y_pred):
    return y_true  - y_pred  
# XOR truth table
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
y = np.array([[0], [1], [1], [0]])              # Outputs
# Initialize parameters
np.random.seed(42)
input_layer_neurons = 2  # Number of input features
hidden_layer_neurons = 2  # Number of neurons in the hidden layer
output_neurons = 1  # Number of output neurons
# Weights and biases
weights_input_hidden = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_layer_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))
# Training parameters
learning_rate = 0.1
epochs = 10000
# Training the model
for epoch in range(epochs):
    # Forward pass
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = tanh(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = tanh(output_layer_input)
    # Calculate Mean Squared Error (MSE)
   # mse = np.mean((y - predicted_output) ** 2)
    cne= -np.mean(np.sum(y * np.log(predicted_output + 1e-9), axis=1))
    # Backpropagation
    error = cne_derivative(y, predicted_output) 
    d_predicted_output = error * tanh_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * tanh_derivative(hidden_layer_output)

    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    # Print MSE every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, MSE: {cne}")

# Testing the model
print("\nTesting XOR Logic:")
for i in range(len(X)):
    hidden_layer_input = np.dot(X[i], weights_input_hidden) + bias_hidden
    hidden_layer_output = tanh(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = tanh(output_layer_input)
    print(f"Input: {X[i]} => Predicted Output: {predicted_output[0]}, Actual Output: {y[i][0]}")
