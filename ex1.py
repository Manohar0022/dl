import numpy as np

class Perceptron:
    def __init__(self, input_dim, learning_rate=0.01, epochs=10):
        self.weights=np.array([0.1,0.1])
        self.bias = -0.7
        self.learning_rate = learning_rate
        self.max_epochs = epochs

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation_function(linear_output)

    def train(self, X, y):
        for mepoch in range(self.max_epochs):
            error_count = 0
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                update = self.learning_rate * (target - prediction)
                self.weights += update * xi
                self.bias += update
                error_count += int(update != 0)

            if error_count == 0:  # Converged
                break

# Example Usage
if __name__ == "__main__":
    # Example dataset (AND gate)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # AND gate outputs

    perceptron = Perceptron(input_dim=2, learning_rate=0.1, epochs=10)
    perceptron.train(X, y)

    print("Trained Weights:", perceptron.weights)
    print("Trained Bias:", perceptron.bias)

    for xi in X:
        print(f"Input: {xi}, Prediction: {perceptron.predict(xi)}")
