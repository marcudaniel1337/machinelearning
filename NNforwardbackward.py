import math
import random

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid (assuming x is output of sigmoid)."""
    return x * (1 - x)

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
        """
        Initialize network with random weights and biases.

        Args:
            input_size (int): Number of input neurons.
            hidden_size (int): Number of neurons in hidden layer.
            output_size (int): Number of output neurons.
            learning_rate (float): Step size for weight updates.
        """
        self.lr = learning_rate

        # Initialize weights with random values
        # input -> hidden weights matrix (hidden_size x input_size)
        self.weights_input_hidden = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
        # hidden -> output weights matrix (output_size x hidden_size)
        self.weights_hidden_output = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(output_size)]

        # Biases for hidden and output layers
        self.bias_hidden = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.bias_output = [random.uniform(-1, 1) for _ in range(output_size)]

    def forward(self, inputs):
        """
        Forward pass: compute activations for hidden and output layers.

        Args:
            inputs (list): Input vector.

        Returns:
            tuple: (hidden layer outputs, final outputs)
        """
        # Calculate hidden layer activations
        hidden_layer_inputs = []
        for i in range(len(self.weights_input_hidden)):
            weighted_sum = sum(w * inp for w, inp in zip(self.weights_input_hidden[i], inputs)) + self.bias_hidden[i]
            hidden_layer_inputs.append(sigmoid(weighted_sum))

        # Calculate output layer activations
        output_layer_outputs = []
        for i in range(len(self.weights_hidden_output)):
            weighted_sum = sum(w * h for w, h in zip(self.weights_hidden_output[i], hidden_layer_inputs)) + self.bias_output[i]
            output_layer_outputs.append(sigmoid(weighted_sum))

        return hidden_layer_inputs, output_layer_outputs

    def backward(self, inputs, hidden_outputs, outputs, targets):
        """
        Backward pass: update weights and biases using error gradients.

        Args:
            inputs (list): Input vector.
            hidden_outputs (list): Outputs from hidden layer.
            outputs (list): Outputs from output layer.
            targets (list): True target values.
        """
        # Calculate output layer error (difference between predicted and actual)
        output_errors = [targets[i] - outputs[i] for i in range(len(targets))]

        # Calculate output layer delta (error * derivative of activation)
        output_deltas = [output_errors[i] * sigmoid_derivative(outputs[i]) for i in range(len(outputs))]

        # Calculate hidden layer errors by propagating output deltas backwards
        hidden_errors = [0] * len(hidden_outputs)
        for i in range(len(hidden_outputs)):
            error_sum = 0
            for j in range(len(output_deltas)):
                error_sum += output_deltas[j] * self.weights_hidden_output[j][i]
            hidden_errors[i] = error_sum

        # Calculate hidden layer deltas
        hidden_deltas = [hidden_errors[i] * sigmoid_derivative(hidden_outputs[i]) for i in range(len(hidden_outputs))]

        # Update weights hidden -> output
        for i in range(len(self.weights_hidden_output)):
            for j in range(len(self.weights_hidden_output[i])):
                change = self.lr * output_deltas[i] * hidden_outputs[j]
                self.weights_hidden_output[i][j] += change

        # Update biases for output layer
        for i in range(len(self.bias_output)):
            self.bias_output[i] += self.lr * output_deltas[i]

        # Update weights input -> hidden
        for i in range(len(self.weights_input_hidden)):
            for j in range(len(self.weights_input_hidden[i])):
                change = self.lr * hidden_deltas[i] * inputs[j]
                self.weights_input_hidden[i][j] += change

        # Update biases for hidden layer
        for i in range(len(self.bias_hidden)):
            self.bias_hidden[i] += self.lr * hidden_deltas[i]

    def train(self, training_data, epochs=1000):
        """
        Train the network on given data.

        Args:
            training_data (list of tuples): Each tuple is (input_vector, target_vector).
            epochs (int): Number of training iterations.
        """
        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in training_data:
                # Forward pass
                hidden_outputs, outputs = self.forward(inputs)

                # Calculate loss (mean squared error)
                loss = sum((targets[i] - outputs[i]) ** 2 for i in range(len(targets))) / len(targets)
                total_loss += loss

                # Backward pass to update weights
                self.backward(inputs, hidden_outputs, outputs, targets)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss / len(training_data):.4f}")

    def predict(self, inputs):
        """
        Predict output for given input.

        Args:
            inputs (list): Input vector.

        Returns:
            list: Network output vector.
        """
        _, outputs = self.forward(inputs)
        return outputs


if __name__ == "__main__":
    # XOR problem (classic simple test)
    training_data = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0]),
    ]

    nn = SimpleNeuralNetwork(input_size=2, hidden_size=2, output_size=1, learning_rate=0.5)
    nn.train(training_data, epochs=2000)

    print("Testing trained network on XOR inputs:")
    for inputs, target in training_data:
        output = nn.predict(inputs)
        print(f"Input: {inputs}, Predicted: {[round(o, 3) for o in output]}, Target: {target}")
