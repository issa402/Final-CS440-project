import numpy as np


def sigmoid(x):
    x = (x - np.mean(x)) / np.std(x)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x: float):
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray):
    return np.where(x > 0, 1.0, 0.0)


def gradient_descent(target, prediction):
    return target - prediction

def mean_squared_error(target, prediction):
    return (prediction - target) ** 2

def mean_squared_error_derivative(target:list[float], prediction:list[float]):
    return 0.5 * mean_squared_error(target, prediction)


class NeturalNetwork:
    def __init__(self):
        self.layers = []
        self.learning_rate = 0.01
        self.tempature = 10

    def add_layer(self, n_neurons, activation='relu'):
        if activation == 'sigmoid':
            activation_func = sigmoid
            activation_derivative = sigmoid_derivative
        elif activation == 'relu':
            activation_func = relu
            activation_derivative = relu_derivative
        else:
            raise ValueError(f"Unsupported activation function: {activation}")


        if len(self.layers) == 0:
            input_size = n_neurons
        else:
            input_size = self.layers[-1]['n_neurons']

        w = np.random.rand(input_size, n_neurons)
        b = np.zeros((1, n_neurons))



        layer = {
            'weights': w,
            'bias': b,
            'activation': activation_func,
            'activation_derivative': activation_derivative,
            'n_neurons': n_neurons,
            'a': [0 * n_neurons] ,  # Activation output
            'input_size': input_size,
            'dw': np.zeros_like(w),  # Gradient of the weights
            'db': np.zeros_like(b),  # Gradient of the bias
        }
        self.layers.append(layer)

    def forward(self, X):
        X = X.reshape(1,-1)
        self.input_data = X  # Store input data for backpropagation
        for layer in self.layers:
            X = layer['activation'](np.dot(X, layer['weights']) + layer['bias'])
            layer['a'] = X  # Store the activation for backpropagation

        return X
    def get_learning_rate(self):
        return self.learning_rate
    def decrease_temp(self):
        self.tempature  *= 0.98
    def backpropagation(self, target, apply=True):
        # Calculate the error for the output layer

        learning_rate = self.get_learning_rate()
        target = target.reshape(1,-1)

        error = self.layers[-1]['a'] - target  # Calculate the error


        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            # Calculate the gradient of the activation function

            da = error * layer['activation_derivative'](layer['a'])
            inputs = self.input_data if i == 0 else self.layers[i - 1]['a']  # Input to the layer

            dW = np.dot(inputs.T, da)  # Gradient of the weights
            db = np.sum(da, axis=0, keepdims=True)  # Gradient of the bias

            if apply:
                # Update the weights and bias
                layer['weights'] += -learning_rate * dW
                layer['bias'] +=  -learning_rate * db
            else:
                # Store the gradients for later use, for batch training

                layer['dw'] += dW
                layer['db'] += db

            # Calculate the error for the next layer
            error = np.dot(da, layer['weights'].T)


    def apply_backpropagation(self):
        learning_rate = self.get_learning_rate()
        for i in range(len(self.layers)):
            layer = self.layers[i]


            layer['weights'] += -learning_rate * layer['dw']
            layer['bias'] += -learning_rate * layer['db']

            # if layer['dw'].shape == (20, 10):
                # print(f"Layer {i} dw: {layer['dw']}, db: {layer['db']}")
                # print (f"Layer {i} weights: {layer['weights']}, bias: {layer['bias']}")

            # Reset the gradients for the next iteration
            layer['dw'] =  np.zeros_like(layer['weights'])
            layer['db'] = np.zeros_like(layer['bias'])

    def make_one_hot(self, y):
        one_hot = np.zeros((len(y), max(y) + 1))
        for i, label in enumerate(y):
            one_hot[i, label] = 1
        return one_hot

    def create_batched_set(self, X,y, batch_percentage=0.2):

        training_set = []
        training_labels = []

        amount_per_batch = int(len(X) * batch_percentage)
        for i in range(0, len(X), amount_per_batch):
            batch = X[i:i + amount_per_batch]
            labels = y[i:i + amount_per_batch]
            training_set.append(batch)
            training_labels.append(labels)
        training_set = np.array(training_set)
        training_labels = np.array(training_labels)

        return training_set, training_labels



    def fit_dataset(self, X, y, validation_set=None, validation_labels=None,epochs=50, batch_size=1):
        y = self.make_one_hot(y)
        validation_labels = self.make_one_hot(validation_labels) if validation_labels is not None else None

        if batch_size > 1:
            X, y = self.create_batched_set(X, y, batch_size)

        for epoch in range(epochs):
            for i in range(len(X)):
                if batch_size == 1:
                    self.fit(X[i], y[i], apply=(batch_size == 1))
                else:
                    for j in range(len(X[i])):
                        self.fit(X[i][j], y[i][j], apply=False)
                    self.apply_backpropagation()
            cost = 0
            if validation_set is not None and validation_labels is not None:
                for i in range(len(validation_set)):
                    validation_predicted = self.predict(validation_set[i])
                    cost += np.sum(mean_squared_error(validation_labels[i], validation_predicted))

                    if epoch % 50 == 0:
                        print(f"Predicted: {validation_predicted[0]}, Expected: {validation_labels[i]} ")
                cost /= len(validation_set)
                standard_d = np.sqrt(cost)

                print(f"Loss: {cost} standard d: {standard_d:.2f} at epoch {epoch}")
            self.decrease_temp()



    def fit(self, X, y, apply=True):
        self.forward(X)
        self.backpropagation(y, apply=apply)



    def predict(self, X):
        return self.forward(X)

    def print_weights(self, to_file=False):
        for i, layer in enumerate(self.layers):
            print(f"Layer {i}: {layer['n_neurons']} neurons : {layer['weights'].shape} weights : {layer['bias'].shape} bias")
            print("Weights:")
            print(layer['weights'])
            print("Bias:")
            print(layer['bias'])
            print()
            print("-" * 30)

        if to_file:
            with open("weights.txt", "w") as f:
                for i, layer in enumerate(self.layers):
                    f.write(f"Layer {i}: {layer['n_neurons']} neurons : {layer['weights'].shape} weights : {layer['bias'].shape} bias\n")
                    f.write("Weights:\n")
                    np.savetxt("weights.txt", layer['weights'], fmt='%.5f')
                    f.write("Bias:\n")
                    np.savetxt("weights.txt", layer['bias'], fmt='%.5f')
                    f.write("\n")
                    f.write("-" * 30 + "\n")
