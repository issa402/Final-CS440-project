import matplotlib.pyplot as plt
import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=100):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model has not been trained yet.")
        return 1 if np.dot(X, self.weights) + self.bias >= 0 else 0

    def display(self, X):
        if self.weights is None:
            raise ValueError("Model has not been trained yet.")
        print(f"Input shape: {X.shape}")
        plt.imshow(X.reshape(20,28), cmap='gray')
        plt.title(f"Prediction: {self.predict(X)}")
        plt.show()

    def fit_dataset(self, X,y , validation_set=None, validation_labels=None):
        for i in range(len(X)):
            self.fit(X[i], y[i], validation_set, validation_labels)
        accuracy = 0
        if validation_set is not None and validation_labels is not None:
            for i in range(len(validation_set)):
                validation_predicted = self.predict(validation_set[i])
                if validation_predicted == validation_labels[i] :
                    accuracy += 1

            accuracy /= len(validation_set)
        print(f"Validation accuracy: {accuracy * 100:.2f}%")


    def fit(self, X, y, validation_set=None, validation_labels=None):
        n_features = X.shape
        if self.weights is None:
            self.weights = np.zeros(n_features)
            self.weights.fill(np.random.uniform(-1, 1))
        if self.bias is None:
            self.bias = 0

        for _ in range(self.n_iter):
            y_predicted = self.predict(X)
            # Update weights and bias
            err = y - y_predicted
            if y_predicted == y:
                err = -err
            self.weights += err * X
            self.bias += err
def train_perceptron(p,digit, data_set, labeled_digits, validation_set, validation_labeled_digits):
    labels = []
    for i in range(len(data_set)):
        labels.append(labeled_digits[i] == digit)

    labels = np.array(labels)

    validation_labels = []
    for i in range(len(validation_labeled_digits)):
        validation_labels.append(validation_labeled_digits[i] == digit)

    validation_labels = np.array(validation_labels)

    p.fit_dataset(data_set, labels, validation_set, validation_labels)

    return p
