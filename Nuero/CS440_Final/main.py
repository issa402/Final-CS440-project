import numpy as np
import neuralnet
from perceptron import Perceptron, train_perceptron
import matplotlib.pyplot as plt

def parse_data(file_path: str) -> np.ndarray:
    data_set = []
    with open(file_path) as f:
        lines = f.readlines()
        for line in range(0, len(lines), 28):
            img = []
            for i in range(0,28):
                for n in range(0, 28):
                    match lines[line + i][n]:
                        case " ":
                            img.append(0)
                        case "\n":
                            img.append(0)
                        case _:
                            img.append(1)
            data_set.append(img)
    return np.array(data_set)

def parse_labels(file_path: str) -> np.ndarray:
    with open(file_path) as f:
        lines = f.readlines()
        labels = np.array([int(line.replace("\n", "")) for line in lines])

    return labels

def print_data_index(data):
    plt.imshow(data.reshape(28, 28), cmap='gray')
    plt.title("Data")
    plt.show()




def perceptron_test(training_set, training_labels, validation_set,validation_labels, test_set, test_labels):


    p = [Perceptron(learning_rate=0.01, n_iter=100) for _ in range(10)]
    for i in range(10):
        p[i] = train_perceptron(p[i], i, training_set, training_labels, validation_set, validation_labels)
        print(f"Perceptron {i} trained.")



    # Test the perceptrons

    for i in range(10):
        accuracy = 0
        for j in range(len(test_set)):
            test_predicted = p[i].predict(test_set[j])
            accuracy += 1 if test_predicted == (test_labels[j] == i) else 0
        accuracy /= len(test_set)
        print(f"Perceptron {i} test accuracy: {accuracy * 100:.2f}%")

def netural_network_test(training_set, training_labels, validation_set,validation_labels, test_set, test_labels):

    n = neuralnet.NeturalNetwork()
    n.add_layer(784,'sigmoid')
    n.add_layer(20,'sigmoid')
    n.add_layer(20,'sigmoid')
    n.add_layer(10)
    # for i in range(len(training_set)):
    #     current_data = training_set[i]
    #     output = n.predict(current_data)
    #     print(f"Predicted: {output}, Expected: {training_labels[i]} ")

    print(f"training set shape: {training_set.shape}, training labels shape: {training_labels.shape}")
    n.print_weights()
    n.fit_dataset(training_set, training_labels, validation_set, validation_labels, batch_size=1)

    # Test the neural network
    accuracy = 0
    for i in range(len(test_set)):
        test_predicted = n.predict(test_set[i])
        accuracy += 1 if np.argmax(test_predicted) == test_labels[i] else 0
    accuracy /= len(test_set)
    print(f"Neural Network test accuracy: {accuracy * 100:.2f}%")


def main():
    data_dir = "./data/"

    # prepare the digits dataset
    training_set = parse_data(data_dir + "digitdata/trainingimages")
    training_labels = parse_labels(data_dir + "digitdata/traininglabels")
    if len(training_set) != len(training_labels):
        print("Data set and labels length mismatch")
        return

    validation_set = parse_data(data_dir + "digitdata/validationimages")
    validation_labels = parse_labels(data_dir + "digitdata/validationlabels")
    if len(validation_set) != len(validation_labels):
        print("Validation set and labels length mismatch")
        return

    test_data = parse_data(data_dir + "digitdata/testimages")
    test_labels = parse_labels(data_dir + "digitdata/testlabels")
    if len(test_data) != len(test_labels):
        print("Test data set githand labels length mismatch")
        return


    # perceptron_test(training_set, training_labels, validation_set, validation_labels, test_data, test_labels)
    netural_network_test(training_set, training_labels, validation_set, validation_labels, test_data, test_labels)










if __name__ == "__main__":
    main()
