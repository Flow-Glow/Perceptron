import matplotlib.pyplot as plt
import numpy as np

import MakeData
import SGD


def compute_grad_with_respect_to_W(x, y):
    return -x * y


def Perceptron_model(weights, point):
    z = (weights[0]*point[0]) + (point[1] * weights[1]) + (point[2] * weights[2])
    return 1 if z >= 0 else -1


def plot_data(X, AvgError, y_hat_array, weights):
    plt.scatter(X[:, 1], X[:, 2], c=list(y_hat_array))
    xx = np.linspace(-10, 10)
    slope = -weights[1] / weights[2]
    intercept = -weights[0] / weights[2]
    yy = intercept + (slope * xx)
    plt.plot(xx, yy, c='r')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'AvgError:{AvgError}')
    plt.axis("equal")
    plt.grid()

    plt.show()


if __name__ == '__main__':
    # ======== HYPER PARAMETERS ======== #
    n_features = 2
    learning_rate = 1
    epoch = 100
    # ================================== #

    X, y = MakeData.generate_data("make_blobs", n_features=n_features, n_samples=500, random_state=0, n_informative=2)
    y = np.where(y == 0, -1, y)
    ones_array = np.ones((X.shape[0], 1))
    #add a column of 1s to the dataset
    X=np.column_stack((ones_array, X))
    print(X)
    weights, y_hat_array, AvgError = SGD.fit(learning_rate, X, n_features, epoch, compute_grad_with_respect_to_W,
                                             Perceptron_model, y)
    print(f"weights:{weights}, AvgError:{AvgError}, y_hat_array:{y_hat_array} ")

    plot_data(X, AvgError, y_hat_array, weights)
