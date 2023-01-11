import numpy as np
import ML_utility
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt


class PerceptronClassifier:
    def __init__(self, learning_rate=0.1, epoch=2000, n_features=2, n_samples=100):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.n_features = n_features
        self.n_samples = n_samples

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        self.weights = np.random.random(self.n_features + 1)
        # Check that X and y have correct shape
        for _ in range(self.epoch):
            for idx, x in enumerate(X):
                z = np.dot(self.weights, x)
                y_hat = 1 if z >= 0 else -1
                if y_hat != y[idx]:
                    for w in range(len(self.weights)):
                        self.weights[w] = self.weights[w] - (self.learning_rate * (-y[idx] * x[w]))

        return self

    def predict(self, X):
        """

        :param X:
        :return:
        """
        Test_predictions = np.zeros(len(X))
        for i in range(len(X)):
            z = np.dot(self.weights, X[i])
            Test_predictions[i] = 1 if z >= 0 else -1
        return Test_predictions

    def score(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        return f"weights:{self.weights}", f"score:{np.sum(self.predict(X) == y) / len(y) * 100}%"

    def plot(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        grid, f0, f1 = ML_utility.GridCompute2D(X)
        Ones_grid = np.column_stack((np.ones((grid.shape[0], 1)), grid))
        y_pred = np.reshape(self.predict(Ones_grid), f0.shape)
        display = DecisionBoundaryDisplay(
            xx0=f0, xx1=f1, response=y_pred
        )
        display.plot()
        display.ax_.scatter(
            X[:, 0], X[:, 1], c=y, edgecolor="black"
        )
        plt.grid(True)
        plt.axis('equal')
        plt.show()


class LogisticClassifier:

    def __init__(self, learning_rate=0.1, epoch=2000, n_features=2, n_samples=100):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.n_features = n_features
        self.n_samples = n_samples

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        self.weights = np.random.random(self.n_features + 1)
        for _ in range(self.epoch):
            for idx, x in enumerate(X):
                z = np.dot(self.weights, x)
                y_hat = 1 / (1 + np.exp(-z))
                if y_hat != y[idx]:
                    for w in range(len(self.weights)):
                        self.weights[w] = self.weights[w] - (self.learning_rate * ((y_hat - y[idx]) * x[w]))

        return self

    def predict(self, X):
        """

        :param X:
        :return:
        """
        Test_predictions = np.zeros(len(X))
        for i in range(len(X)):
            z = np.dot(self.weights, X[i])
            Test_predictions[i] = 1 / (1 + np.exp(-z))
        return Test_predictions

    def score(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        Test_predictions = self.predict(X)
        error = -np.sum(y * np.log(Test_predictions) + (1 - y) * np.log(1 - Test_predictions))
        return f"weights:{self.weights}", f"Error:{error}"

    def plot(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        grid, f0, f1 = ML_utility.GridCompute2D(X)
        Ones_grid = np.column_stack((np.ones((grid.shape[0], 1)), grid))
        y_pred = np.reshape(self.predict(Ones_grid), f0.shape)
        display = DecisionBoundaryDisplay(
            xx0=f0, xx1=f1, response=y_pred
        )
        display.plot()
        display.ax_.scatter(
            X[:, 0], X[:, 1], c=y, edgecolor="black"
        )
        plt.grid(True)
        plt.axis('equal')
        plt.show()


if __name__ == "__main__":
    n_samples = 1000
    n_features = 2
    X, L_y = ML_utility.generate_data("make_blobs", n_features=n_features, n_samples=n_samples, centers=2,
                                      cluster_std=.1)
    P_y = np.where(L_y == 0, -1, L_y)

    X_ones = np.column_stack((np.ones((X.shape[0], 1)), X))
    perceptron = PerceptronClassifier().fit(X_ones, P_y)
    logistic = LogisticClassifier().fit(X_ones, L_y)
    print(perceptron.score(X_ones, P_y))
    print(logistic.score(X_ones, L_y))
    grid, f0, f1 = ML_utility.GridCompute2D(X)
    perceptron.plot(X, P_y)
    logistic.plot(X, L_y)
