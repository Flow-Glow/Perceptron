import numpy as np


# respect_to_list = [func_1,func2,...]
def fit(Learning_rate, X, n_features, epoch, respect_to_w, model, y) -> [np.ndarray, np.ndarray, float]:
    """
    :param y: the true labels
    :param Learning_rate: The learning rate of the model
    :param X: The training data
    :param n_features: number of weights in the model
    :param epoch: number of epochs in the training
    :param respect_to_w: The function that compute the gradient with respect to w
    :param model: The model that we want to train on
    :return:
    Weights: The weights of the model
    Y_hat_array: The y-hat of the training process
    AvgError: the average error of the training process

    """
    y_hat_array = np.zeros(len(y))
    err_array = np.zeros(epoch)
    weights = np.random.random(n_features + 1)
    for e in range(epoch):
        err_count = 0

        for idx, x in enumerate(X):
            y_hat = model(weights, x)
            y_hat_array[idx] = y_hat
            if y_hat != y[idx]:
                err_count += 1
                for w in range(len(weights)):
                    weights[w] = weights[w] - (Learning_rate * respect_to_w(y[idx], x[w]))
                # weights[0] = weights[0] - (Learning_rate * respect_to_w(y[idx], 1))
                # weights[1] = weights[1] - (Learning_rate * respect_to_w(y[idx], x[0]))
                # weights[2] = weights[2] - (Learning_rate * respect_to_w(y[idx], x[1]))
        err_array[e] = err_count / len(y)
    AvgError = np.mean(err_array)

    return weights, y_hat_array, AvgError
