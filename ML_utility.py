import sklearn.datasets
import numpy as np


def generate_data(dataset_type, **kwargs):
    # make a list with all the dataset from sklearn.datasets
    dataset_list = [dataset for dataset in dir(sklearn.datasets) if not dataset.startswith('_')]
    if dataset_type not in dataset_list:
        raise ValueError(f"dataset_type must be in {dataset_list}")
    X, y = getattr(sklearn.datasets, dataset_type)(**kwargs)

    return X, y


def GridCompute2D(X, indexes: list = [0, 1], n_points: int = 100):
    """

    :param X: Data to compute grid
    :param indexs: The indexs (NEEDS 2)
    :param n_points: number of points generated from linspace
    :return:
    """

    if len(indexes) != 2:
        return "Needs at least 2 indexes"
    feature_0, feature_1 = np.meshgrid(
        np.linspace(X[:, indexes[0]].min(), X[:, indexes[0]].max(), n_points),
        np.linspace(X[:, indexes[1]].min(), X[:, indexes[1]].max(), n_points)
    )
    grid = np.vstack([feature_0.ravel(), feature_1.ravel()]).T
    return grid, feature_0, feature_1

