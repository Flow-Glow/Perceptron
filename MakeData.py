import sklearn.datasets


def generate_data(dataset_type, n_features=None, n_samples=None, random_state=None, n_informative=None):
    # make a list with all the dataset from sklearn.datasets
    dataset_list = [dataset for dataset in dir(sklearn.datasets) if not dataset.startswith('_')]
    # check if the dataset type is in the list\
    if dataset_type in dataset_list:
        X, y = getattr(sklearn.datasets, dataset_type)(n_samples=n_samples, n_features=n_features,
                                                           random_state=random_state)

        return X, y
    else:
        raise ValueError(f"dataset_type must be in {dataset_list}")
