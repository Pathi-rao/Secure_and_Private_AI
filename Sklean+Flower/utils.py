from typing import Tuple, Union, List
import numpy as np
from sklearn.linear_model import LogisticRegression

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


# The get_model_parameters function returns the model parameters. These are found in the coef_ and intercept_ attributes for LogisticRegression .
def get_model_parameters(model):
    """Returns the paramters of a sklearn LogisticRegression model"""
    if model.fit_intercept:
        params = (model.coef_, model.intercept_) # ????
    else:
        params = (model.coef_,)
    return params


# The set_model_params function sets/updates the model's parameters. Here care needs to be taken to set the parameters using the same order/index in 
# which they were returned by get_model_parameters.
def set_model_params(
    model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model"""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


"""
    The function set_initial_params zero-initializes the parameters of the model. This requires prior information about the attribute names, 
    the number of classes and features of your dataset to calculate the size of the parameter matrices of the model. An alternative method for 
    initializing the parameters could be to fit the model using a few dummy samples that mimic the dimensions of the actual dataset.
    """

def set_initial_params(model: LogisticRegression):
    """
    Sets initial parameters as zeros
    """
    n_classes = 23 # threat types
    n_features = 33 # Number of features in dataset
    model.classes_ = np.array([i for i in range(23)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList: # returns list of Xy (read more about function annotations)  ????
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), 
        np.array_split(y, num_partitions))
    )
