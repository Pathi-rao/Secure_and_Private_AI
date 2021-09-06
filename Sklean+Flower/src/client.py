# do the necessary imports
import warnings
import flwr as fl
import numpy as np
import datahandler as dh

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import utils

# dataset info
"""
    https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
    Objective: Build a network intrusion detector, a predictive model 
    capable of distinguishing between ``bad'' connections, called intrusions or attacks, and ``good'' normal connections. This database contains a standard 
    set of data to be audited, which includes a wide variety of intrusions simulated in a military network environment.

    """

X_train, X_test, y_train, y_test = dh.data_processor()

#print(X_train.shape, y_train.shape)

# Split train set into 10 partitions and randomly use one for training.
partition_id = np.random.choice(10)
(X_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]

#print(X_train.shape, y_train.shape)


""" 
    Initialize the logistic regression model in the client. Let's have the client train for just a single iteration in each round by setting max_iter=1.
    Also, don't forget to set warm_start=True, otherwise, the model's parameters get refreshed when we call .fit. We don't want to reset the global parameters 
    sent by the server.

    """

# Create LogisticRegression Model
model = LogisticRegression(
penalty="l2",
max_iter=1, # local epoch
warm_start=True, # prevent refreshing weights when fitting
)

"""
    Next, we have to set the initial parameters of the model since the instance attributes used to save the model's parameters aren't created until .fit is called.
    But the server might want to set them or request them before fitting as is usually the case in federated learning. So, we create the parameter attributes and zero-initialize
    them using utils.set_initial_params(model).

    """
# Setting initial parameters, akin to model.compile for keras models
utils.set_initial_params(model)

"""
    Now it is time to define the Flower client. The client is derived from the class fl.client.NumPyClient. It needs to define the following three methods:

    get_parameters : Returns the current local model paramters. The utility function get_model_parameters does this for us.
    
    fit: Defines the steps to train the model on the locally held dataset. It also receives global model parameters and other configuration information from the server. 
    We update the local model's parameters using the received global parameters using utils.set_model_params(model, parameters) and train it on the local dataset. 
    This method also sends back the local model's parameters after training, the size of the training set and a dict communicating arbitrary values back to the server.

    evaluate: This method is meant for evaluating the provided parameters using a locally held dataset. It returns the loss along with other details such as the size 
    of the test set, accuracy, etc., back to the server. Here, we calculate the loss value of the model explicitly using sklearn.metrics.log_loss. This is done explicitly 
    because there is no public attribute in LogisticRegression that saves the loss value like, for instance, a TensorFlow model's history. Make sure to use the proper
    loss function corresponding to your model.

    """


class KDDClient(fl.client.NumPyClient):
    def get_parameters(self): # type: ignore
        return utils.get_model_parameters(model)

    def fit(self, parameters, config): # type: ignore
        utils.set_model_params(model, parameters)
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
            print(f"Training finished for round {config['rnd']}")
        return utils.get_model_parameters(model), len(X_train), {}

    def evaluate(self, parameters, config): # type: ignore
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, len(X_test), {"accuracy": accuracy}

    def disconnect (self):
        fl.common.typing.Disconnect('disconnecting...')

fl.client.start_numpy_client("localhost:8080", client=KDDClient())
