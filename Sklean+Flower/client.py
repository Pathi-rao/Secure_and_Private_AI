# do the necessary imports
import warnings
import flwr as fl
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import utils

# dataset 
"""
    https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
    This is the data set used for The Third International Knowledge Discovery and Data Mining Tools Competition, which was held in conjunction with KDD-99 
    The Fifth International Conference on Knowledge Discovery and Data Mining. The competition task was to build a network intrusion detector, a predictive model 
    capable of distinguishing between ``bad'' connections, called intrusions or attacks, and ``good'' normal connections. This database contains a standard 
    set of data to be audited, which includes a wide variety of intrusions simulated in a military network environment.

    """

