import math

import numpy as np


# Calculate hinge loss given training and prediction label vectors.
def hinge_loss(train_y, pred_y):
    return np.maximum(np.zeros(train_y.size), 1 - train_y*pred_y)


# Calculate squared loss given training and prediction label vectors.
def squared_loss(train_y, pred_y):
    return (train_y-pred_y)**2


# Calculate logistic loss given training and prediction label vectors.
def logistic_loss(train_y, pred_y):
    return np.log(1+np.exp(-train_y*pred_y)) / math.log(2)


# Calculate the value of L1 regularizer given vector of linear classifer weights.
def l1_reg(w):
    return np.absolute(w[:-1]).sum()


# Calculate the value of L2 regularizer given vector of linear classifier weights.
def l2_reg(w):
    return np.sum(w[:-1]**2)


def train_classifier(train_x, train_y, learn_rate, loss, lambda_val, regularizer):
    return None


# Return prediction label vectors given vector of learned classifier weights
# and test examples.
def test_classifier(w, test_x):
    activation = np.dot(padding(test_x), w).reshape((test_x.shape[0], 1))
    return np.apply_along_axis(lambda y: 1 if y > 0 else -1, 1, activation)


# Return a float between 0.0 and 1.0 representing the classification accuracy.
def compute_accuracy(test_y, pred_y):
    return (test_y == pred_y).sum() / test_y.size


# Return a string representing author's Temple AccessNet.
def get_id():
    return 'tug21976'


# Provide paddings of aritifical coordinates of value 1 as a new column at the
# end of X.
def padding(X):
    return np.append(X, np.ones((X.shape[0], 1)), 1)


def main():
    return None


if __name__ == "__main__":
    main()
