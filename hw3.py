import numpy as np


# Calculate hinge loss given training and prediction label vectors.
def hinge_loss(train_y, pred_y):
    return np.maximum(np.zeros(train_y.size), np.ones(train_y.size) - train_y*pred_y)


# Calculate squared loss given training and prediction label vectors.
def squared_loss(train_y, pred_y):
    return (train_y-pred_y)**2


def logistic_loss(train_y, pred_y):
    return None


def l1_reg(w):
    return None


def l2_reg(w):
    return None


def train_classifier(train_x, train_y, learn_rate, loss, lambda_val, regularizer):
    return None


def test_classifier(w, test_x):
    return None


# Return a float between 0.0 and 1.0 representing the classification accuracy.
def compute_accuracy(test_y, pred_y):
    return (test_y == pred_y).sum() / test_y.size


# Return a string representing author's Temple AccessNet.
def get_id():
    return 'tug21976'


def main():
    return None


if __name__ == "__main__":
    main()
