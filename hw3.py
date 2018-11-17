import numpy as np


def hinge_loss(train_y, pred_y):
    return None


def squared_loss(train_y, pred_y):
    return None


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
