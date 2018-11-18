import math

import numpy as np


# Calculate hinge loss given training and prediction label vectors.
def hinge_loss(train_y, pred_y):
    return np.maximum(np.zeros(train_y.size), 1 - train_y*pred_y).sum()


# Calculate squared loss given training and prediction label vectors.
def squared_loss(train_y, pred_y):
    return np.sum((1-train_y*pred_y)**2)


# Calculate logistic loss given training and prediction label vectors.
def logistic_loss(train_y, pred_y):
    return np.log(1+np.exp(-train_y*pred_y)).sum() / math.log(2)


# Calculate the value of L1 regularizer given vector of linear classifer weights.
def l1_reg(w):
    return np.absolute(w[:-1]).sum()


# Calculate the value of L2 regularizer given vector of linear classifier weights.
def l2_reg(w):
    return np.sum(w[:-1]**2)


# Return vector of learned linear classifier weights given training data,
# learning rate, loss function, lambda tradeoff parameter, and regularizer.
def train_classifier(train_x, train_y, learn_rate, loss, lambda_val=1.0, regularizer=None):
    train_x_padded = padding(train_x)
    weights = 0.2 * np.random.randn(train_x_padded.shape[1])
    gradient = num_gradient(calc_loss_with_reg, weights, loss, lambda_val, regularizer, train_x, train_y)
    while not np.allclose(gradient, np.zeros(weights.size), atol=1e-12):
        weights = weights - learn_rate*gradient
        gradient = num_gradient(calc_loss_with_reg, weights, loss, lambda_val, regularizer, train_x, train_y)
    return weights


# Return prediction label vectors given vector of learned classifier weights
# and test examples.
def test_classifier(w, test_x):
    activation = test_regressor(w, test_x).reshape((test_x.shape[0], 1))
    return np.apply_along_axis(lambda y: 1 if y > 0 else -1, 1, activation)


# Return a float between 0.0 and 1.0 representing the classification accuracy.
def compute_accuracy(test_y, pred_y):
    return (test_y == pred_y).sum() / test_y.size


# Return a string representing author's Temple AccessNet.
def get_id():
    return 'tug21976'


# Detect and remove outliers that are more than k standard deviation from the
# mean of that dimension. Default for k is 6.
def detect_outliers(train_x, k=6):
    for i in range(train_x.shape[1]):
        col = train_x[:, i]
        train_x = train_x[np.absolute(col-np.mean(col)) <= k*np.std(col)]
    return train_x


# Calculate mean and standard deviation of each column of train_x for testing.
def calc_mean_and_std(train_x):
    return np.apply_along_axis(np.mean, 1, train_x), np.apply_along_axis(np.std, 1, train_x)


# Normalize training examples by centering and variance scaling.
def center_and_var_scale(X, mean, std):
    return np.apply_along_axis(lambda i: (X[:, i]-mean[i]) / std[i], 0, np.arange(X.shape[1]))


# Provide paddings of aritifical coordinates of value 1 as a new column at the
# end of X.
def padding(X):
    return np.append(X, np.ones((X.shape[0], 1)), 1)


# Return regression vector (or activations for classification) given venctor of
# learned weights and test examples.
def test_regressor(w, test_x):
    return np.dot(padding(test_x), w)


# Calculate loss given vector of linear classifier weights, loss function, and
# training data.
def calculate_loss(w, func, train_x, train_y):
    return func(train_y, test_regressor(w, train_x))


# Calculate loss with regularization given vector or linear classifier weights,
# loss function, regularizer, and training data.
def calc_loss_with_reg(w, loss, lambda_val, regularizer, train_x, train_y):
    loss = calculate_loss(w, loss, train_x, train_y)
    if regularizer is not None:
        loss += lambda_val*regularizer(w)
    return loss


# Calculate the numerical gradient of func with arguments params.
def num_gradient(func, params, *args):
    indices = np.arange(params.size).reshape((params.size, 1))
    return np.apply_along_axis(lambda i: part_num_diff(func, params, i, *args), 1, indices).flatten()


# Perform partial numerical differentiation on func on the index-th dimension
# with arguments params.
def part_num_diff(func, params, index, *args):
    sqrt_eps = math.sqrt(np.finfo(float).eps)
    h = sqrt_eps*params[index]
    inc_params = np.copy(params)
    inc_params[index] += h
    return (func(inc_params, *args)-func(params, *args)) / h


def main():
    return None


if __name__ == '__main__':
    main()
