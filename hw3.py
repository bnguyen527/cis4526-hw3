import itertools
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ID = 'tug21976'
MAX_ITERS = 100000
SQRT_EPS = math.sqrt(np.finfo(float).eps)   # Squared root of machine epsilon
LEARN_RATES = [1e-5, 1e-4, 1e-3]
LAMBDA_VALUES = [0.1, 0.5, 1.0, 2.0]


# Calculate hinge loss given training and prediction label vectors.
def hinge_loss(train_y, pred_y):
    return np.maximum(np.zeros(train_y.size), 1 - train_y*pred_y).sum()


# Calculate squared loss given training and prediction label vectors.
def squared_loss(train_y, pred_y):
    return np.sum((1-train_y*pred_y)**2)    # Squared loss in classification context


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
    train_x_padded = padding(train_x)   # Add artificial coordinates of 1 at the end of train_x
    weights = 0.2 * np.random.randn(train_x_padded.shape[1])    # Initialize to small weights
    gradient = num_gradient(calc_loss_with_reg, weights, loss, lambda_val, regularizer, train_x, train_y)   # Calculate numerical gradient
    step_size = learn_rate * gradient   # Calculate step size
    epoch = 0   # For time-keeping as well as monitoring progress
    # costs = []
    cost = calc_loss_with_reg(weights, loss, lambda_val, regularizer, train_x, train_y) # Calculate loss for monitoring
    # costs.append(cost)
    print('Epoch 0: loss {}'.format(cost))  # Display epoch and loss for monitoring
    # # Only calculating for the first 5000 epochs in learn_rate_selection
    # while not np.allclose(step_size, np.zeros(weights.size), atol=1e-4) and epoch < 5000:
    # Stop when step size is lower than absolute tolerance or when epoch has exceeded max number of iterations
    while not np.allclose(step_size, np.zeros(weights.size), atol=1e-4) and epoch < MAX_ITERS:
        weights = weights - step_size   # Update weights
        epoch += 1
        if epoch % 100 == 0:    # Display epoch and loss every 100 epochs
            cost = calc_loss_with_reg(weights, loss, lambda_val, regularizer, train_x, train_y) # Calculate loss for monitoring
            # costs.append(cost)
            print('Epoch {}: loss {}'.format(epoch, cost))
        gradient = num_gradient(calc_loss_with_reg, weights, loss, lambda_val, regularizer, train_x, train_y)
        step_size = learn_rate * gradient   # Update step size
    # cost = calc_loss_with_reg(weights, loss, lambda_val, regularizer, train_x, train_y) # Calculate loss for monitoring
    # costs.append(cost)
    return weights#, np.asarray(costs), epoch


# Return prediction label vectors given vector of learned classifier weights
# and test examples.
def test_classifier(w, test_x):
    activation = test_regressor(w, test_x).reshape((test_x.shape[0], 1))    # Activations for classification
    return np.apply_along_axis(lambda y: 1 if y > 0 else -1, 1, activation) # Predict based on activations


# Return a float between 0.0 and 1.0 representing the classification accuracy.
def compute_accuracy(test_y, pred_y):
    return (test_y == pred_y).sum() / test_y.size


# Return a string representing author's Temple AccessNet.
def get_id():
    return ID


# Return a set of accuracies from n-fold cross-validation for a certain set of
# hyperparameters, loss functions, and regularizers.
def n_fold_cross_validation(num_folds, train_x, train_y, learn_rate, loss, lambda_val=1.0, regularizer=None):
    shuffled_indices = np.random.permutation(range(train_x.shape[0]))   # randomize
    split_indices = np.array_split(range(train_x.shape[0]), num_folds)  # n-fold
    accuracies = []  # performance over n folds
    for i in range(num_folds):
        # Validation set
        valid_x_split = train_x[shuffled_indices][split_indices[i]]
        valid_y_split = train_y[shuffled_indices][split_indices[i]]
        # Train set
        train_x_split = np.delete(train_x, split_indices[i], 0)
        train_y_split = np.delete(train_y, split_indices[i])
        train_x_split_normalized, valid_x_split_normalized = normalize(train_x_split, valid_x_split)    # Normalize training and test examples
        weights = train_classifier(train_x_split_normalized, train_y_split, learn_rate, loss, lambda_val, regularizer)
        pred_y = test_classifier(weights, valid_x_split_normalized)
        accuracies.append(compute_accuracy(valid_y_split, pred_y))
    return accuracies


# Display loss values with epoch for different learning rates, for manual
# selection.
def learn_rate_selection(train_x, train_y, loss, lambda_val=1.0, regularizer=None):
    for learn_rate in LEARN_RATES:
        print('Learning rate {}:\n'.format(learn_rate))
        train_classifier(train_x, train_y, learn_rate, loss, lambda_val, regularizer)
        print('\n')


# Display basic statistics as well as t-test results for an array of possible
# values of lambda, based on n-fold cross-validation.
def lambda_selection(num_folds, train_x, train_y, learn_rate, loss, regularizer):
    lambda_candidates = dict()
    for lambda_val in LAMBDA_VALUES:
        accuracies = n_fold_cross_validation(num_folds, train_x, train_y, learn_rate, loss, lambda_val, regularizer)
        lambda_candidates[lambda_val] = accuracies
    print('{}-Fold Cross-Validation'.format(num_folds))    
    for lambda_val in LAMBDA_VALUES:
        accuracies = lambda_candidates[lambda_val]
        print('Accuracies for {}: {}'.format(lambda_val, accuracies))
        print('Mean: {}\tStandard Deviation: {}'.format(np.mean(accuracies), np.std(accuracies)))
    # Each combination to compare
    for lambda1, lambda2 in itertools.combinations(LAMBDA_VALUES, 2):
        # Only take p-value
        print('p-value for t-test between {} and {}: {}'.format(lambda1, lambda2, stats.ttest_ind(
            lambda_candidates[lambda1], lambda_candidates[lambda2], equal_var=False)[1]))


# Return only the part of the data used for binary classification as in
# specifications, with labels converted into +1/-1 values.
def preprocess(df):
    # Data set only contains quality values in range 3-9, and only ranges 3-5 and 7-9 are used for binary classification.
    df_processed = df[df['quality'] != 6].copy()
    quality = df_processed['quality'].values.reshape((df_processed.shape[0], 1))    # Take only 'quality' column
    df_processed.loc[:, 'quality'] = np.apply_along_axis(lambda q: 1 if q > 6 else -1, 1, quality)  # Turn raw values into +1/-1 labels
    return df_processed


# Detect and remove outliers that are more than k standard deviation from the
# mean of that dimension. Default for k is 6.
def detect_outliers(data, k=6):
    # Since this is raw data, only detect outliers in the feature dimensions and exclude labels
    for i in range(data.shape[1] - 1):
        col = data[:, i]
        data = data[np.absolute(col-np.mean(col)) <= k*np.std(col)] # Only keep examples that are within k std from mean
    return data


# Return a balanced oversampled version of training data.
def oversampling(train_x, train_y):
    label_counts = np.unique(train_y, return_counts=True)   # Get counts of classes
    majority = train_x[train_y == label_counts[0][0]]   # All existing examples of majority class
    minority = train_x[train_y == label_counts[0][1]]   # All existing examples of minority class
    # Indices for random sampling with replacement from the minority class to balance with the majority
    sample_indices = np.random.choice(range(label_counts[1][1]), label_counts[1][0], replace=True)
    # Combine the majority and oversamples of minority class to have balanced data set
    train_x_balanced = np.append(majority, minority[sample_indices], axis=0)
    # Orders are known, so labels are just same number of majority labels and minority labels
    train_y_balanced = np.append(label_counts[0][0] * np.ones(label_counts[1][0]),
                                 label_counts[0][1] * np.ones(label_counts[1][0]))
    return train_x_balanced, train_y_balanced   # Return balanced data set


# Normalize training and test examples by centering and variance scaling.
def normalize(train_x, test_x):
    mean, std = calc_mean_and_std(train_x)  # Calculate mean of training examples
    return center_and_var_scale(train_x, mean, std), center_and_var_scale(test_x, mean, std)    # Normalize training and test examples both based on training's mean and std


# Calculate mean and standard deviation of each column of train_x for testing.
def calc_mean_and_std(train_x):
    return np.apply_along_axis(np.mean, 1, train_x), np.apply_along_axis(np.std, 1, train_x)


# Normalize training examples by centering and variance scaling.
def center_and_var_scale(X, mean, std):
    # This uses apply_along_axis on an index array
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
    cost = calculate_loss(w, loss, train_x, train_y)
    if regularizer is not None: # If used with regularizer
        cost += lambda_val*regularizer(w)
    return cost


# Calculate the numerical gradient of func with arguments params.
def num_gradient(func, params, *args):
    indices = np.arange(params.size).reshape((params.size, 1))
    # This uses apply_along_axis on indices array
    return np.apply_along_axis(lambda i: part_num_diff(func, params, i, *args), 1, indices).flatten()


# Perform partial numerical differentiation on func on the index-th dimension
# with arguments params.
def part_num_diff(func, params, index, *args):
    h = SQRT_EPS*params[index]  # Calculate small increment based on machine epsilon
    inc_params = np.copy(params)
    inc_params[index] += h
    return (func(inc_params, *args)-func(params, *args)) / h


# Display accuracies and basic statistics for the linear method specified.
def display_accuracies(method, accuracies):
    print('Accuracies for {}: {}'.format(method, accuracies))
    print('Mean: {}\tStandard Deviation: {}'.format(np.mean(accuracies), np.std(accuracies)))


def main():
    df = pd.read_csv("winequality-white.csv", sep=';')
    df_processed = preprocess(df)   # Process data
    data = df_processed.values  # Turn Pandas DataFrame into Numpy n-dimensional array
    data_no_outliers = detect_outliers(data)    # Outlier detection
    X = data_no_outliers[:, :-1]
    Y = data_no_outliers[:, -1]
    X_balanced, Y_balanced = oversampling(X, Y) # Oversampling for balanced data

    # sample_indices = np.random.choice(range(X_balanced.shape[0]), 100, replace=False)  # random subsampling
    # sample_x = X_balanced[sample_indices]
    # sample_y = Y_balanced[sample_indices]
    # # Following we will call train_classifier directly without cross-validation, so we need to normalize examples.
    # mean, std = calc_mean_and_std(sample_x)
    # sample_x_normalized = center_and_var_scale(sample_x, mean, std)
    
    # print('Learn rate selection for Soft-Margin SVM:\n')
    # learn_rate_selection(sample_x_normalized, sample_y, hinge_loss, regularizer=l2_reg)
    # print('Learn rate selection for Logistic Regression:\n')
    # learn_rate_selection(sample_x_normalized, sample_y, logistic_loss)
    svm_learn_rate = 1e-4   # Chosen from learning rate selection
    log_regr_learn_rate = 1e-4    # Chosen from learning rate selection

    num_folds = 5
    # print('Lambda selection for Soft-Margin SVM:\n')
    # lambda_selection(num_folds, sample_x, sample_y, svm_learn_rate, hinge_loss, l2_reg)
    # print('Above was lambda selection for Soft-Margin SVM')
    svm_lambda = 0.5    # Chosen from lambda selection
        
    start = time.time() # For calculating running time
    svm_accuracies = n_fold_cross_validation(num_folds, X_balanced, Y_balanced, svm_learn_rate, hinge_loss, svm_lambda, l2_reg)
    svm_time = time.time() - start
    print('Finished {}-fold cross-validation for Soft-Margin SVM\n'.format(num_folds))
    display_accuracies('Soft-Margin SVM', svm_accuracies)
    print('Time: {} s'.format(svm_time))
    
    start = time.time()
    log_regr_accuracies = n_fold_cross_validation(num_folds, X_balanced, Y_balanced, log_regr_learn_rate, logistic_loss)
    log_regr_time = time.time() - start
    print('Finished {}-fold cross-validation for Logistic Regression\n'.format(num_folds))
    display_accuracies('Logistic Regression', log_regr_accuracies)
    print('Time: {} s'.format(log_regr_time))

    # shuffled_indices = np.random.permutation(range(X_balanced.shape[0]))   # randomize
    # split_idx = (num_folds-1)*X_balanced.shape[0] // num_folds    # to recreate a fold
    # X_split = X_balanced[shuffled_indices][:split_idx, :]
    # Y_split = Y_balanced[shuffled_indices][:split_idx]
    # mean, std = calc_mean_and_std(X_split)
    # X_split_normalized = center_and_var_scale(X_split, mean, std)   # normalize the split examples
    # # Train Soft-Margin SVM to get losses
    # _, costs, last_epoch = train_classifier(X_split_normalized, Y_split, svm_learn_rate, hinge_loss, svm_lambda, l2_reg)
    # epochs = 100 * np.arange(costs.size - 1)    # Arrange epochs
    # epochs = np.insert(epochs, epochs.size, last_epoch) # Add final epoch
    # # Plot loss as a function of epoch
    # plt.plot(epochs, costs)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.show()


if __name__ == '__main__':
    main()
