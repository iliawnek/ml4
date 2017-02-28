import numpy as np
import csv
import multiprocessing as mp

#############
# Load data #
#############

X = []
T = []

with open('epi_stroma_data.tsv', 'rb') as tsv:
    i = 0
    for line in csv.reader(tsv, delimiter='\t'):
        if i != 0:
            X.append([float(x) for x in line[1:]])
            T.append([int(line[0])])
        i += 1

X = np.array(X)
T = np.array(T)

data_point_count = X.shape[0]
feature_count = X.shape[1]
fold_count = 23
fold_size = data_point_count / fold_count

###########################################
# K-nearest neighbours + cross-validation #
###########################################

# def classify_knn(X, T, x_new, K):
#     # compute distance-class mappings
#     distances = []
#     for i, x in enumerate(X):
#         distances.append((np.linalg.norm(x_new - X[i]), T[i][0]))
#
#     # find most popular class in K-nearest neighbours
#     distances = sorted(distances, key=lambda d: d[0])
#     epi_votes = 0
#     stroma_votes = 0
#     for k in range(K):
#         if distances[k][1] == 1:
#             epi_votes += 1
#         else:
#             stroma_votes += 1
#     if epi_votes >= stroma_votes:
#         return 1
#     else:
#         return 2
#
# min_K = 200
# max_K = 200  # K from 1 to max_K
# step_K = 10
# errors = {}
#
# for K in range(min_K, max_K + 1, step_K):
#     error_count = 0
#     for fold in range(fold_count):
#         print 'K = %d, fold = %d' % (K, fold + 1)
#
#         # divide folds into validation and training data
#         lower = fold_size * fold
#         upper = fold_size * (fold + 1)
#         X_fold = X[lower:upper]
#         X_train = np.delete(X, np.arange(lower, upper, 1), 0)
#         T_fold = T[lower:upper]
#         T_train = np.delete(T, np.arange(lower, upper, 1), 0)
#
#         def count_error(i):
#             classification = classify_knn(X_train, T_train, X_fold[i], K)
#             if classification != T_fold[i][0]:
#                 return 1
#             return 0
#
#         pool = mp.Pool(4)
#         results = pool.map(count_error, range(fold_size))
#         error_count += sum(results)
#
#     average_error = float(error_count) / float(fold_count)
#     print '---'
#     print 'K = %d, error = %f' % (K, average_error)
#     print '---'
#     errors[K] = average_error
#
# print errors

##########################
# Naive Bayes classifier #
##########################

def bayes(X, T):

    def classify_bayes(X, T, x_new, parameters):
        prob = {}
        for cl in parameters:
            prob[cl] = parameters[cl]['prior']
            for i, m in enumerate(parameters[cl]['mean']):
                vari = parameters[cl]['vars'][i]
                prob[cl] *= 1.0 / np.sqrt(2.0 * np.pi * vari)
                prob[cl] *= np.exp((-0.5 / vari) * (x_new[i] - m) ** 2)
        if prob[1] > prob[2]:
            return 1
        else:
            return 2

    error_count = 0

    for fold in range(fold_count):
        # divide folds into validation and training data
        lower = fold_size * fold
        upper = fold_size * (fold + 1)
        X_fold = X[lower:upper]
        X_train = np.delete(X, np.arange(lower, upper, 1), 0)
        T_fold = T[lower:upper]
        T_train = np.delete(T, np.arange(lower, upper, 1), 0)

        # calculate Gaussian parameters
        parameters = {}
        for cl in [1, 2]:
            data_pos = np.where(T_train == cl)[0]
            class_pars = {}
            class_pars['mean'] = X_train[data_pos, :].mean(axis=0)
            class_pars['vars'] = X_train[data_pos, :].var(axis=0)
            class_pars['prior'] = 1.0 * len(data_pos) / len(X_train)
            parameters[cl] = class_pars

        predictions = np.zeros((400,))

        def count_error(i):
            classification = classify_bayes(X_train, T_train, X_fold[i], parameters)
            if classification != T_fold[i][0]:
                return 1
            return 0

        # pool = mp.Pool(4)
        # results = pool.map(count_error, range(fold_size))
        # error_count += sum(results)

        for i in range(fold_size):
            error_count += count_error(i)

    average_error = float(error_count) / float(fold_count)
    return average_error

# bayes(X, T)

#####################
# Feature selection #
#####################

# Calculate performance of each feature in isolation.
feature_errors = []
for i in range(feature_count):
    feature = X[:,[i]]
    feature_errors.append((i, bayes(feature, T)))

feature_errors = sorted(feature_errors, key=lambda d: d[1])
import pprint
pp = pprint.PrettyPrinter()
pp.pprint(feature_errors)