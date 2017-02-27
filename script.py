import numpy as np
# import pylab as plt
import csv

# Get data from tsv file.

X = []
T = []

with open('epi_stroma_data.tsv', 'rb') as tsv:
    i = 0
    keys = []
    for line in csv.reader(tsv, delimiter='\t'):
        if i != 0:
            X.append([float(x) for x in line[1:]])
            T.append([float(line[0])])
        i += 1

X = np.array(X)
T = np.array(T)

data_point_count = X.shape[0]
feature_count = X.shape[1]

# Perform cross-validation.

def classify(X, t, x_new, K):
    # compute distance-class mappings
    distances = []
    for i, x in enumerate(X):
        distances.append((np.linalg.norm(x_new - X[i]), t[i]))

    # find most popular class in K-nearest neighbours
    distances = sorted(distances, key=lambda d: d[0])
    epi_votes = 0
    stroma_votes = 0
    for k in range(K):
        if distances[k][1] == 1:
            epi_votes += 1
        else:
            stroma_votes += 1
    if epi_votes >= stroma_votes:
        return 1
    else:
        return 2

max_K = 5  # K from 1 to max_K
fold_count = 23
fold_size = data_point_count / fold_count
errors = np.zeros(max_K)

for K in range(1, max_K + 1):
    error_count = 0
    for fold in range(fold_count):
        print 'K = %d, fold = %d' % (K, fold + 1)
        # divide folds into validation and training data
        lower = fold_size * fold
        upper = fold_size * (fold + 1)
        X_fold = X[lower:upper]
        X_train = np.delete(X, np.arange(lower, upper, 1), 0)
        T_fold = T[lower:upper]
        T_train = np.delete(T, np.arange(lower, upper, 1), 0)

        for x, t in zip(X_fold, T_fold):
            classification = classify(X_train, T_train, x, K)
            if classification != t:
                error_count += 1

    errors[K - 1] = error_count / fold_count

print errors