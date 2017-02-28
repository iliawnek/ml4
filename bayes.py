import numpy as np
import csv
import multiprocessing as mp

# Get data from tsv file.

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


# Classify using a Naive Bayes classifier.

def classify(X, T, x_new, parameters):
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


fold_count = 23
fold_size = data_point_count / fold_count
error_count = 0

for fold in range(fold_count):
    print 'fold = %d' % (fold + 1)

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
        classification = classify(X_train, T_train, X_fold[i], parameters)
        if classification != T_fold[i][0]:
            return 1
        return 0


    pool = mp.Pool(4)
    results = pool.map(count_error, range(fold_size))
    error_count += sum(results)

average_error = error_count / fold_count
print '---'
print 'error = %d' % (average_error)
print '---'
