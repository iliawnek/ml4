import numpy as np
import csv

#############
# Load data #
#############

X = []
T = []
feature_names = []

with open('epi_stroma_data.tsv', 'rb') as tsv:
    i = 0
    for line in csv.reader(tsv, delimiter='\t'):
        if i != 0:
            X.append([float(x) for x in line[1:]])
            T.append([int(line[0])])
        else:
            feature_names = line
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

def classify_knn(X, T, x_new, K):
    # compute distance-class mappings
    distances = []
    for i, x in enumerate(X):
        distances.append((np.linalg.norm(x_new - X[i]), T[i][0]))

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

min_K = 200
max_K = 200  # K from 1 to max_K
step_K = 10

for K in range(min_K, max_K + 1, step_K):
    true_epi = 0
    true_stroma = 0
    false_epi = 0
    false_stroma = 0

    for fold in range(fold_count):
        print 'K = %d, fold = %d' % (K, fold + 1)

        # divide folds into validation and training data
        lower = fold_size * fold
        upper = fold_size * (fold + 1)
        X_fold = X[lower:upper]
        X_train = np.delete(X, np.arange(lower, upper, 1), 0)
        T_fold = T[lower:upper]
        T_train = np.delete(T, np.arange(lower, upper, 1), 0)

        for i in range(fold_size):
            classification = classify_knn(X_train, T_train, X_fold[i], K)
            actual_class = T_fold[i][0]
            if classification == actual_class:
                if classification == 1:
                    true_epi += 1
                elif classification == 2:
                    true_stroma += 1
            elif classification != actual_class:
                if classification == 1:
                    false_epi += 1
                elif classification == 2:
                    false_stroma += 1

    error_count = false_epi + false_stroma
    average_error = (float(error_count) / float(fold_count)) / float(fold_size)
    tp = float(true_epi) / float(fold_count)
    tn = float(true_stroma) / float(fold_count)
    fp = float(false_epi) / float(fold_count)
    fn = float(false_stroma) / float(fold_count)
    mcc = ((tp*tn)-(fp*fn)) / (((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**(0.5))
    print '---'
    print 'K = %d' % K
    print '0/1 loss = %f' % average_error
    print 'MCC = %f' % (mcc)
    print '---'


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

    true_epi = 0
    true_stroma = 0
    false_epi = 0
    false_stroma = 0

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

        for i in range(fold_size):
            classification = classify_bayes(X_train, T_train, X_fold[i], parameters)
            actual_class = T_fold[i][0]
            if classification == actual_class:
                if classification == 1:
                    true_epi += 1
                elif classification == 2:
                    true_stroma += 1
            elif classification != actual_class:
                if classification == 1:
                    false_epi += 1
                elif classification == 2:
                    false_stroma += 1


    error_count = false_epi + false_stroma
    average_error = (float(error_count) / float(fold_count)) / float(fold_size)
    tp = float(true_epi) / float(fold_count)
    tn = float(true_stroma) / float(fold_count)
    fp = float(false_epi) / float(fold_count)
    fn = float(false_stroma) / float(fold_count)
    mcc = ((tp*tn)-(fp*fn)) / (((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**(0.5))
    print '---'
    print '0/1 loss = %f' % average_error
    print 'MCC = %f' % (mcc)
    print '---'
    return average_error

bayes(X, T)

#####################
# Feature selection #
#####################

feature_errors = []
for i in range(feature_count):
    print i
    feature = X[:, [i]]
    epi_indices = np.where(T == 1)[0]
    stroma_indices = np.where(T == 2)[0]
    epi_greater = 0.0
    stroma_greater = 0.0
    for epi in epi_indices:
        for stroma in stroma_indices:
            if feature[epi][0] > feature[stroma][0]:
                epi_greater += 1.0
            elif feature[epi][0] < feature[stroma][0]:
                stroma_greater += 1.0
    performance = abs(((epi_greater / (epi_greater + stroma_greater)) - 0.5) * 2.0)
    feature_errors.append((i, performance))
feature_errors = sorted(feature_errors, key=lambda d: d[1])

for j in range(1, feature_count + 1):
    best_features = feature_errors[:j]
    X_best = X[:, best_features]

    print 'average error for top %d features = %f' % (j, bayes(X_best, T))
