from sklearn import svm, preprocessing, metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import fetch_mldata
import os
import GPyOpt
import numpy as np



mnist = fetch_mldata('MNIST original',
                     data_home=os.path.join(os.getcwd(), "data"))

X, y = mnist.data, mnist.target

# Randomly split into 1% train (700) while keeping class balance.
# Feature vector length 784

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.99, random_state=0)
for train_index, test_index in sss.split(X, y):
    X_train = X[train_index]
    y_train = y[train_index]
    print("X_train size: %s, %s" % (X_train.shape))
    print("Y_train size: %s" % (y_train.shape))

# It is usually a good idea to scale the data for SVM training.
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)

nfold = 5


# Objective: Jaccard Similarity
def fit_svc_val(x):
    print("fit_svc_val")
    print(x)
    x = np.atleast_2d(np.exp(x))
    fs = np.zeros((x.shape[0], 1))
    for i in range(x.shape[0]):
        fs[i] = 0
        for n in range(nfold):
            idx = np.array(range(X_train.shape[0]))
            idx_valid = np.logical_and(idx >= X_train.shape[0] / nfold * n,
                                       idx < X_train.shape[0] / nfold * (n + 1))
            idx_train = np.logical_not(idx_valid)
            svc = svm.SVC(C=x[i, 0], gamma=x[i, 1])
            svc.fit(X_train[idx_train], y_train[idx_train])
            # SVC.score defaults to sklearn.metrics.accuracy_score which defaults to
            # sklearn.metrics.jaccard_similarity_score for multiclass model

            # fraction of misclassifications (float)
            fs[i] += metrics.zero_one_loss(y_train[idx_valid],
                                           svc.predict(X_train[idx_valid]))
        fs[i] *= 1. / nfold
    print(fs)
    return sum(fs)

space = GPyOpt.Design_space(space=[{'name': 'C', 'type': 'continuous', 'domain':(-7.,11.)},
                                   {'name': 'gamma', 'type': 'continuous', 'domain': (-12.,3.)}])


objective = GPyOpt.core.task.SingleObjective(fit_svc_val)

X_init2 = np.array([[8.69844261, -4.12113258],
                    [5.39363158, 2.98316543]])


def X_init_k(k):
    return np.array(
        list(X_init2.tolist()+
             GPyOpt.util.stats.initial_design('random', space, k-2).tolist()))
