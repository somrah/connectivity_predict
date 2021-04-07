"""
Population analysis of the impact of connectivity deficits on behavioral outcomes

Author: Bertrand Thirion, 2021

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import (
    cross_val_score, ShuffleSplit, StratifiedShuffleSplit)


n_permutations = 0
scoring = 'neg_mean_squared_error'

# Redo the thing the data with age
df = pd.read_csv('liste_patients_gliome_final_total_avec_AGE_NSC.csv',
                 index_col=0)
df1 = pd.read_csv('probability.csv', index_col=0)
df2 = pd.read_csv('proportion.csv', index_col=0)

df = df[df.index.astype('str') != 'nan']
df.drop(labels='CorticoThalamic_4', axis=1, inplace=True)
networks = df.columns[:-4].tolist() + df.columns[-1:].tolist()
networks = np.array(networks)

others = df.columns[-5:-4].tolist() + df.columns[-1:].tolist()
X_ = df[others].values

X1 = np.hstack((df1.values, X_))
X2 = np.hstack((df2.values, X_))

# baseline
labels = networks
X = df[networks].values

do_probability = False
do_proportion = False

if do_probability:
    X = X1
    labels = list(df1.columns) + others
if do_proportion:
    X = X2
    labels = list(df2.columns) + others


# get the target
y = df['diff_diff'].values
plt.figure()
plt.hist(y, bins=10)

# define classifier
clf = RandomForestRegressor(max_depth=2)  # max_depth=2, max_features=1

#define cross_validation scheme
cv = ShuffleSplit(n_splits=100, test_size=.25, random_state=0)

# compute cross-val score
r2_ = cross_val_score(clf, X, y, cv=cv,n_jobs=5)
print(r2_.mean())

mae_ = cross_val_score(clf, X, y, cv=cv, n_jobs=5,
                       scoring=scoring)
mmae = mae_.mean()
print('rf:', mmae)

"""
# compare to permutation distribution
maes = []
y_ = y.copy()
for _ in range(n_permutations):
    np.random.shuffle(y_)
    maes_ = cross_val_score(clf, X, y_, cv=cv, n_jobs=5,
                       scoring=scoring)
    maes.append(np.mean(maes_))

plt.figure()
plt.hist(maes, bins=10)
plt.plot(mmae, 0, '*r')
print(np.sum(maes > mmae))
"""

# attempt with Ridge regression
from sklearn.linear_model import RidgeCV
clf = RidgeCV()
mae_ = cross_val_score(clf, X, y, cv=cv, n_jobs=5,
                       scoring=scoring)
mmae = mae_.mean()
print('ridge:', mmae)
"""
for _ in range(n_permutations):
    np.random.shuffle(y_)
    maes_ = cross_val_score(clf, X, y_, cv=cv, n_jobs=5,
                       scoring=scoring)
    maes.append(np.mean(maes_))
print(np.sum(maes > mmae))

plt.figure()
plt.hist(maes, bins=10)
plt.plot(mmae, 0, '*r')
"""


# attempt with GBT
from sklearn.ensemble import GradientBoostingRegressor
clf = GradientBoostingRegressor()
mae_ = cross_val_score(clf, X, y, cv=cv, n_jobs=5,
                       scoring=scoring)
mmae = mae_.mean()
print('GBT: ', mmae)


# X = X[y < 1]
# y = y[y < 1]

threshold = 1.5
yb = y > threshold
scoring = 'roc_auc'
class_names = ['y<%f' % threshold, 'y>%f' % threshold,]


clf = RandomForestClassifier(max_depth=2)  # max_depth=2, max_features=1

#define cross_validation scheme
cv = StratifiedShuffleSplit(n_splits=100, test_size=.25, random_state=0)

# compute cross-val score
acc = cross_val_score(clf, X, yb, cv=cv,n_jobs=5, scoring=scoring)
print('Ternary accuracy, RF: ', acc.mean())

clf.fit(X, yb)
print(clf.feature_importances_)
print(np.array(labels)[np.argsort(clf.feature_importances_)[-5:]])


# Make an ROC curve

from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, yb, test_size=.5,
                                                    random_state=0)

y_score = clf.fit(X_train, y_train).predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score.T[0], pos_label=0)
lw = 2
plt.figure()
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % np.mean(acc))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc='lower right')
plt.savefig('/tmp/roc.png')

if n_permutations > 0:
    y_ = yb.copy()
    accs = []
    for _ in range(n_permutations):
        np.random.shuffle(y_)
        acc_ = cross_val_score(clf, X, y_, cv=cv, n_jobs=5,
                               scoring=scoring)
        accs.append(np.mean(acc_))
        
    print(np.sum(accs > acc.mean()))


# try with single tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=3)
acc = cross_val_score(clf, X, yb, cv=cv,n_jobs=5, scoring=scoring)
print('Binary accuracy, tree: ', acc.mean())


# 
# is classification significantly good ?
# Does a tree work ?
# Feature importance
# add age
clf.fit(X, yb)
from sklearn import tree

plt.figure(figsize=(8, 8))
annotations = tree.plot_tree(
    clf, feature_names=labels, class_names=class_names,
    fontsize=6, impurity=False)
plt.savefig('/tmp/tree.pdf', dpi=300)
# plt.savefig('/tmp/tree.png', dpi=300)
plt.savefig('/tmp/tree.svg')


#############################################################################
# Three-way classification
yt = (y > -1.5).astype(int) +  (y > 1.5).astype(int)
scoring = 'roc_auc_ovr'
class_names = ['y < -1.5', '-1.5 < y< 1.5', 'y > 1.5']

clf = RandomForestClassifier(max_depth=2)  # max_depth=2, max_features=1

#define cross_validation scheme
cv = StratifiedShuffleSplit(n_splits=100, test_size=.25, random_state=0)

# compute cross-val score
acc = cross_val_score(clf, X, yt, cv=cv,n_jobs=5, scoring=scoring)
print('Ternary accuracy, RF: ', acc.mean())

clf.fit(X, yt)
print(clf.feature_importances_)
print(np.array(labels)[np.argsort(clf.feature_importances_)[-5:]])


# Make an ROC curve

from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, yt, test_size=.5,
                                                    random_state=0)

y_score = clf.fit(X_train, y_train).predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score.T[0], pos_label=0)
lw = 2
plt.figure()
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % np.mean(acc))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic, ternary problem ')
plt.legend(loc='lower right')
plt.savefig('/tmp/roc_ternary.png')


clf = DecisionTreeClassifier(max_depth=3)
acc = cross_val_score(clf, X, yt, cv=cv,n_jobs=5, scoring=scoring)
print('Ternary accuracy, tree: ', acc.mean())
clf.fit(X, yt)

plt.figure(figsize=(8, 8))
annotations = tree.plot_tree(
    clf, feature_names=labels, class_names=class_names,
    fontsize=6, impurity=False)
plt.savefig('/tmp/tree_ternary.pdf', dpi=300)
plt.savefig('/tmp/tree._ternary.svg')


plt.show(block=False)
