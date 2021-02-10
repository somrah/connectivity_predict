"""
Population analysis of the impact of connectivity deficits on behavioral outcomes

Author: Bertrand Thirion, 2021

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, ShuffleSplit





df = pd.read_csv('liste_patients_gliome_final_total.csv', index_col=0)

# Drop a column with only Nans
df.drop(labels='CorticoThalamic_4', axis=1, inplace=True)

networks = df.columns[:-2]
X = df.values[:, :-2]
y = df.values[:, -1]

plt.figure()
plt.hist(y, bins=10)

# define classifier
clf = RandomForestRegressor(max_depth=2)  # max_depth=5, max_features=1

#define cross_validation scheme
cv = ShuffleSplit(n_splits=100, test_size=.25, random_state=0)

# compute cross-val score
r2_ = cross_val_score(clf, X, y, cv=cv,n_jobs=5)
print(r2_.mean())

mae_ = cross_val_score(clf, X, y, cv=cv, n_jobs=5,
                       scoring='neg_mean_squared_error')
mmae = mae_.mean()
print(mmae)

# compare to permutation distribution
maes = []
y_ = y.copy()
for _ in range(100):
    np.random.shuffle(y_)
    maes_ = cross_val_score(clf, X, y_, cv=cv, n_jobs=5,
                       scoring='neg_mean_squared_error')
    maes.append(np.mean(maes_))

plt.figure()
plt.hist(maes, bins=10)
plt.plot(mmae, 0, '*r')

# Try OOB errors on Random Forests
clf = RandomForestRegressor(max_depth=2, oob_score=True)
clf.fit(X, y)
oob_score_ref = clf.oob_score_
oob_scores = []
for _ in range(100):
    np.random.shuffle(y_)
    clf.fit(X, y_)
    oob_scores.append(clf.oob_score_)

plt.figure()
plt.hist(oob_scores, bins=10)
plt.plot(oob_score_ref, 0, '*r')

# attempt with Ridge regression
from sklearn.linear_model import RidgeCV
clf = RidgeCV()
mae_ = cross_val_score(clf, X, y, cv=cv, n_jobs=5,
                       scoring='neg_mean_squared_error')
mmae = mae_.mean()
print(mmae)

# attempt with GBT
from sklearn.ensemble import GradientBoostingRegressor
clf = GradientBoostingRegressor()
mae_ = cross_val_score(clf, X, y, cv=cv, n_jobs=5,
                       scoring='neg_mean_squared_error')
mmae = mae_.mean()
print(mmae)


plt.show(block=False)
