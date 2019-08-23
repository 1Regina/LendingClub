# %%
# Import packages and libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pydotplus
from scipy.stats import randint
from time import time
# from patsy import dmatrices, dmatrix
import re
# import pickle
import statsmodels.api as sm
from sklearn.linear_model import lars_path
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn.externals.six import StringIO

from IPython.display import Image, display
import sklearn.metrics as metrics
from sklearn.metrics import f1_score
# from keras.utils import to_categorical
# from keras.models import Sequential
# from keras.layers import Dense, Activation
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import BaggingClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.utils import resample
# from sklearn.utils import shuffle

seed = 5
np.random.seed(seed)

#%%
# Prepare dataset
loan_impt = pd.read_csv("loan_impt.csv", low_memory=False)

# Validation on different linear models
kf = KFold(n_splits=5, shuffle=True, random_state = np.random.seed(seed))
X, y = loan_impt.drop(["loan_status"],
                    axis=1), loan_impt["loan_status"]

# hold out 20% of the data for final testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state= np.random.seed(seed))
#%%
# RandomForest Parameters
# build a classifier
clf = RandomForestClassifier(n_estimators=20)

# specify parameters and distributions to sample from
param_dist = {"max_depth": [3, None],
              "max_features": (3, 5),
              "min_samples_split": randint(2, 11),
              "criterion": ["gini", "entropy"],
              "bootstrap": [True, False]}

# Next go to:Randomized Search Hyperparameters Estimation
#%%
#Randomized Search Hyperparameters Estimation - Do this for every model
# Utility function to report best scores
def report(results, n_top=1):  # specify how many scores to print
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=5, iid=False)

# start = time()
random_search.fit(X, y)
# print("RandomizedSearchCV took %.2f seconds for %d candidates"
#       " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)

y_pred_proba = random_search.best_estimator_.predict_proba(X_test)[:,1]
#%%
