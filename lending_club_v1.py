# %%
# Import packages and libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pydotplus
from IPython.display import Image
from IPython.display import display
from ipywidgets import interactive, FloatSlider
from subprocess import call
import os
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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import linear_model
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn.externals.six import StringIO

import sklearn.metrics as metrics 
import sklearn.metrics as f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve , auc
from sklearn.metrics import log_loss
# from keras.utils import to_categorical
# from keras.models import Sequential
# from keras.layers import Dense, Activation
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import resample
from sklearn.utils import shuffle

seed = 5
np.random.seed(seed)

# Display all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# %%
# read the downloaded file
loan_data_full = pd.read_csv('loan.csv', low_memory=False)

# %%
# Filter data for interested columns
loan_data_raw1 = loan_data_full[['loan_amnt', 'term', 'int_rate', 'installment', 'grade',
                                 'home_ownership', 'annual_inc', 'dti', 'out_prncp', 'total_pymnt',
                                 'total_rec_prncp', 'total_rec_int', 'mort_acc', 'pct_tl_nvr_dlq',
                                 'delinq_2yrs', 'disbursement_method', 'purpose',
                                 'application_type', 'verification_status', 'loan_status']]

loan_data_raw1.head()
loan_data_raw1.info()
loan_data_raw1.describe()
# %%
# Export dataset to csv
loan_data_raw1.to_csv("loan_data_raw.csv", index=False)
loan_data_raw = pd.read_csv("loan_data_raw.csv", low_memory=False)
loan_data_raw.head()

# %%
# Rename the columns
loan_data_raw.set_axis(['loan', 'term', 'interest', 'monthly_due',
                        'grade', 'home_ownership', 'annual_inc', 'othdebt_inc',
                        'outst_loan', 'total_paid', 'total_princi_paid', 'total_interest_paid',
                        'mortgage_acc', 'percent_trades_nvr_delin', 'delin_2yrs',
                        'disbursement_mtd', 'purpose', 'loan_type', 'verification', 'loan_status'], axis=1, inplace=True)
# %%
# Remove joint applications and unverified income
loan_all_types = loan_data_raw[(loan_data_raw.loan_type == "Individual") & 
                                                            (loan_data_raw.verification != "Not Verified")]
loan_all_types.info()
# %%
# Drop unwanted columns since filter is complete
loan_status = loan_all_types.drop(['loan_type', 'verification'], 1)
loan_status.head()
loan_status.to_csv("loan_status.csv", index=False)

# %%
# Count different types of loans and charts them
loan_status['loan_status'].value_counts()
f, ax = plt.subplots(figsize=(7, 3))
sns.countplot(y="loan_status", data=loan_status, color="orange", 
              order=loan_status['loan_status'].value_counts().index)
# plt.savefig("loan_status.svg")
# %%
# Count different types of purpose and charts them
loan_status['purpose'].value_counts()
f, ax = plt.subplots(figsize=(7, 3))
sns.countplot(y="purpose", data=loan_status, color="orange",
              order=loan_status['purpose'].value_counts().index)
plt.show
#%%
loan_status = pd.read_csv("loan_status.csv", low_memory=False)
#Wordcloud of purpose with current loans
wordcloud = WordCloud(background_color='white',colormap = "inferno", max_font_size = 50).generate_from_frequencies(loan_status['purpose'].value_counts())
#wordcloud.recolor(color_func = grey_color_func)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# %%
# Remove current loans
loan_not_curr = loan_status[(loan_status.loan_status != "Current")]
loan_not_curr.info()
#loan_not_curr.to_csv("loan_not_curr.csv", index=False)

# %%
# Drop rows with blank cells
loan_not_curr.dropna(how='any', inplace=True)
all_features = loan_not_curr.reset_index(drop=True)
loan_not_curr.to_csv("loan_not_curr.csv", index=False)
loan_not_curr.info()
# %%
# Count different types of purpose among good and bad loans and charts them
loan_not_curr['purpose'].value_counts()
f, ax = plt.subplots(figsize=(7, 3))
sns.countplot(y="purpose", data=loan_not_curr, color="orange",
              order=loan_not_curr['purpose'].value_counts().index)
plt.show
#plt.savefig("purpose.svg")

#%%
# Display wordcloud on purpose categories
#loan_not_curr = pd.read_csv("loan_not_curr.csv", low_memory=False)

wordcloud = WordCloud(background_color='white',colormap = "plasma", max_font_size = 50).generate_from_frequencies(loan_not_curr['purpose'].value_counts())
#wordcloud.recolor(color_func = grey_color_func)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
#%%
#WordCloud for Fully Paid
loan_paid = loan_status[(loan_status.loan_status == "Fully Paid")]
wordcloud = WordCloud(background_color='white',colormap = "inferno", max_font_size = 50).generate_from_frequencies(loan_paid['purpose'].value_counts())
#wordcloud.recolor(color_func = grey_color_func)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
#%%
#WordCloud for Default
loan_default = loan_status[(loan_status.loan_status != "Fully Paid") & (loan_status.loan_status != "Current")]
wordcloud = WordCloud(background_color='white',colormap = "inferno", max_font_size = 50).generate_from_frequencies(loan_default['purpose'].value_counts())
#wordcloud.recolor(color_func = grey_color_func)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
# %%
#loan_not_curr = pd.read_csv("loan_not_curr.csv", low_memory=False)
# Specify labels for clean up categorical data
categories_labels = {"term": {" 36 months": 0, " 60 months": 1},
                     "grade": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6},
                     "home_ownership": {"OWN": 0, "RENT": 1, "MORTGAGE": 2, "ANY": 3},
                     "disbursement_mtd": {"Cash": 0, "DirectPay": 1},
                     "purpose": {
                     "debt_consolidation": 0, "credit_card": 1,
                     "home_improvement": 2, "house": 2, "major_purchase": 2, "moving": 2,
                     "car": 3, "medical": 3, "other": 3, "renewable_energy": 3,
                     "small_business": 3, "vacation": 3, "wedding": 3, },
                     "loan_status": {"Fully Paid": 0, "Charged Off": 1, "Default": 1,
                                     "In Grace Period": 1, "Late (16-30 days)": 1, "Late (31-120 days)": 1}}

loans_data = loan_not_curr.replace(categories_labels)
loans_data.info()
loans_data.head()

# %%
# Engineer a feature for monthly instalment / monthly income over 13 months
loans_engg = loans_data.copy()
loans_engg["loan_inc"] = (loans_engg['monthly_due'].apply(
    lambda x: x*100)) / ((loans_engg['annual_inc']).div(13))
loans_engg["%paid"] = (loans_engg['total_princi_paid'].apply(
    lambda x: x*100))/(loans_engg['loan'])
all_features = loans_engg[[c for c in loans_engg if c not in
                           ['othdebt_inc', 'loan_inc', 'outst_loan', 'total_paid', "%paid", 'total_princi_paid', 'total_interest_paid',
                            'mortgage_acc', 'percent_trades_nvr_delin', 'delin_2yrs',	'disbursement_mtd',
                            'purpose', 'loan_status']]
                          + ['othdebt_inc','loan_inc', 'outst_loan', 'total_paid', "%paid", 'total_princi_paid', 'total_interest_paid',
                             'mortgage_acc', 'percent_trades_nvr_delin', 'delin_2yrs',	'disbursement_mtd',
                             'purpose', 'loan_status']]
all_features.head()


# %%
# Reduced features to prevent data leakage
all_features.head()
reduced = all_features[['loan', 'term', 'interest',
                        'grade', 'othdebt_inc', 'loan_inc',
                        'purpose', 'loan_status']]

reduced.head()
reduced.to_csv("loan_reduced.csv", index=False)

# %%
# use heatmap to see correlation of reduced features
sns.heatmap(reduced.corr(), cmap="YlGnBu", annot=True, vmin=-1, vmax=1)
plt.xticks(rotation=87)
# %%
# use heatmap to see correlation of all features
sns.heatmap(all_features.corr(), cmap="YlGnBu", annot=False, vmin=-1, vmax=1)
plt.xticks(rotation=87)
#%%
# Heatmap to show low correlation of loan and purpose to loan_status
sns.heatmap(reduced[['loan','purpose','loan_status']].corr(), cmap="YlGnBu", annot=True, vmin=-1, vmax=1)
plt.yticks(rotation=0)

# %% pairplot. Takes forever so went to jupyter notebook for this
sns.pairplot(reduced, height=1.2, aspect=1.5, hue='loan_status')
# %%
# Model selection and feature selection
# 1. OLS Model
# Slice data into features and target
X = reduced.drop(columns=["loan_status"]).astype(float)
y = reduced.loc[:, "loan_status"].astype(float)
#%%
# fit model with loan_status as target
loan_status = sm.OLS(y, X, data=reduced)
loan_status_analysis = loan_status.fit()

# summarize OLS Regression model
loan_status_analysis.summary()

# %%
# Steamline to 5 impt features
reduced = pd.read_csv("loan_reduced.csv", low_memory=False)
loan_impt=reduced[['term', 'interest', 'grade',
                    'othdebt_inc',	'loan_inc',
                    'loan_status']]
loan_impt.to_csv("loan_impt.csv", index = False)

#%%
# use heatmap to see correlation of impt features
sns.heatmap(loan_impt.corr(), cmap = "YlGnBu",
            annot = True, vmin = -1, vmax = 1)
plt.xticks(rotation =45)
#%%
loan_reduced = pd.read_csv("loan_reduced.csv", low_memory=False)
X = loan_reduced.drop(columns=["loan_status"]).astype(float)
y = loan_reduced.loc[:, "loan_status"].astype(float)
# %%
# Linear Regression, Ridge
# Split the data into 3 portions: 60% for training, 20% for validation (used to select the model), 20% for final testing evaluation.
# With reduced features

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=np.random.seed(seed))
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=.25, random_state=np.random.seed(seed))

# %%
# Features selection with LASSO
# Scale the variables
std = StandardScaler()
std.fit(X_train.values)
X_tr = std.transform(X_train.values)

# Finding the lars paths
print("Computing regularization path using the LARS ...")
alphas, _, coefs = lars_path(X_tr, y_train.values, method='lasso')

# plotting the LARS path
xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

plt.figure(figsize=(8, 8))
plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.axis('tight')
plt.legend(X_train.columns, loc=1)
plt.show()


# %%
#  Features selection with Ridge - Not required anymore
# Scale the variables
std = StandardScaler()
std.fit(X_train.values)
X_tr = std.transform(X_train.values)

# Finding the lars paths
print("Computing regularization path using the LARS ...")
alphas, _, coefs = lars_path(X_tr, y_train.values, method='ridge')

# plotting the LARS path
xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

plt.figure(figsize=(8, 8))
plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('Ridge Path')
plt.axis('tight')
plt.legend(X_train.columns)
plt.show()


# %% 
# pairplot - Done via Jupyter notebook
# sns.pairplot(loan_impt, height = 1.2, aspect = 1.5, hue = 'loan_status')

#%%
# 1. OLS Model
# Slice data into features and target
X = loan_impt.drop(columns=["loan_status"]).astype(float)
y = loan_impt.loc[:, "loan_status"].astype(float)

# fit model with loan_status as target
loan_status = sm.OLS(y, X, data=loan_impt)
loan_status_analysis = loan_status.fit()

# summarize OLS Regression model
loan_status_analysis.summary()

# %%
# Validation on different linear models
kf = KFold(n_splits=5, shuffle=True, random_state = np.random.seed(seed))
X, y = loan_impt.drop(["loan_status"],
                    axis=1), loan_impt["loan_status"]

# hold out 20% of the data for final testing
X, X_test, y, y_test = train_test_split(X, y, test_size=.2, random_state= np.random.seed(seed))
# further partition X, y into datasets X_train, y_train (60% of original) and X_val, y_val (20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.25, random_state=np.random.seed(seed))

# #set up the 5 models we're choosing from:

lm = LinearRegression()

# #Feature scaling for train, val, and test so that we can run our ridge, lasso, elasticnet model on each
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train.values)
X_val_scaled = scaler.transform(X_val.values)
X_test_scaled = scaler.transform(X_test.values)

#Ridge
lm_ridge = Ridge(alpha=1)
#Lasso
lm_lasso = Lasso(alpha=1)
#ElasticNet
lm_elastic = ElasticNet(alpha = 1, l1_ratio= 0.5)

# #Feature transforms for train, val, and test so that we can run our poly model on each
poly = PolynomialFeatures(degree=2) 

X_train_poly = poly.fit_transform(X_train.values)
X_val_poly = poly.transform(X_val.values)
X_test_poly = poly.transform(X_test.values)

lm_poly = LinearRegression()

# Show R^2 scores for these models
cvs_lm = cross_val_score(lm, X_test, y_test, cv=kf, scoring='r2')
#print(("Linear Regression test R^:"),( cvs_lm))
print(("Linear Regression test mean R^:"), round(np.mean(cvs_lm),3), "+-", round(np.std(cvs_lm),3) )
cvs_ridge = cross_val_score(lm_ridge, X_test_scaled, y_test, cv=kf, scoring='r2')
#print(('Ridge Regression test R^2:'), cvs_ridge)
print( ("Ridge Regression test mean R^:"), round(np.mean(cvs_ridge),3), "+-", round(np.std(cvs_ridge),3) )
cvs_lasso = cross_val_score(lm_lasso, X_test_scaled, y_test, cv=kf, scoring='r2')
#print(('Lasso Regression test R^2:'),cvs_lasso)
print( ("Lasso Regression test mean R^:"), round(np.mean(cvs_lasso),3), "+-", round(np.std(cvs_lasso),3) )
cvs_elastic = cross_val_score(lm_elastic, X_test_scaled, y_test, cv=kf, scoring='r2')
#print(('ElasticNet Regression test R^2:'),cvs_elastic)
print( ("ElasticNet Regression test mean R^:"), round(np.mean(cvs_elastic),3), "+-", round(np.std(cvs_elastic),3) )
cvs_poly = cross_val_score(lm_poly, X_test_poly, y_test, cv=kf, scoring='r2')
#print(('Degree 2 polynomial regression test R^2:'),cvs_poly)
print( ('Degree 2 polynomial regression test mean R^2:'), round(np.mean(cvs_poly),3), "+-", round(np.std(cvs_poly),3) )


#%%
# Count the 2 types of loans in the final laon set and charts them
loan_impt['loan_status'].value_counts()
f, ax = plt.subplots(figsize=(7, 3))
sns.countplot(y="loan_status", data=loan_impt, color="c",
              order=loan_impt['loan_status'].value_counts().index)

# %%
# Define X for features and y fpr target
loan_impt = pd.read_csv("loan_impt.csv", low_memory=False)
X, y = loan_impt.drop(["loan_status"],
                    axis=1), loan_impt["loan_status"]
kf = KFold(n_splits=5, shuffle=True, random_state = np.random.seed(seed))
# hold out 20% of the data for final testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state= np.random.seed(seed))
#%%
# Define template to test models and their average scores
def quick_test(model, X, y):
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=.2, random_state= np.random.seed(seed))
    model.fit(xtrain, ytrain)
    return model.score(xtest, ytest)

def quick_test_afew_times(model, X, y, n=5):
    return np.mean([quick_test(model, X, y) for j in range(n)])

#%%
#KNN
knn = KNeighborsClassifier(n_neighbors=7,weights='distance',algorithm='auto')
knn_best = knn.fit(X_train, y_train)
y_pred = knn_best.predict(X_test)
print(quick_test_afew_times(knn_best, X_test, y_test))
print("Log-loss on knn: {:6.4f}".format(log_loss(y_test, knn.predict_proba(X_test))))
# Classification Report 
print(classification_report(y_test, y_pred))

#ROC Curve for KNN
knn_roc_auc = roc_auc_score(y_test, knn.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, knn.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='K Nearest Neighbours (area = %0.2f)' % knn_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC AUC KNearestNeigh.png')
plt.show()

#%%
# Using Decision Tree
decisiontree = DecisionTreeClassifier(criterion='gini', max_depth=3, max_features= 3, min_samples_leaf=2)
decisiontree_best = decisiontree.fit(X_train, y_train)
y_pred = decisiontree_best.predict(X_test)
# Do the test 10 times with a Decision Tree and get the average score
print(quick_test_afew_times(decisiontree_best, X_test, y_test))
print("Log-loss on Decision Tree : {:6.4f}".format(log_loss(y_test, decisiontree.predict_proba(X_test))))
# Classification Report 
print(classification_report(y_test, y_pred))

#ROC Curve for Decision Tree
decisiontree_roc_auc = roc_auc_score(y_test, decisiontree.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, decisiontree.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Decision Tree (area = %0.2f)' % decisiontree_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC AUC DecisionTree.png')
plt.show()


#%%
# # Tree classifier graph - give up 
# dtree=DecisionTreeClassifier()
# dtree.fit(X_train,y_train)
# dot_data = StringIO()
# export_graphviz(dtree, out_file=None,  
#                 filled=True, rounded=True,max_depth = 5,
#                 special_characters=True, feature_names=X_train.columns,precision = 2)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

# #Image(filename = 'tree.png')
# #Image(graph.create_png())
# os.system('dot','-Tpng', 'tree.dot', '-o', 'tree.png')

#%%
# Using Random Forest 
randomforest = RandomForestClassifier(n_estimators=30, bootstrap =True, criterion='gini', max_depth= 3, max_features=5, min_samples_split= 9)
randomforest.fit(X_train, y_train)
y_pred = randomforest.predict(X_test)
# Do the test 10 times with Random Forest and get the average score
print(quick_test_afew_times(randomforest, X_test, y_test))
print("Log-loss on RandomForest: {:6.4f}".format(log_loss(y_test, randomforest.predict_proba(X_test))))
# Classification Report 
print(classification_report(y_test, y_pred))

#ROC Curve for Random Forest
randomforest_roc_auc = roc_auc_score(y_test, randomforest.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, randomforest.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Random Forest (area = %0.2f)' % randomforest_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC AUC RandomForest.png')
plt.show()
#%%
#Gaussian Naive Bayes 
gnb = GaussianNB(priors= None)
gnb_best = gnb.fit(X_train, y_train)
y_pred = gnb_best.predict(X_test)
print(quick_test_afew_times(gnb_best, X_test, y_test))
print("Log-loss on Gaussian: {:6.4f}".format(log_loss(y_test, gnb.predict_proba(X_test))))
# Classification Report 
print(classification_report(y_test, y_pred))

#ROC Curve for Gaussian Naive Bayes
gnb_roc_auc = roc_auc_score(y_test, gnb.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, gnb.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Gaussian Naive Bayes (area = %0.2f)' % gnb_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC AUC GaussianNB.png')
plt.show()

#%%
#BernoulliNB
bnb = BernoulliNB(fit_prior=True, alpha=0)
bnb_best = bnb.fit(X_train, y_train)
y_pred = bnb_best.predict(X_test)
print(quick_test_afew_times(bnb_best, X_test, y_test))
print("Log-loss on BernoulliNB: {:6.4f}".format(log_loss(y_test, bnb.predict_proba(X_test))))
# Classification Report 
print(classification_report(y_test, y_pred))

#ROC Curve for BernoulliNB
bnb_roc_auc = roc_auc_score(y_test, bnb.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, bnb.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='BernoulliNB (area = %0.2f)' % bnb_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC AUC BernoulliNB.png')
plt.show()


#%%
#Using Logistic Regression
logreg = LogisticRegression(tol=0.8, solver='saga', penalty='none', multi_class='auto', l1_ratio=0.6, C=0.2)
logreg_best = logreg.fit(X_train, y_train)
y_pred = logreg_best.predict(X_test)
print (quick_test_afew_times(logreg, X_test, y_test))
print("Log-loss on logistic regression: {:6.4f}".format(log_loss(y_test, logreg.predict_proba(X_test))))
# Classification Report 
print(classification_report(y_test, y_pred))

#ROC Curve for Logistic Regression
logreg_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logreg_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC AUC LogisticReg.png')
plt.show()

#%%
# Using Linear SVC
linearsvc = LinearSVC()
linearsvc_best = linearsvc.fit(X_train, y_train)
y_pred = linearsvc_best.predict(X_test)
# Do the test 10 times with a LinearSVC and get the average score
print(quick_test_afew_times(linearsvc_best, X_test, y_test))
#print("Log-loss on Linear SVC : {:6.4f}".format(linearsvc(y_test, linearsvc_best.predict_proba(X_test))))
# Classification Report 
print(classification_report(y_test, y_pred))

#ROC Curve for Linear SVC
linearsvc_roc_auc = roc_auc_score(y_test, linearsvc_best.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, linearsvc_best.predict(X_test)[:,1]) # no predict_proba
plt.figure()
plt.plot(fpr, tpr, label='Linear SVC (area = %0.2f)' % linearsvc_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC AUC Linear.png')
plt.show()

#%%
#Bagging
bagging_logit = BaggingClassifier(linear_model.LogisticRegression(solver='lbfgs', max_iter=10000, C=100000), 
                                  n_estimators=50, max_samples=0.50, max_features=0.80, verbose=10)
bagging_logit.fit(X_train, y_train)
y_pred = bagging_logit.predict(X_test)
print(quick_test_afew_times(bagging_logit, X_test, y_test))
print("Log-loss on bagging_logit: {:6.4f}".format(log_loss(y_test, bagging_logit.predict_proba(X_test))))
# Classification Report 
print(classification_report(y_test, y_pred))

#ROC Curve for Bagging
bagging_logit_roc_auc = roc_auc_score(y_test, bagging_logit.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, bagging_logit.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Bagging Classifier (Linear and Logistic Regression) (area = %0.2f)' % bagging_logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC AUC bagging_logit.png')
plt.show()

#%%
# plot the ROC curves
plt.figure(figsize=(10,10))
y_pred = knn.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='yellow',
       lw=3, label='KNN (area = %0.2f)' % knn_roc_auc)
y_pred = decisiontree.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='blue',
       lw=2, label='DecisionTree (area = %0.2f)' % decisiontree_roc_auc)
y_pred = randomforest.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='green',
       lw=2, label='RandomForest (area = %0.2f)' % randomforest_roc_auc)
y_pred = gnb.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='white',
       lw=3, label='GaussianNB (area = %0.2f)' % gnb_roc_auc)
y_pred = bnb.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='orange',
       lw=3, label='BernoulliNB (area = %0.2f)' % bnb_roc_auc)
y_pred = logreg.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='red',
       lw=2, label='LogisticRegression (area = %0.2f)' % logreg_roc_auc)
y_pred = linearsvc.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='cyan',
       lw=2, label='LinearSVC (area = %0.2f)' % linearsvc_roc_auc)
y_pred = bagging_logit.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='magenta',
       lw=2, label='Bagging LinearSVC_LogReg (area = %0.2f)' % bagging_logit_roc_auc)
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic', fontsize=17)
plt.legend(loc='lower right', fontsize=13)
plt.show()
plt.savefig('ROC AUC all models.png')

#%%
