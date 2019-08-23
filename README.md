# Metis3-Lending-Club
Project 3 Lending Club Loans default predictor

Reducing the dataset to individuals and verified loans and finally removing the empty rows, current loans and collapsing all partially paid/late loans as default against the fully-paid ones for analysis and classification model-development. 

Checked that purpose has no influence on status outcome.

Selected 18 and + 2 customised ones to do a heatmap on correlation and remove those which pertain to data leakage and low correlation. 

Use LASSO to further rank the 7 features and heatmap on the weak ones (namely loan and purpose). Remove them and reduce to 5 features.
Pairplot shows that the data is not separable.

Develop 8 models (with train-test split of 80%-20%) and compared the F1, AUC and Log-loss metrics. F1 and AUC to be maximise but Log-loss to be minimised. Knn and Randomforest are the best. Tree diagram is done on decision trees. 

Use RandomisedCVSearch to do CV and hyperparameter tunings and apply these to the 8 models (which includes one bagging one of Logistic + SVC). While the F1 score improves, AUC deterioriated drastically. Since AUC is the adopted model, decided to do without hyperparameter tunings to illustrate.

With the KNN, apply the confusion matrix to find the threshold for a high recall. 

Writeup @ https://medium.com/@1reginacheong/using-machine-learning-for-investment-decisions-8df70396b5a9
"# Lending-Club" 
