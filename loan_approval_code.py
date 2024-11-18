# source: https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset

# importing required modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold

# importing dataset
data = pd.read_csv('loan_approval_dataset.csv')
print(data.head())

# trimming extra spaces from colnames and dropping extra columns
colnames = list(data.columns)
for i in colnames: data.rename(columns = {i: i.strip()}, inplace = True)
data = data.drop('loan_id', axis = 'columns')
print(data.head())

# transforming 'string' values to 'numeric'
encoder = LabelEncoder()
self_employed = encoder.fit_transform(data['self_employed'])
education = encoder.fit_transform(data['education'])
loan_status = encoder.fit_transform(data['loan_status'])
data['self_employed'] = self_employed
data['education'] = education
data['loan_status'] = loan_status
print(data.head())

# extracting feature & outcome and getting feature importance
feature = data.drop('loan_status', axis = 'columns')
outcome = data['loan_status'].values
estimator = DecisionTreeClassifier(random_state = 0)
selector = RFECV(estimator = estimator, cv = 5)
selector.fit(feature, outcome)

# getting selected features
selected_features = list(feature.columns[selector.support_])
feature = feature[selected_features]
print(feature.head())

# model efficieny test using cross validation score
model = RandomForestClassifier(random_state = 0)
feature = StandardScaler().fit_transform(feature)
kfold = KFold(n_splits = 5)
cv_score = cross_val_score(model, feature, outcome, cv = kfold).mean()
print(cv_score)

# model training & prediction results
xtrain, xtest, ytrain, ytest = train_test_split(feature, outcome, test_size = 0.2)
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
score = r2_score(ytest, ypred)
print(score)

# visualisation of results by confusion matrix
cnf_matrix = confusion_matrix(ytest, ypred)
plt.title('Prediction Results')
sns.heatmap(cnf_matrix, cmap = 'GnBu', fmt = '.0f', annot = True)
plt.show()