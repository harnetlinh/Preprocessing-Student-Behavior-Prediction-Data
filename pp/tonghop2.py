import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import numpy as np
import re, ast
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE, ADASYN, SMOTENC, RandomOverSampler
from imblearn.under_sampling import TomekLinks, ClusterCentroids, NearMiss, RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, log_loss
from sklearn.svm import SVC
from collections import Counter
from sklearn.naive_bayes import GaussianNB

from imblearn.pipeline import Pipeline

# Load the dataset
combine_data = pd.read_csv('sampled_students_data_new.csv')

# Create a new column to differentiate dropout and non-dropout students
combine_data['dropout_status'] = combine_data['semester_3_status'].apply(lambda x: 0 if x == 'THO' else 1)

# Pattern to split the JSON formatted data
decimal_pattern = re.compile(r"Decimal\('(\d+\.\d+)'\)")

# Apply pattern on selected column
combine_data['semester_1'] = combine_data['semester_1'].apply(
    lambda x: decimal_pattern.sub(r"'\1'", str(x)))
combine_data['semester_2'] = combine_data['semester_2'].apply(
    lambda x: decimal_pattern.sub(r"'\1'", str(x)))
combine_data['semester_3'] = combine_data['semester_3'].apply(
    lambda x: decimal_pattern.sub(r"'\1'", str(x)))

df_arr = []

# Transform the data into normal format
for index, row in combine_data.iterrows():
    sem = ast.literal_eval(row['semester_1'])
    sem_df = pd.DataFrame.from_dict(sem, orient='index').reset_index()\
        .rename(columns={'index': 'subject_code'})
    sem_df['student_code'] = row['student_code']
    sem_df['dropout_status'] = row['dropout_status']
    sem_df['semester'] = 1
    df_arr.append(sem_df)

for index, row in combine_data.iterrows():
    sem = ast.literal_eval(row['semester_2'])
    sem_df = pd.DataFrame.from_dict(sem, orient='index').reset_index()\
        .rename(columns={'index': 'subject_code'})
    sem_df['student_code'] = row['student_code']
    sem_df['dropout_status'] = row['dropout_status']
    sem_df['semester'] = 2
    df_arr.append(sem_df)

for index, row in combine_data.iterrows():
    sem = ast.literal_eval(row['semester_3'])
    sem_df = pd.DataFrame.from_dict(sem, orient='index').reset_index()\
        .rename(columns={'index': 'subject_code'})
    sem_df['student_code'] = row['student_code']
    sem_df['dropout_status'] = row['dropout_status']
    sem_df['semester'] = 3
    df_arr.append(sem_df)

combine_df = pd.concat(df_arr)

combine_df = combine_df.dropna()

combine_df["attendance_rate"] = combine_df["attendance_rate"].astype(float)

combine_df["average_score"] = combine_df["average_score"].astype(float)

combine_df['total_score'] = combine_df['average_score'] * combine_df['number_of_credit']

combine_df["total_credit"] = combine_df["number_of_credit"] * combine_df["learnt_times"]

combine_df["credit_passed"] = (1 / combine_df["learnt_times"] * combine_df["total_credit"]).where(combine_df["passed"] == True, 0)

cg_df = combine_df.groupby('student_code').agg({
    'total_credit': 'sum',
    'credit_passed': 'sum',
    'number_of_credit': 'sum',
    'attendance_rate': 'mean',
    'total_score': 'sum'
}).reset_index()

cg_df["passed_percent"] = cg_df["credit_passed"] / cg_df["total_credit"] * 100

cg_df['average_score'] = cg_df['total_score'] / cg_df['number_of_credit']

merged_df = pd.merge(combine_data, cg_df, on='student_code')

df = merged_df[['dropout_status', 
                'attendance_rate', 
                'average_score',
                'passed_percent']]

# Split the data into training/validation set and test set with ratio of 2/8
X_train_val,X_test,y_train_val,y_test = train_test_split(df.drop(['dropout_status'], axis=1),
                                                        df['dropout_status'], 
                                                        test_size=0.2, random_state=42)

# Split the training/validation set into training set and validation set with ratio of 2/8
X_train,X_val,y_train,y_val = train_test_split(X_train_val, y_train_val,
                                               test_size=0.2, random_state=42)

# Initializing SMOTE
smote = SMOTE(random_state=42)

# Apply SMOTE for oversampling
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# df_resampled = pd.concat([X_train_res, y_train_res], axis=1)

# # Correlation Matrix to evaluate after resampling with SMOTE
# fig, ax = plt.subplots(figsize=(14, 10))
# sns.heatmap(df_resampled.corr(), annot=True, annot_kws={'size': 15}, cmap='crest', linewidths=.1)
# ax.set_title('Correlation Matrix Heatmap (Oversampling - SMOTE)')
# plt.show()

"""
Logistic Regression
"""
# lr = LogisticRegression(random_state=42)

# lr.fit(X_train_res, y_train_res)

# y_val_pred = lr.predict(X_val)

# print("Validation Set Evaluation")
# print(confusion_matrix(y_val, y_val_pred))
# print(classification_report(y_val, y_val_pred))

# y_test_pred = lr.predict(X_test)

# # Evaluate the model on the test set
# print("Accuracy:", accuracy_score(y_test, y_test_pred))
# print('Precision: ', precision_score(y_test, y_test_pred))
# print('Recall: ', recall_score(y_test, y_test_pred))
# print('F1: ', f1_score(y_test, y_test_pred))

# sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d')
# plt.title('Accuracy Score: {}'.format(accuracy_score(y_test, y_test_pred)))
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.show()

"""
Gaussian Naive Bayes
"""
# nb = GaussianNB()

# nb.fit(X_train_res, y_train_res)

# y_val_pred = nb.predict(X_val)

# print("Validation Set Evaluation")
# print(confusion_matrix(y_val, y_val_pred))
# print(classification_report(y_val, y_val_pred))

# y_test_pred = nb.predict(X_test)

# # Evaluate the model on the test set
# print("Accuracy:", accuracy_score(y_test, y_test_pred))
# print('Precision: ', precision_score(y_test, y_test_pred))
# print('Recall: ', recall_score(y_test, y_test_pred))
# print('F1: ', f1_score(y_test, y_test_pred))

# sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d')
# plt.title('Accuracy Score: {}'.format(accuracy_score(y_test, y_test_pred)))
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.show()

"""
Random Forest Classifier
"""
# rf = RandomForestClassifier(random_state=42)

# rf.fit(X_train_res, y_train_res)

# y_val_pred = rf.predict(X_val)

# print("Validation Set Evaluation")
# print(confusion_matrix(y_val, y_val_pred))
# print(classification_report(y_val, y_val_pred))

# y_test_pred = rf.predict(X_test)

# # Evaluate the model on the test set
# print("Accuracy:", accuracy_score(y_test, y_test_pred))
# print('Precision: ', precision_score(y_test, y_test_pred))
# print('Recall: ', recall_score(y_test, y_test_pred))
# print('F1: ', f1_score(y_test, y_test_pred))

# sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d')
# plt.title('Accuracy Score: {}'.format(accuracy_score(y_test, y_test_pred)))
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.show()

"""
Balance Random Forest Classifier 
BRFC đã tự cân bằng tập dữ liệu trong quá trình xây dựng model, không sử dụng
các phương pháp resampling tại đây
(BRFC sử dụng undersampling)
"""
brf = BalancedRandomForestClassifier(random_state=42)

brf.fit(X_train, y_train)

# Evaluate on the validation set
y_val_pred = brf.predict(X_val)

# Classification report and confusion matrix for validation set
print("Validation Set Evaluation")
print(confusion_matrix(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

# Evaluate on the test set
y_test_pred = brf.predict(X_test)

# Evaluate the model on the test set
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print('Precision: ', precision_score(y_test, y_test_pred))
print('Recall: ', recall_score(y_test, y_test_pred))
print('F1: ', f1_score(y_test, y_test_pred))

sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d')
plt.title('Accuracy Score: {}'.format(accuracy_score(y_test, y_test_pred)))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
