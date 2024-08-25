import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import numpy as np
import re, ast
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE, ADASYN, SMOTENC, RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, classification_report

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

rf = RandomForestClassifier(random_state=42)

rf.fit(X_train_res, y_train_res)

y_val_pred = rf.predict(X_val)

print("Validation Set Evaluation")
print(confusion_matrix(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

y_test_pred = rf.predict(X_test)

# Evaluate the model on the test set
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print('Precision: ', precision_score(y_test, y_test_pred))
print('Recall: ', recall_score(y_test, y_test_pred))
print('F1: ', f1_score(y_test, y_test_pred))

fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d', annot_kws={'size': 20})
ax.set_title('Random Forest Classifier - Accuracy Score: {:.4f}'.format(accuracy_score(y_test, y_test_pred)), fontsize=20)
ax.set_ylabel('Actual', fontsize=20)
ax.set_xlabel('Predicted', fontsize=20)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.collections[0].colorbar.ax.tick_params(labelsize=16)
plt.show()

# Save the figure
output_path = 'Model Evaluation - Random Forest Classifier - 3 Features.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()


