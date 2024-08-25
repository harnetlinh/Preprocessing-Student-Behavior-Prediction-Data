import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import numpy as np
import re, ast
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE, ADASYN, SMOTENC, RandomOverSampler
from sklearn.linear_model import LogisticRegression
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

df1, df2, df3 = [], [], []

# Transform the data into normal format
for index, row in combine_data.iterrows():
    sem = ast.literal_eval(row['semester_1'])
    sem_df = pd.DataFrame.from_dict(sem, orient='index').reset_index()\
        .rename(columns={'index': 'subject_code'})
    sem_df['student_code'] = row['student_code']
    sem_df['dropout_status'] = row['dropout_status']
    sem_df['semester'] = 1
    df1.append(sem_df)

for index, row in combine_data.iterrows():
    sem = ast.literal_eval(row['semester_2'])
    sem_df = pd.DataFrame.from_dict(sem, orient='index').reset_index()\
        .rename(columns={'index': 'subject_code'})
    sem_df['student_code'] = row['student_code']
    sem_df['dropout_status'] = row['dropout_status']
    sem_df['semester'] = 2
    df2.append(sem_df)

for index, row in combine_data.iterrows():
    sem = ast.literal_eval(row['semester_3'])
    sem_df = pd.DataFrame.from_dict(sem, orient='index').reset_index()\
        .rename(columns={'index': 'subject_code'})
    sem_df['student_code'] = row['student_code']
    sem_df['dropout_status'] = row['dropout_status']
    sem_df['semester'] = 3
    df3.append(sem_df)

combine_df1 = pd.concat(df1).dropna()
combine_df2 = pd.concat(df2).dropna()
combine_df3 = pd.concat(df3).dropna()

array_loop = [combine_df1, combine_df2, combine_df3]

merged_df = combine_data

for index, item in enumerate(array_loop):
    item["total_credit"] = item["number_of_credit"] * item["learnt_times"]
    item["credit_passed"] = (1 / item["learnt_times"] * item["total_credit"]).where(item["passed"] == True, 0)
    df = item.groupby('student_code').agg({
        'total_credit': 'sum',
        'credit_passed': 'sum',
    }).reset_index()
    df[f"semester_{index+1}_passed_percent"] = df["credit_passed"] / df["total_credit"] * 100
    dfm = df[["student_code", f"semester_{index+1}_passed_percent"]]
    merged_df = pd.merge(merged_df, dfm, on='student_code')

df = merged_df[['dropout_status', 
                'semester_1_attendance_rate', "semester_1_average_score", "semester_1_passed_percent",
                'semester_2_attendance_rate', "semester_2_average_score", "semester_2_passed_percent",
                'semester_3_attendance_rate', "semester_3_average_score", "semester_3_passed_percent"]]

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

lr = LogisticRegression(random_state=42)

lr.fit(X_train_res, y_train_res)

y_val_pred = lr.predict(X_val)

print("Validation Set Evaluation")
print(confusion_matrix(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

y_test_pred = lr.predict(X_test)

# Evaluate the model on the test set
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print('Precision: ', precision_score(y_test, y_test_pred))
print('Recall: ', recall_score(y_test, y_test_pred))
print('F1: ', f1_score(y_test, y_test_pred))

fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d', annot_kws={'size': 20})
ax.set_title('Logistic Regression - Accuracy Score: {:.4f}'.format(accuracy_score(y_test, y_test_pred)), fontsize=20)
ax.set_ylabel('Actual', fontsize=20)
ax.set_xlabel('Predicted', fontsize=20)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.collections[0].colorbar.ax.tick_params(labelsize=16)
plt.show()

# Save the figure
output_path = 'Model Evaluation - Logistic Regression - 9 Features.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()


