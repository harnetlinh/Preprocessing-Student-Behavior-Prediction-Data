import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import numpy as np
import re, ast
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE, ADASYN, SMOTENC, RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, classification_report, log_loss
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

# Data splitting and SMOTE initialization
X_train_val, X_test, y_train_val, y_test = train_test_split(
    df.drop(['dropout_status'], axis=1),
    df['dropout_status'],
    test_size=0.2,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.2,
    random_state=42
)

smote = SMOTE(random_state=42)

# Initialize RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Parameter grid to include class_weight
param_grid_rf = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [None, 10, 20, 30],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4],
    'rf__class_weight': [None, 'balanced', 'balanced_subsample']  # Adding class_weight options
}

# Create the pipeline
pipeline_rf = Pipeline(steps=[('smote', smote), ('rf', rf)])

# Set up StratifiedKFold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Set up GridSearchCV
grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=kfold, scoring='f1', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

# Best hyperparameters for Random Forest
best_params_rf = grid_search_rf.best_params_
print(f"Best Hyperparameters for Random Forest: {best_params_rf}")

# Best model for Random Forest
best_model_rf = grid_search_rf.best_estimator_

# Predictions and evaluation on validation set
y_val_pred_rf = best_model_rf.predict(X_val)
print("Validation Set Evaluation")
print(confusion_matrix(y_val, y_val_pred_rf))
print(classification_report(y_val, y_val_pred_rf))

# Log loss calculation on validation set
y_val_pred_rf_prob = best_model_rf.predict_proba(X_val)  # Get probability predictions for log loss
val_log_loss_rf = log_loss(y_val, y_val_pred_rf_prob)
print(f"Validation Log Loss: {val_log_loss_rf}")

# Evaluate on the test set
y_test_pred_rf = best_model_rf.predict(X_test)
y_test_pred_rf_prob = best_model_rf.predict_proba(X_test)  # Get probability predictions for log loss
print("Accuracy:", accuracy_score(y_test, y_test_pred_rf))
print('Precision: ', precision_score(y_test, y_test_pred_rf))
print('Recall: ', recall_score(y_test, y_test_pred_rf))
print('F1: ', f1_score(y_test, y_test_pred_rf))

# Log loss calculation on test set
test_log_loss_rf = log_loss(y_test, y_test_pred_rf_prob)
print(f"Test Log Loss: {test_log_loss_rf}")

# Visualization of confusion matrix and log loss
fig, ax = plt.subplots(1, 2, figsize=(20, 8))

# Plot confusion matrix for test set
sns.heatmap(confusion_matrix(y_test, y_test_pred_rf), annot=True, fmt='d', annot_kws={'size': 20}, ax=ax[0])
ax[0].set_title('Confusion Matrix\nRandom Forest Classifier - Accuracy Score: {:.4f}'.format(accuracy_score(y_test, y_test_pred_rf)), fontsize=15)
ax[0].set_ylabel('Actual', fontsize=15)
ax[0].set_xlabel('Predicted', fontsize=15)

# Plot Log Loss for validation and test sets
ax[1].bar(['Validation Log Loss', 'Test Log Loss'], [val_log_loss_rf, test_log_loss_rf], color=['blue', 'orange'])
ax[1].set_title('Log Loss Evaluation', fontsize=15)
ax[1].set_ylabel('Log Loss', fontsize=15)

plt.tight_layout()
plt.show()

# Save the figure
output_path = 'Model Evaluation - Random Forest with Hyperparameter Tuning - Log Loss.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()



