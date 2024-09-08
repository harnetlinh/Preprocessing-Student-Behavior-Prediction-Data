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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the dataset
combine_data = pd.read_csv('sampled_students_data_new.csv')

# Create a new column to differentiate dropout and non-dropout students
combine_data['dropout_status'] = combine_data['semester_3_status'].apply(lambda x: 0 if x == 'THO' else 1)

combine_data = combine_data.dropna()

df = combine_data[['dropout_status', 
                'semester_1_attendance_rate', "semester_1_average_score",
                'semester_2_attendance_rate', "semester_2_average_score",
                'semester_3_attendance_rate', "semester_3_average_score"]]

# Split the data into training/validation set and test set with ratio of 2/8
X_train_val,X_test,y_train_val,y_test = train_test_split(df.drop(['dropout_status'], axis=1),
                                                        df['dropout_status'], 
                                                        test_size=0.2, random_state=42)

# Split the training/validation set into training set and validation set with ratio of 2/8
X_train,X_val,y_train,y_val = train_test_split(X_train_val, y_train_val,
                                               test_size=0.2, random_state=42)


scaler = MinMaxScaler()

# Min-max scaling
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.fit_transform(X_val)
X_test_scaled = scaler.fit_transform(X_test)

metric_list = []

lr = LogisticRegression(random_state=42)

lr.fit(X_train_scaled, y_train)

y_val_pred = lr.predict(X_val_scaled)

print("Validation Set Evaluation")
print(confusion_matrix(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

y_test_pred = lr.predict(X_test_scaled)

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
output_path = 'Model Evaluation - Logistic Regression - No Resampling - 6 Features.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()



