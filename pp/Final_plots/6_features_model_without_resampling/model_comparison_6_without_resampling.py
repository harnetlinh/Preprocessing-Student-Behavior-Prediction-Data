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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
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

"""
Logistic Regression
"""
lr = LogisticRegression(random_state=42)

lr.fit(X_train_scaled, y_train)

y_val_pred = lr.predict(X_val_scaled)

print("Validation Set Evaluation")
print(confusion_matrix(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

y_test_pred = lr.predict(X_test_scaled)

# Evaluate the model on the test set
print("Accuracy:", balanced_accuracy_score(y_test, y_test_pred))
print('Precision: ', precision_score(y_test, y_test_pred, average='macro'))
print('Recall: ', recall_score(y_test, y_test_pred, average='macro'))
print('F1: ', f1_score(y_test, y_test_pred, average='macro'))

metric_list.append({
    "Model": "Logistic Regression",
    "Accuracy": balanced_accuracy_score(y_test, y_test_pred),
    "Precision": precision_score(y_test, y_test_pred, average='macro'),
    "Recall": recall_score(y_test, y_test_pred, average='macro'),
    "F1 Score": f1_score(y_test, y_test_pred, average='macro')
})

sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d')
plt.title('Accuracy Score: {}'.format(balanced_accuracy_score(y_test, y_test_pred)))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

"""
Gaussian Naive Bayes
"""
nb = GaussianNB()

nb.fit(X_train_scaled, y_train)

y_val_pred = nb.predict(X_val_scaled)

print("Validation Set Evaluation")
print(confusion_matrix(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

y_test_pred = nb.predict(X_test_scaled)

# Evaluate the model on the test set
print("Accuracy:", balanced_accuracy_score(y_test, y_test_pred))
print('Precision: ', precision_score(y_test, y_test_pred, average='macro'))
print('Recall: ', recall_score(y_test, y_test_pred, average='macro'))
print('F1: ', f1_score(y_test, y_test_pred, average='macro'))

metric_list.append({
    "Model": "Gaussian Naive Bayes",
    "Accuracy": balanced_accuracy_score(y_test, y_test_pred),
    "Precision": precision_score(y_test, y_test_pred, average='macro'),
    "Recall": recall_score(y_test, y_test_pred, average='macro'),
    "F1 Score": f1_score(y_test, y_test_pred, average='macro')
})

sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d')
plt.title('Accuracy Score: {}'.format(balanced_accuracy_score(y_test, y_test_pred)))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

"""
Random Forest Classifier
"""
rf = RandomForestClassifier(random_state=42)

rf.fit(X_train_scaled, y_train)

y_val_pred = rf.predict(X_val_scaled)

print("Validation Set Evaluation")
print(confusion_matrix(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

y_test_pred = rf.predict(X_test_scaled)

# Evaluate the model on the test set
print("Accuracy:", balanced_accuracy_score(y_test, y_test_pred))
print('Precision: ', precision_score(y_test, y_test_pred, average='macro'))
print('Recall: ', recall_score(y_test, y_test_pred, average='macro'))
print('F1: ', f1_score(y_test, y_test_pred, average='macro'))

metric_list.append({
    "Model": "Random Forest Classifier",
    "Accuracy": balanced_accuracy_score(y_test, y_test_pred),
    "Precision": precision_score(y_test, y_test_pred, average='macro'),
    "Recall": recall_score(y_test, y_test_pred, average='macro'),
    "F1 Score": f1_score(y_test, y_test_pred, average='macro')
})

sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d')
plt.title('Accuracy Score: {}'.format(balanced_accuracy_score(y_test, y_test_pred)))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

"""
Balance Random Forest Classifier 
BRFC đã tự cân bằng tập dữ liệu trong quá trình xây dựng model, không sử dụng
các phương pháp resampling tại đây
(BRFC sử dụng undersampling)
"""
brf = BalancedRandomForestClassifier(random_state=42)

brf.fit(X_train_scaled, y_train)

# Evaluate on the validation set
y_val_pred = brf.predict(X_val_scaled)

# Classification report and confusion matrix for validation set
print("Validation Set Evaluation")
print(confusion_matrix(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

# Evaluate on the test set
y_test_pred = brf.predict(X_test_scaled)

# Evaluate the model on the test set
print("Accuracy:", balanced_accuracy_score(y_test, y_test_pred))
print('Precision: ', precision_score(y_test, y_test_pred, average='macro'))
print('Recall: ', recall_score(y_test, y_test_pred, average='macro'))
print('F1: ', f1_score(y_test, y_test_pred, average='macro'))

metric_list.append({
    "Model": "Balanced Random Forest Classifier",
    "Accuracy": balanced_accuracy_score(y_test, y_test_pred),
    "Precision": precision_score(y_test, y_test_pred, average='macro'),
    "Recall": recall_score(y_test, y_test_pred, average='macro'),
    "F1 Score": f1_score(y_test, y_test_pred, average='macro')
})

sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d')
plt.title('Accuracy Score: {}'.format(balanced_accuracy_score(y_test, y_test_pred)))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

"""
Support Vector Classifier
"""
svc = SVC(kernel='rbf', probability=True, random_state=42)

svc.fit(X_train_scaled, y_train)

# Evaluate on the validation set
y_val_pred = svc.predict(X_val_scaled)

# Classification report and confusion matrix for validation set
print("Validation Set Evaluation")
print(confusion_matrix(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

# Evaluate on the test set
y_test_pred = svc.predict(X_test_scaled)

# Evaluate the model on the test set
print("Accuracy:", balanced_accuracy_score(y_test, y_test_pred))
print('Precision: ', precision_score(y_test, y_test_pred, average='macro'))
print('Recall: ', recall_score(y_test, y_test_pred, average='macro'))
print('F1: ', f1_score(y_test, y_test_pred, average='macro'))

metric_list.append({
    "Model": "Support Vector Classifier",
    "Accuracy": balanced_accuracy_score(y_test, y_test_pred),
    "Precision": precision_score(y_test, y_test_pred, average='macro'),
    "Recall": recall_score(y_test, y_test_pred, average='macro'),
    "F1 Score": f1_score(y_test, y_test_pred, average='macro')
})

sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d')
plt.title('Accuracy Score: {}'.format(balanced_accuracy_score(y_test, y_test_pred)))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

metrics_df = pd.DataFrame(metric_list)

# Melt the DataFrame to long format
metrics_melted = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

fig, ax = plt.subplots(figsize=(15, 10))
sns.barplot(x="Model", y="Score", hue="Metric", data=metrics_melted)
ax.set_title("Model Comparison Based on Evaluation Metrics (6 Features)")
ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.legend(title="Metrics", loc="upper left", bbox_to_anchor=(1, 1))

# Save the figure
output_path = 'Model Comparison Based on Evaluation Metrics (6 Features) without resampling.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()

metrics_df.to_csv('model_comparison_6_without_resampling.csv', index=False)
