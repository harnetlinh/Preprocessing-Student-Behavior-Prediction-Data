import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import numpy as np
import re, ast
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.utils import resample
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE, ADASYN, SMOTENC, RandomOverSampler
from imblearn.under_sampling import TomekLinks, ClusterCentroids, NearMiss, RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.cluster import MiniBatchKMeans

"""
1. Thống kê cụ thể có bao nhiêu dropout và non-dropout
2. Tính tỉ lệ tín chỉ đã pass. 
Dựa vào learnt_time, nếu learnt_times = 2 và Passed = True thì tức là 1 lần fail và 1 lần pass, nhân số tín chỉ ra ta sẽ có tỉ lệ là 50% còn Passed = False thì tỉ lệ pass là 0%. Mình sẽ tính tổng số tín chỉ nhân từ learnt_times này, dựa theo cả trạng thái pass của môn. Em bổ sung thêm cột đó trong dữ liệu và plot thử correlation với boxplot
3. Chạy thử 1 vài phương pháp resampling trong đó cả under-sampling (Tomek Links, NearMiss, ClusterCentroids) và over-sampling (SMOTE, ADASYN,  SMOTEN)
4. Chạy thử confusion matrix trước và sau khi resampling với mô hình logistic, naive bayes
"""

# Load the dataset
combine_data = pd.read_csv('sampled_students_data_new.csv')

# Create a new column to differentiate dropout and non-dropout students
combine_data['dropout_status'] = combine_data['semester_3_status'].apply(lambda x: 0 if x == 'THO' else 1)
"""
task 1
"""
# status_counts = combine_data['dropout_status'].value_counts()
# plt.figure(figsize=(6, 6))
# plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
# plt.title('Dropout Status Distribution - Pie Chart')
# plt.show()
# plt.figure(figsize=(6, 6))

# status_counts.plot(kind='bar', color=['skyblue', 'lightcoral'])
# plt.title('Dropout Status Distribution - Column Chart')
# plt.xlabel('Dropout Status')
# plt.ylabel('Number of Students')
# plt.xticks(rotation=0)
# for idx, count in enumerate(status_counts):
#     plt.text(idx, count, str(count), ha='center', va='bottom', fontsize=12)
# plt.show()
"""
task 2 + 3
"""
# Pattern to split the JSON formatted data
decimal_pattern = re.compile(r"Decimal\('(\d+\.\d+)'\)")

# Apply pattern on selected column
combine_data['semester_1'] = combine_data['semester_1'].apply(
    lambda x: decimal_pattern.sub(r"'\1'", str(x)))

df_array = []

# Transform the data into normal format
for index, row in combine_data.iterrows():
    sem = ast.literal_eval(row['semester_1'])
    sem_df = pd.DataFrame.from_dict(sem, orient='index').reset_index()\
        .rename(columns={'index': 'subject_code'})
    sem_df['student_code'] = row['student_code']
    sem_df['dropout_status'] = row['dropout_status']
    sem_df['semester'] = 1
    df_array.append(sem_df)
combine_df = pd.concat(df_array)

combine_df["total_credit"] = combine_df["number_of_credit"] * combine_df["learnt_times"]

combine_df["credit_passed"] = (1 / combine_df["learnt_times"] * combine_df["total_credit"]).where(combine_df["passed"] == True, 0)

cg_df = combine_df.groupby('student_code').agg({
    'total_credit': 'sum',
    'credit_passed': 'sum'
}).reset_index()

cg_df["passed_percent"] = cg_df["credit_passed"] / cg_df["total_credit"] * 100

merged_df = pd.merge(combine_data, cg_df, on='student_code')

# # Box Plot for Semester 1
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.boxplot(x='dropout_status', y='passed_percent', data=merged_df)
# ax.set_title('Boxplot of Passed Credit Percentage for Semester 1')
# ax.set_xlabel('Dropout Status')
# ax.set_ylabel('Passed Credit Percentage')

# # Save the figure
# output_path = 'box_plot_semester_3_markers.png'
# fig.savefig(output_path, bbox_inches='tight')
# plt.show()

# # Select variables to evaluate the relationship (Average Score Semester, Attedance Rate Semester, Dropout Status)
# selected_df = merged_df[["passed_percent", "dropout_status"]]
# selected_df['dropout_status'] = selected_df['dropout_status'].apply(lambda x: 0 if x == 'Dropout' else 1)
# # Correlation matrix for Average Score Semester 1 and Attendance Rate Semester 1 versus Dropout Status
# fig, ax = plt.subplots(figsize=(14, 10))
# sns.heatmap(selected_df.corr(), annot=True, annot_kws={'size': 15}, cmap='crest', linewidths=.1)
# ax.set_title('Correlation Matrix Heatmap of Average Score and Attendance Rate with Dropout Status in Semester 1')
# plt.show()

# # Save the figure
# output_path = 'Correlation Matrix Heatmap of Average Score and Attendance Rate with Dropout Status in Semester 1.png'
# fig.savefig(output_path, bbox_inches='tight')
# plt.show()

merged_df.head

df = merged_df[['dropout_status', 
                'semester_1_attendance_rate', 
                'semester_1_average_score',
                'passed_percent']]
df = df.fillna(0)
# df['dropout_status'] = df['dropout_status'].apply(lambda x: 0 if x == 'Dropout' else 1)

# Split the data into training/validation set and test set with ratio of 2/8
X_train_val,X_test,y_train_val,y_test = train_test_split(df.drop(['dropout_status'], axis=1),
                                                        df['dropout_status'], 
                                                        test_size=0.2, random_state=42)

# Split the training/validation set into training set and validation set with ratio of 2/8
X_train,X_val,y_train,y_val = train_test_split(X_train_val, y_train_val,
                                               test_size=0.2, random_state=42)

"""
Under-sampling: TomekLinks
"""
# # Initializing TomekLinks()
# tl = TomekLinks()

# # Apply SMOTE for oversampling
# X_train_res, y_train_res = tl.fit_resample(X_train, y_train)

# df_resampled = pd.concat([X_train_res, y_train_res], axis=1)

# # Correlation Matrix to evaluate after resampling with SMOTE
# fig, ax = plt.subplots(figsize=(14, 10))
# sns.heatmap(df_resampled.corr(), annot=True, annot_kws={'size': 15}, cmap='crest', linewidths=.1)
# ax.set_title('Correlation Matrix Heatmap of Average Score, Attendance Rate, Passed Credit Percentage with Dropout Status in Semester 1 (After resampling)')
# plt.show()

# # Save the figure
# output_path = 'Correlation Matrix Heatmap of Average Score, Attendance Rate, Passed Credit Percentage with Dropout Status in Semester 1 (After resampling).png'
# fig.savefig(output_path, bbox_inches='tight')
# plt.show()

"""
Under-sampling: ClusterCentroids
"""
# # Initializing ClusterCentroids()
# cc = ClusterCentroids(
#     estimator=MiniBatchKMeans(n_init=1, random_state=0), random_state=42
# )
# # Apply SMOTE for oversampling
# X_train_res, y_train_res = cc.fit_resample(X_train, y_train)

# df_resampled = pd.concat([X_train_res, y_train_res], axis=1)

# # Correlation Matrix to evaluate after resampling with SMOTE
# fig, ax = plt.subplots(figsize=(14, 10))
# sns.heatmap(df_resampled.corr(), annot=True, annot_kws={'size': 15}, cmap='crest', linewidths=.1)
# ax.set_title('Correlation Matrix Heatmap of Average Score, Attendance Rate, Passed Credit Percentage with Dropout Status in Semester 1 (After resampling)')
# plt.show()

# # Save the figure
# output_path = 'Correlation Matrix Heatmap of Average Score, Attendance Rate, Passed Credit Percentage with Dropout Status in Semester 1 (After resampling).png'
# fig.savefig(output_path, bbox_inches='tight')
# plt.show()

"""
Under-sampling: NearMiss
"""
# # Initializing NearMiss()
# nm = NearMiss()

# # Apply SMOTE for oversampling
# X_train_res, y_train_res = nm.fit_resample(X_train, y_train)

# df_resampled = pd.concat([X_train_res, y_train_res], axis=1)

# # Correlation Matrix to evaluate after resampling with SMOTE
# fig, ax = plt.subplots(figsize=(14, 10))
# sns.heatmap(df_resampled.corr(), annot=True, annot_kws={'size': 15}, cmap='crest', linewidths=.1)
# ax.set_title('Correlation Matrix Heatmap of Average Score, Attendance Rate, Passed Credit Percentage with Dropout Status in Semester 1 (After resampling)')
# plt.show()

# # Save the figure
# output_path = 'Correlation Matrix Heatmap of Average Score, Attendance Rate, Passed Credit Percentage with Dropout Status in Semester 1 (After resampling).png'
# fig.savefig(output_path, bbox_inches='tight')
# plt.show()

"""
Under-sampling: RandomUnderSampler
"""
# # Initializing RandomUnderSampler
# rus = RandomUnderSampler()

# # Apply SMOTE for oversampling
# X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

# df_resampled = pd.concat([X_train_res, y_train_res], axis=1)

# # Correlation Matrix to evaluate after resampling with SMOTE
# fig, ax = plt.subplots(figsize=(14, 10))
# sns.heatmap(df_resampled.corr(), annot=True, annot_kws={'size': 15}, cmap='crest', linewidths=.1)
# ax.set_title('Correlation Matrix Heatmap of Average Score, Attendance Rate, Passed Credit Percentage with Dropout Status in Semester 1 (After resampling)')
# plt.show()

# # Save the figure
# output_path = 'Correlation Matrix Heatmap of Average Score, Attendance Rate, Passed Credit Percentage with Dropout Status in Semester 1 (After resampling).png'
# fig.savefig(output_path, bbox_inches='tight')
# plt.show()

"""
Over-sampling: SMOTE
"""
# # Initializing SMOTE
# smote = SMOTE(random_state=42)

# # Apply SMOTE for oversampling
# X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# df_resampled = pd.concat([X_train_res, y_train_res], axis=1)

# # Correlation Matrix to evaluate after resampling with SMOTE
# fig, ax = plt.subplots(figsize=(14, 10))
# sns.heatmap(df_resampled.corr(), annot=True, annot_kws={'size': 15}, cmap='crest', linewidths=.1)
# ax.set_title('Correlation Matrix Heatmap of Average Score, Attendance Rate, Passed Credit Percentage with Dropout Status in Semester 1 (After resampling)')
# plt.show()

# # Save the figure
# output_path = 'Correlation Matrix Heatmap of Average Score, Attendance Rate, Passed Credit Percentage with Dropout Status in Semester 1 (After resampling).png'
# fig.savefig(output_path, bbox_inches='tight')
# plt.show()
"""
Over-sampling: ADASYN
"""
# # Initializing ADASYN
# ada = ADASYN(random_state=42)

# # Apply SMOTE for oversampling
# X_train_res, y_train_res = ada.fit_resample(X_train, y_train)

# df_resampled = pd.concat([X_train_res, y_train_res], axis=1)

# # Correlation Matrix to evaluate after resampling with SMOTE
# fig, ax = plt.subplots(figsize=(14, 10))
# sns.heatmap(df_resampled.corr(), annot=True, annot_kws={'size': 15}, cmap='crest', linewidths=.1)
# ax.set_title('Correlation Matrix Heatmap of Average Score, Attendance Rate, Passed Credit Percentage with Dropout Status in Semester 1 (After resampling)')
# plt.show()

# # Save the figure
# output_path = 'Correlation Matrix Heatmap of Average Score, Attendance Rate, Passed Credit Percentage with Dropout Status in Semester 1 (After resampling).png'
# fig.savefig(output_path, bbox_inches='tight')
# plt.show()
"""
Over-sampling: SMOTENC
"""
# # Initializing SMOTENC
# sm = SMOTENC(random_state=42, categorical_features=[0])

# # Apply SMOTE for oversampling
# X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# df_resampled = pd.concat([X_train_res, y_train_res], axis=1)

# # Correlation Matrix to evaluate after resampling with SMOTE
# fig, ax = plt.subplots(figsize=(14, 10))
# sns.heatmap(df_resampled.corr(), annot=True, annot_kws={'size': 15}, cmap='crest', linewidths=.1)
# ax.set_title('Correlation Matrix Heatmap of Average Score, Attendance Rate, Passed Credit Percentage with Dropout Status in Semester 1 (After resampling)')
# plt.show()

# # Save the figure
# output_path = 'Correlation Matrix Heatmap of Average Score, Attendance Rate, Passed Credit Percentage with Dropout Status in Semester 1 (After resampling).png'
# fig.savefig(output_path, bbox_inches='tight')
# plt.show()
"""
Over-sampling: RandomOverSampler
"""
# # Initializing RandomOverSampler
# ros = RandomOverSampler(random_state=42)

# # Apply SMOTE for oversampling
# X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

# df_resampled = pd.concat([X_train_res, y_train_res], axis=1)

# # Correlation Matrix to evaluate after resampling with SMOTE
# fig, ax = plt.subplots(figsize=(14, 10))
# sns.heatmap(df_resampled.corr(), annot=True, annot_kws={'size': 15}, cmap='crest', linewidths=.1)
# ax.set_title('Correlation Matrix Heatmap of Average Score, Attendance Rate, Passed Credit Percentage with Dropout Status in Semester 1 (After resampling)')
# plt.show()

# # Save the figure
# output_path = 'Correlation Matrix Heatmap of Average Score, Attendance Rate, Passed Credit Percentage with Dropout Status in Semester 1 (After resampling).png'
# fig.savefig(output_path, bbox_inches='tight')
# plt.show()
