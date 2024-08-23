import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re, ast

# Load the dataset
combine_data = pd.read_csv('sampled_students_data_new.csv')

# Create a new column to differentiate dropout and non-dropout students
combine_data['dropout_status'] = combine_data['semester_3_status'].apply(lambda x: 'Dropout' if x == 'THO' else 'Non-Dropout')

# Pattern to split the JSON formatted data
decimal_pattern = re.compile(r"Decimal\('(\d+\.\d+)'\)")

# Apply pattern on selected column
combine_data['semester_1'] = combine_data['semester_1'].apply(
    lambda x: decimal_pattern.sub(r"'\1'", str(x)))
combine_data['semester_2'] = combine_data['semester_2'].apply(
    lambda x: decimal_pattern.sub(r"'\1'", str(x)))
combine_data['semester_3'] = combine_data['semester_3'].apply(
    lambda x: decimal_pattern.sub(r"'\1'", str(x)))

df_arr, df_arr1, df_arr2, df_arr3 = [], [], [], []

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

# Box Plot Average Score
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='dropout_status', y='average_score', data=merged_df)
ax.set_title('Boxplot of Average Score')
ax.set_xlabel('Dropout Status')
ax.set_ylabel('Average Score')
plt.show()
# Save the figure
output_path = 'Boxplot of final Average Score.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()

# Box Plot Attendance Rate
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='dropout_status', y='attendance_rate', data=merged_df)
ax.set_title('Boxplot of Attendance Rate')
ax.set_xlabel('Dropout Status')
ax.set_ylabel('Attendance Rate')
plt.show()
# Save the figure
output_path = 'Boxplot of final Attendance Rate.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()

# Box Plot Passed Credit Percentage
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='dropout_status', y='passed_percent', data=merged_df)
ax.set_title('Boxplot of Passed Credit Percentage')
ax.set_xlabel('Dropout Status')
ax.set_ylabel('Passed Credit Percentage')
plt.show()
# Save the figure
output_path = 'Boxplot of final Passed Credit Percentage.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()

# Select variables to evaluate the relationship (Average Score Semester, Attedance Rate Semester, Dropout Status)
selected_df = merged_df[["average_score", "passed_percent", "attendance_rate", "dropout_status"]]

selected_df.loc[:, 'dropout_status'] = selected_df['dropout_status'].apply(lambda x: 0 if x == 'Dropout' else 1)

# Correlation matrix for Average Score Semester 1 and Attendance Rate Semester 1 versus Dropout Status
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(selected_df.corr(), annot=True, annot_kws={'size': 15}, cmap='crest', linewidths=.1)
ax.set_title('Correlation Matrix Heatmap of Average Score, Attendance Rate, Passed Credit Percentage with Dropout Status')
plt.show()
# Save the figure
output_path = 'Correlation Matrix Heatmap of Average Score, Attendance Rate, Passed Credit Percentage with Dropout Status.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()
