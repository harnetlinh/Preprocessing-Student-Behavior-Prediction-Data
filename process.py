import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import re

results_1 = pd.read_csv('results_1.csv')
results_2 = pd.read_csv('results_2.csv')
results_3 = pd.read_csv('results_3.csv')
results_4 = pd.read_csv('results_4.csv')
results_5 = pd.read_csv('results_5.csv')
results_6 = pd.read_csv('results_6.csv')

combine_data = pd.concat([results_1, results_2, results_3], axis=0)

combine_data["status"] = combine_data["semester_3_status"].apply(
    lambda x: 'THO' if x == 'THO' else 'HDI')

combine_data = combine_data.reset_index()

# Scatterplot từng kì
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
sns.scatterplot(x='semester_1_attendance_rate', y='semester_1_average_score',
                hue='status', data=combine_data, ax=axs[0])
sns.scatterplot(x='semester_2_attendance_rate', y='semester_2_average_score',
                hue='status', data=combine_data, ax=axs[1])
sns.scatterplot(x='semester_3_attendance_rate', y='semester_3_average_score',
                hue='status', data=combine_data, ax=axs[2])
plt.show()

# Histplot Average Score từng kì
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
sns.histplot(data=combine_data, x='semester_1_average_score', hue='status', kde=True, ax=axs[0])
sns.histplot(data=combine_data, x='semester_2_average_score', hue='status', kde=True, ax=axs[1])
sns.histplot(data=combine_data, x='semester_3_average_score', hue='status', kde=True, ax=axs[2])
plt.show()

# Histplot Attendance Rate từng kì
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
sns.histplot(data=combine_data, x='semester_1_attendance_rate', hue='status', kde=True, ax=axs[0])
sns.histplot(data=combine_data, x='semester_2_attendance_rate', hue='status', kde=True, ax=axs[1])
sns.histplot(data=combine_data, x='semester_3_attendance_rate', hue='status', kde=True, ax=axs[2])
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(10, 10))
sns.boxplot(x='status', y='semester_1_attendance_rate', data=combine_data, ax=axs[0])
sns.boxplot(x='status', y='semester_2_attendance_rate', data=combine_data, ax=axs[1])
sns.boxplot(x='status', y='semester_3_attendance_rate', data=combine_data, ax=axs[2])
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(10, 10))
sns.boxplot(x='status', y='semester_1_average_score', data=combine_data, ax=axs[0])
sns.boxplot(x='status', y='semester_1_average_score', data=combine_data, ax=axs[1])
sns.boxplot(x='status', y='semester_1_average_score', data=combine_data, ax=axs[2])
plt.show()

decimal_pattern = re.compile(r"Decimal\('(\d+\.\d+)'\)")

combine_data['semester_1'] = combine_data['semester_1'].apply(
    lambda x: decimal_pattern.sub(r"'\1'", str(x)))
combine_data['semester_2'] = combine_data['semester_2'].apply(
    lambda x: decimal_pattern.sub(r"'\1'", str(x)))
combine_data['semester_3'] = combine_data['semester_3'].apply(
    lambda x: decimal_pattern.sub(r"'\1'", str(x)))

df_arr = []

# TH xét theo từng kì
df_arr1 = []
df_arr2 = []
df_arr3 = []

for index, row in combine_data.iterrows():
    sem = ast.literal_eval(row['semester_1'])
    sem_df = pd.DataFrame.from_dict(sem, orient='index').reset_index()\
        .rename(columns={'index': 'subject_code'})
    sem_df['student_code'] = row['student_code']
    sem_df['status'] = row['status']
    sem_df['semester'] = 1
    df_arr.append(sem_df)
    df_arr1.append(sem_df)

for index, row in combine_data.iterrows():
    sem = ast.literal_eval(row['semester_2'])
    sem_df = pd.DataFrame.from_dict(sem, orient='index').reset_index()\
        .rename(columns={'index': 'subject_code'})
    sem_df['student_code'] = row['student_code']
    sem_df['status'] = row['status']
    sem_df['semester'] = 2
    df_arr.append(sem_df)
    df_arr2.append(sem_df)

for index, row in combine_data.iterrows():
    sem = ast.literal_eval(row['semester_3'])
    sem_df = pd.DataFrame.from_dict(sem, orient='index').reset_index()\
        .rename(columns={'index': 'subject_code'})
    sem_df['student_code'] = row['student_code']
    sem_df['status'] = row['status']
    sem_df['semester'] = 3
    df_arr.append(sem_df)
    df_arr3.append(sem_df)

combine_df = pd.concat(df_arr)
combine_df = combine_df.dropna()
combine_df["attendance_rate"] = combine_df["attendance_rate"].astype(float)
combine_df["average_score"] = combine_df["average_score"].astype(float)
combine_df['total_score'] = combine_df['average_score'] * combine_df['number_of_credit']

# Boxplot Average Score by Prefix
plt.figure(figsize=(14, 6))
sns.boxplot(x='subject_code', y='average_score', hue='status', data=combine_df)
plt.title('Boxplot Average Score by Prefix')
plt.xlabel('Prefix')
plt.ylabel('Average Score')
plt.legend(title='Status')
plt.show()

cg_df = combine_df.groupby('student_code').agg({
    'attendance_rate': 'mean',
    'number_of_credit': 'sum',
    'total_score': 'sum'
}).reset_index()

cg_df['average_score'] = cg_df['total_score'] / cg_df['number_of_credit']

merged_df = pd.merge(combine_data, cg_df, on='student_code')

merged_df = merged_df[['student_code', 'status', 'attendance_rate',
                       'average_score']]

# Scatterplot Average Score vs Attendance Percentage (tổng hợp)
plt.figure(figsize=(10, 10))
sns.scatterplot(x='average_score', y='attendance_rate',
                hue='status', data=merged_df)
plt.title('Scatter Plot of Average Score vs Attendance Percentage')
plt.xlabel('Average Score')
plt.ylabel('Attendance Percentage')
plt.show()

# Boxplot Average Score (tổng hợp)
plt.figure(figsize=(10, 10))
sns.boxplot(x='status', y='average_score', data=merged_df)
plt.title('Boxplot of Average Score')
plt.xlabel('Status')
plt.ylabel('Average Score')
plt.show()

# Histogram Average Score (tổng hợp)
plt.figure(figsize=(10, 10))
sns.histplot(x='average_score', hue='status', data=merged_df, kde=True)
plt.title('Histogram of Average Score')
plt.xlabel('Average Score')
plt.ylabel('Frequency')
plt.show()

# Cumulative Distribution Average Score
plt.figure(figsize=(10, 6))
sns.ecdfplot(data=merged_df, x='average_score', hue='status')
plt.title('Cumulative Distribution Plot of Average Score: THO vs HDI')
plt.xlabel('Average Score')
plt.ylabel('Cumulative Probability')
plt.show()

# Boxplot Attendance Rate (tổng hợp)
plt.figure(figsize=(10, 6))
sns.boxplot(x='status', y='attendance_rate', data=merged_df)
plt.title('Boxplot of Attendance Percentages: THO vs HDI')
plt.xlabel('Status Group')
plt.ylabel('Attendance Percentage')
plt.show()

# Histogram Attendance Rate (tổng hợp)
plt.figure(figsize=(10, 6))
sns.histplot(data=merged_df, x='attendance_rate', hue='status', kde=True)
plt.title('Histogram of Attendance Percentages: THO vs HDI')
plt.xlabel('Attendance Percentage')
plt.ylabel('Frequency')
plt.show()

# Cumulative Distribution Attendance Rate
plt.figure(figsize=(10, 6))
sns.ecdfplot(data=merged_df, x='attendance_rate', hue='status')
plt.title('Cumulative Distribution Plot of Attendance Percentages: THO vs HDI')
plt.xlabel('Attendance Percentages')
plt.ylabel('Cumulative Probability')
plt.show()

merged_df['status'] = merged_df['status'].apply(\
    lambda x: 0 if x == 'THO' else 1)
merged_df.drop('student_code', axis=1, inplace=True)

# Correlation Map tổng hợp
plt.figure(figsize=(10, 10))
sns.heatmap(merged_df.corr(), annot=True, cmap='crest', linewidth=.1)
plt.title('Correlation Heatmap')
plt.show()

# combine_df1 = pd.concat(df_arr1)
# combine_df2 = pd.concat(df_arr2)
# combine_df3 = pd.concat(df_arr3)
# ind_com = [combine_df1, combine_df2, combine_df3]

# for idx, cdf in enumerate(ind_com):
#     cdf = cdf.dropna()
#     cdf["attendance_rate"] = cdf["attendance_rate"].astype(float)
#     cdf["average_score"] = cdf["average_score"].astype(float)
#     plt.figure(figsize=(14, 6))
#     sns.boxplot(x='subject_code', y='average_score', hue='status', data=cdf)
#     plt.title(f'Boxplot Average Score Prefix Semester {idx}')
#     plt.xlabel('Subject Code')
#     plt.ylabel('Average Score')
#     plt.legend(title='Status')
#     plt.show()

unique_subjects = combine_df['subject_code'].unique()

for subject in unique_subjects:
    subject_data = combine_df[combine_df['subject_code'] == subject]
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    sns.scatterplot(x='attendance_rate', y='average_score', hue='status', data=subject_data, ax=axs[0])
    sns.histplot(data=subject_data, x='average_score', ax=axs[1], kde=True, hue='status')
    fig.suptitle(f'Subject {subject}')
    plt.show()

# Correlation Map Attendance Rate by Prefix only
pivot_df = combine_df.pivot(index=['student_code', 'status'], columns='subject_code', values=['attendance_rate'])
pivot_df.columns = ['{}_{}'.format(col[0], col[1]) for col in pivot_df.columns]
pivot_df.reset_index(inplace=True)
pivot_df['status'] = pivot_df['status'].apply(\
    lambda x: 0 if x == 'THO' else 1)
pivot_df.drop('student_code', axis=1, inplace=True)
plt.figure(figsize=(30, 30))
sns.heatmap(pivot_df.corr(), annot=True, cmap='crest', linewidth=.1)
plt.title('Correlation Heatmap')
plt.show()

# Correlation Map Average Score by Prefix only
pivot_df = combine_df.pivot(index=['student_code', 'status'], columns='subject_code', values=['average_score'])
pivot_df.columns = ['{}_{}'.format(col[0], col[1]) for col in pivot_df.columns]
pivot_df.reset_index(inplace=True)
pivot_df['status'] = pivot_df['status'].apply(\
    lambda x: 0 if x == 'THO' else 1)
pivot_df.drop('student_code', axis=1, inplace=True)
plt.figure(figsize=(30, 30))
sns.heatmap(pivot_df.corr(), annot=True, cmap='crest', linewidth=.1)
plt.title('Correlation Heatmap')
plt.show()

# Correlation Map of both Attendance Rate and Average Score by Prefix
pivot_df = combine_df.pivot(index=['student_code', 'status'], columns='subject_code', values=['attendance_rate', 'average_score'])
pivot_df.columns = ['{}_{}'.format(col[0], col[1]) for col in pivot_df.columns]
pivot_df.reset_index(inplace=True)
pivot_df['status'] = pivot_df['status'].apply(\
    lambda x: 0 if x == 'THO' else 1)
pivot_df.drop('student_code', axis=1, inplace=True)
plt.figure(figsize=(30, 30))
sns.heatmap(pivot_df.corr(), annot=True, cmap='crest', linewidth=.1)
plt.title('Correlation Heatmap')
plt.show()


