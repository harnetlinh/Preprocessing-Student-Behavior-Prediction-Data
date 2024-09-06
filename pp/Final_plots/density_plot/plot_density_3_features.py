import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import numpy as np
import re, ast
from scipy import stats

# Load the dataset
combine_data = pd.read_csv('sampled_students_data_new.csv')

# Create a new column to differentiate dropout and non-dropout students
combine_data['Dropout Status'] = combine_data['semester_3_status'].apply(lambda x: "Dropout" if x == 'THO' else "Non-Dropout")

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
    sem_df['Dropout Status'] = row['Dropout Status']
    sem_df['semester'] = 1
    df_arr.append(sem_df)

for index, row in combine_data.iterrows():
    sem = ast.literal_eval(row['semester_2'])
    sem_df = pd.DataFrame.from_dict(sem, orient='index').reset_index()\
        .rename(columns={'index': 'subject_code'})
    sem_df['student_code'] = row['student_code']
    sem_df['Dropout Status'] = row['Dropout Status']
    sem_df['semester'] = 2
    df_arr.append(sem_df)

for index, row in combine_data.iterrows():
    sem = ast.literal_eval(row['semester_3'])
    sem_df = pd.DataFrame.from_dict(sem, orient='index').reset_index()\
        .rename(columns={'index': 'subject_code'})
    sem_df['student_code'] = row['student_code']
    sem_df['Dropout Status'] = row['Dropout Status']
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

df = merged_df[['Dropout Status', 
                'attendance_rate', 
                'average_score',
                'passed_percent']]

fig, axs = plt.subplots(1, 3, figsize=(30, 10))
sns.histplot(data=df, x="average_score", hue="Dropout Status", kde=True, ax=axs[0], stat='percent', common_norm=False)
mean_avg = df['average_score'].mean()
median_avg = df['average_score'].median()
mode_avg = df['average_score'].mode()[0]

axs[0].set_title("Distribution of Average Score", fontsize=20)
axs[0].set_xlabel("Average Score", fontsize=20)
axs[0].set_ylabel("", fontsize=20)
axs[0].tick_params(axis='x', labelsize=15)
axs[0].tick_params(axis='y', labelsize=15)
axs[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x / 100:.2f}'))
axs[0].axvline(mean_avg, color='red', linestyle='--', label=f'Mean: {mean_avg:.2f}')
axs[0].axvline(median_avg, color='green', linestyle='--', label=f'Median: {median_avg:.2f}')
axs[0].axvline(mode_avg, color='blue', linestyle='--', label=f'Mode: {mode_avg:.2f}')

sns.histplot(data=df, x="attendance_rate", hue="Dropout Status", kde=True, ax=axs[1], stat='percent', common_norm=False)
mean_att = df['attendance_rate'].mean()
median_att = df['attendance_rate'].median()
mode_att = df['attendance_rate'].mode()[0]

axs[1].set_title("Distribution of Attendance Rate", fontsize=20)
axs[1].set_xlabel("Attendance Rate", fontsize=20)
axs[1].set_ylabel("", fontsize=20)
axs[1].tick_params(axis='x', labelsize=15)
axs[1].tick_params(axis='y', labelsize=15)
axs[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x / 100:.2f}'))
axs[1].axvline(mean_att, color='red', linestyle='--', label=f'Mean: {mean_att:.2f}')
axs[1].axvline(median_att, color='green', linestyle='--', label=f'Median: {median_att:.2f}')
axs[1].axvline(mode_att, color='blue', linestyle='--', label=f'Mode: {mode_att:.2f}')

sns.histplot(data=df, x="passed_percent", hue="Dropout Status", kde=True, ax=axs[2], stat='percent', common_norm=False)
mean_pass = df['passed_percent'].mean()
median_pass = df['passed_percent'].median()
mode_pass = df['passed_percent'].mode()[0]

axs[2].set_title("Distribution of Passed Credit Rate", fontsize=20)
axs[2].set_xlabel("Passed Credit Rate", fontsize=20)
axs[2].set_ylabel("", fontsize=20)
axs[2].tick_params(axis='x', labelsize=15)
axs[2].tick_params(axis='y', labelsize=15)
axs[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x / 100:.2f}'))
axs[2].axvline(mean_pass, color='red', linestyle='--', label=f'Mean: {mean_pass:.2f}')
axs[2].axvline(median_pass, color='green', linestyle='--', label=f'Median: {median_pass:.2f}')
axs[2].axvline(mode_pass, color='blue', linestyle='--', label=f'Mode: {mode_pass:.2f}')

# Save the figure
output_path = 'Distribution of Average Score, Attendance Rate and Passed Credit Rate.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()