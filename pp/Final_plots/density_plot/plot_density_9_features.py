import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import numpy as np
import re, ast

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

df1, df2, df3 = [], [], []

# Transform the data into normal format
for index, row in combine_data.iterrows():
    sem = ast.literal_eval(row['semester_1'])
    sem_df = pd.DataFrame.from_dict(sem, orient='index').reset_index()\
        .rename(columns={'index': 'subject_code'})
    sem_df['student_code'] = row['student_code']
    sem_df['Dropout Status'] = row['Dropout Status']
    sem_df['semester'] = 1
    df1.append(sem_df)

for index, row in combine_data.iterrows():
    sem = ast.literal_eval(row['semester_2'])
    sem_df = pd.DataFrame.from_dict(sem, orient='index').reset_index()\
        .rename(columns={'index': 'subject_code'})
    sem_df['student_code'] = row['student_code']
    sem_df['Dropout Status'] = row['Dropout Status']
    sem_df['semester'] = 2
    df2.append(sem_df)

for index, row in combine_data.iterrows():
    sem = ast.literal_eval(row['semester_3'])
    sem_df = pd.DataFrame.from_dict(sem, orient='index').reset_index()\
        .rename(columns={'index': 'subject_code'})
    sem_df['student_code'] = row['student_code']
    sem_df['Dropout Status'] = row['Dropout Status']
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

df = merged_df[['Dropout Status', 
                'semester_1_attendance_rate', "semester_1_average_score", "semester_1_passed_percent",
                'semester_2_attendance_rate', "semester_2_average_score", "semester_2_passed_percent",
                'semester_3_attendance_rate', "semester_3_average_score", "semester_3_passed_percent"]]

fig, axs = plt.subplots(3, 3, figsize=(30, 30))
sns.kdeplot(data=df, x="semester_1_average_score", hue="Dropout Status", multiple="stack", ax=axs[0,0])
axs[0,0].set_title("Density Plot Average Score Semester 1", fontsize=20)
axs[0,0].set_xlabel("Average Score", fontsize=20)
axs[0,0].set_ylabel("Density", fontsize=20)
axs[0,0].tick_params(axis='x', labelsize=15)
axs[0,0].tick_params(axis='y', labelsize=15)

sns.kdeplot(data=df, x="semester_2_average_score", hue="Dropout Status", multiple="stack", ax=axs[0,1])
axs[0,1].set_title("Density Plot Average Score Semester 2", fontsize=20)
axs[0,1].set_xlabel("Average Score", fontsize=20)
axs[0,1].set_ylabel("Density", fontsize=20)
axs[0,1].tick_params(axis='x', labelsize=15)
axs[0,1].tick_params(axis='y', labelsize=15)

sns.kdeplot(data=df, x="semester_3_average_score", hue="Dropout Status", multiple="stack", ax=axs[0,2])
axs[0,2].set_title("Density Plot Average Score Semester 3", fontsize=20)
axs[0,2].set_xlabel("Average Score", fontsize=20)
axs[0,2].set_ylabel("Density", fontsize=20)
axs[0,2].tick_params(axis='x', labelsize=15)
axs[0,2].tick_params(axis='y', labelsize=15)

sns.kdeplot(data=df, x="semester_1_attendance_rate", hue="Dropout Status", multiple="stack", ax=axs[1,0])
axs[1,0].set_title("Density Plot Attendance Rate Semester 1", fontsize=20)
axs[1,0].set_xlabel("Attendance Rate", fontsize=20)
axs[1,0].set_ylabel("Density", fontsize=20)
axs[1,0].tick_params(axis='x', labelsize=15)
axs[1,0].tick_params(axis='y', labelsize=15)

sns.kdeplot(data=df, x="semester_2_attendance_rate", hue="Dropout Status", multiple="stack", ax=axs[1,1])
axs[1,1].set_title("Density Plot Attendance Rate Semester 2", fontsize=20)
axs[1,1].set_xlabel("Attendance Rate", fontsize=20)
axs[1,1].set_ylabel("Density", fontsize=20)
axs[1,1].tick_params(axis='x', labelsize=15)
axs[1,1].tick_params(axis='y', labelsize=15)

sns.kdeplot(data=df, x="semester_3_attendance_rate", hue="Dropout Status", multiple="stack", ax=axs[1,2])
axs[1,2].set_title("Density Plot Attendance Rate Semester 3", fontsize=20)
axs[1,2].set_xlabel("Attendance Rate", fontsize=20)
axs[1,2].set_ylabel("Density", fontsize=20)
axs[1,2].tick_params(axis='x', labelsize=15)
axs[1,2].tick_params(axis='y', labelsize=15)

sns.kdeplot(data=df, x="semester_1_passed_percent", hue="Dropout Status", multiple="stack", ax=axs[2,0])
axs[2,0].set_title("Density Plot Passed Credit Percentage Semester 1", fontsize=20)
axs[2,0].set_xlabel("Passed Credit Percentage", fontsize=20)
axs[2,0].set_ylabel("Density", fontsize=20)
axs[2,0].tick_params(axis='x', labelsize=15)
axs[2,0].tick_params(axis='y', labelsize=15)

sns.kdeplot(data=df, x="semester_2_passed_percent", hue="Dropout Status", multiple="stack", ax=axs[2,1])
axs[2,1].set_title("Density Plot Passed Credit Percentage Semester 2", fontsize=20)
axs[2,1].set_xlabel("Passed Credit Percentage", fontsize=20)
axs[2,1].set_ylabel("Density", fontsize=20)
axs[2,1].tick_params(axis='x', labelsize=15)
axs[2,1].tick_params(axis='y', labelsize=15)

sns.kdeplot(data=df, x="semester_3_passed_percent", hue="Dropout Status", multiple="stack", ax=axs[2,2])
axs[2,2].set_title("Density Plot Passed Credit Percentage Semester 3", fontsize=20)
axs[2,2].set_xlabel("Passed Credit Percentage", fontsize=20)
axs[2,2].set_ylabel("Density", fontsize=20)
axs[2,2].tick_params(axis='x', labelsize=15)
axs[2,2].tick_params(axis='y', labelsize=15)

# Save the figure
output_path = 'Density Plot of Average Score, Attendance Rate and Passed Credit Percentage through semesters.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()