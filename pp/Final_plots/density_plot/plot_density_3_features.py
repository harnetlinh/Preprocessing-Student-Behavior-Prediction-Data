import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import numpy as np
import re, ast
from scipy import stats
import matplotlib.lines as mlines

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

group_df = df.groupby('Dropout Status')

# Average Score statistics
mean_avg = group_df['average_score'].mean()
median_avg = group_df['average_score'].median()
mode_avg = group_df['average_score'].apply(lambda x: x.mode()[0])

# Attendance Rate statistics
mean_att = group_df['attendance_rate'].mean()
median_att = group_df['attendance_rate'].median()
mode_att = group_df['attendance_rate'].apply(lambda x: x.mode()[0])

# Passed Credit Rate statistics
mean_pass = group_df['passed_percent'].mean()
median_pass = group_df['passed_percent'].median()
mode_pass = group_df['passed_percent'].apply(lambda x: x.mode()[0])

mean_line = [mlines.Line2D([], [], color='red', linestyle='--', label='Non-Dropout Mean'),
             mlines.Line2D([], [], color='green', linestyle='--', label='Dropout Mean')]
median_line = [mlines.Line2D([], [], color='red', linestyle=':', label='Non-Dropout Median'),
               mlines.Line2D([], [], color='green', linestyle=':', label='Dropout Median')]
mode_line = [mlines.Line2D([], [], color='red', linestyle='-', label='Non-Dropout Mode'),
             mlines.Line2D([], [], color='green', linestyle='-', label='Dropout Mode')]

fig, axs = plt.subplots(1, 3, figsize=(30, 10))
sns.histplot(data=df, x="average_score", hue="Dropout Status", kde=True, ax=axs[0], stat='percent', common_norm=False)
axs[0].set_title("Distribution of Average Score", fontsize=18)
axs[0].set_xlabel("Average Score", fontsize=18)
axs[0].set_ylabel("", fontsize=18)
axs[0].tick_params(axis='x', labelsize=18)
axs[0].tick_params(axis='y', labelsize=18)
axs[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x / 100:.2f}'))

palette = sns.color_palette()
hue_labels = ['Non-dropout', 'Dropout']
hue_colors = [palette[i] for i in range(len(hue_labels))]
hue_handles = [mlines.Line2D([], [], color=color, marker='o', linestyle='', label=label)
               for color, label in zip(hue_colors, hue_labels)]

for status, color in zip(df['Dropout Status'].unique(), ['red', 'green']):
    axs[0].axvline(mean_avg[status], color=color, linestyle='--')
    axs[0].axvline(median_avg[status], color=color, linestyle=':')
    axs[0].axvline(mode_avg[status], color=color, linestyle='-')
axs[0].legend(handles=hue_handles, title='Dropout Status', loc='upper right')
axs[0].add_artist(axs[0].get_legend())
axs[0].legend(handles=mean_line + median_line + mode_line, title='Statistics', loc='upper left')

sns.histplot(data=df, x="attendance_rate", hue="Dropout Status", kde=True, ax=axs[1], stat='percent', common_norm=False)
axs[1].set_title("Distribution of Attendance Rate", fontsize=18)
axs[1].set_xlabel("Attendance Rate", fontsize=18)
axs[1].set_ylabel("", fontsize=18)
axs[1].tick_params(axis='x', labelsize=18)
axs[1].tick_params(axis='y', labelsize=18)
axs[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x / 100:.2f}'))
for status, color in zip(df['Dropout Status'].unique(), ['red', 'green']):
    axs[1].axvline(mean_att[status], color=color, linestyle='--')
    axs[1].axvline(median_att[status], color=color, linestyle=':')
    axs[1].axvline(mode_att[status], color=color, linestyle='-')
axs[1].legend(handles=hue_handles, title='Dropout Status', loc='upper right')
axs[1].add_artist(axs[1].get_legend())
axs[1].legend(handles=mean_line + median_line + mode_line, title='Statistics', loc='upper left')

sns.histplot(data=df, x="passed_percent", hue="Dropout Status", kde=True, ax=axs[2], stat='percent', common_norm=False)
axs[2].set_title("Distribution of Passed Credit Rate", fontsize=18)
axs[2].set_xlabel("Passed Credit Rate", fontsize=18)
axs[2].set_ylabel("", fontsize=18)
axs[2].tick_params(axis='x', labelsize=18)
axs[2].tick_params(axis='y', labelsize=18)
axs[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x / 100:.2f}'))
for status, color in zip(df['Dropout Status'].unique(), ['red', 'green']):
    axs[2].axvline(mean_pass[status], color=color, linestyle='--')
    axs[2].axvline(median_pass[status], color=color, linestyle=':')
    axs[2].axvline(mode_pass[status], color=color, linestyle='-')
axs[2].legend(handles=hue_handles, title='Dropout Status', loc='upper right')
axs[2].add_artist(axs[2].get_legend())
axs[2].legend(handles=mean_line + median_line + mode_line, title='Statistics', loc='upper left')

# Save the figure
output_path = 'Distribution of Average Score, Attendance Rate and Passed Credit Rate.png'
fig.savefig(output_path, bbox_inches='tight', dpi=300)
plt.show()