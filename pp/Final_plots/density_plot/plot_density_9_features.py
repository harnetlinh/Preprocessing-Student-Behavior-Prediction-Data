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
# plt.rcParams.update({'font.size': 28})
# plt.rcParams.update({'axes.labelsize': 28})
# plt.rcParams.update({'axes.titlesize': 28})
# plt.rcParams.update({'xtick.labelsize': 28})
# plt.rcParams.update({'ytick.labelsize': 28})
# plt.rcParams.update({'legend.fontsize': 24})

group_df = df.groupby('Dropout Status')

# Average Score Statistics
mean_avg1 = group_df['semester_1_average_score'].mean()
mean_avg2 = group_df['semester_2_average_score'].mean()
mean_avg3 = group_df['semester_3_average_score'].mean()

median_avg1 = group_df['semester_1_average_score'].median()
median_avg2 = group_df['semester_2_average_score'].median()
median_avg3 = group_df['semester_3_average_score'].median()

mode_avg1 = group_df['semester_1_average_score'].apply(lambda x: x.mode()[0])
mode_avg2 = group_df['semester_2_average_score'].apply(lambda x: x.mode()[0])
mode_avg3 = group_df['semester_3_average_score'].apply(lambda x: x.mode()[0])

# Attendance Rate Statistics
mean_att1 = group_df['semester_1_attendance_rate'].mean()
mean_att2 = group_df['semester_2_attendance_rate'].mean()
mean_att3 = group_df['semester_3_attendance_rate'].mean()

median_att1 = group_df['semester_1_attendance_rate'].median()
median_att2 = group_df['semester_2_attendance_rate'].median()
median_att3 = group_df['semester_3_attendance_rate'].median()

mode_att1 = group_df['semester_1_attendance_rate'].apply(lambda x: x.mode()[0])
mode_att2 = group_df['semester_2_attendance_rate'].apply(lambda x: x.mode()[0])
mode_att3 = group_df['semester_3_attendance_rate'].apply(lambda x: x.mode()[0])

# Passed Credit Rate Statistics
mean_pass1 = group_df['semester_1_passed_percent'].mean()
mean_pass2 = group_df['semester_2_passed_percent'].mean()
mean_pass3 = group_df['semester_3_passed_percent'].mean()

median_pass1 = group_df['semester_1_passed_percent'].median()
median_pass2 = group_df['semester_2_passed_percent'].median()
median_pass3 = group_df['semester_3_passed_percent'].median()

mode_pass1 = group_df['semester_1_passed_percent'].apply(lambda x: x.mode()[0])
mode_pass2 = group_df['semester_2_passed_percent'].apply(lambda x: x.mode()[0])
mode_pass3 = group_df['semester_3_passed_percent'].apply(lambda x: x.mode()[0])

mean_line = [mlines.Line2D([], [], color='red', linestyle='--', label='Non-Dropout Mean'),
             mlines.Line2D([], [], color='green', linestyle='--', label='Dropout Mean')]
median_line = [mlines.Line2D([], [], color='red', linestyle=':', label='Non-Dropout Median'),
               mlines.Line2D([], [], color='green', linestyle=':', label='Dropout Median')]
mode_line = [mlines.Line2D([], [], color='red', linestyle='-', label='Non-Dropout Mode'),
             mlines.Line2D([], [], color='green', linestyle='-', label='Dropout Mode')]

fig, axs = plt.subplots(3, 3, figsize=(45, 45))
sns.histplot(data=df, x="semester_1_average_score", hue="Dropout Status", kde=True, ax=axs[0,0], stat='percent', common_norm=False)
mean_avg1 = df['semester_1_average_score'].mean()
median_avg1 = df['semester_1_average_score'].median()
mode_avg1 = df['semester_1_average_score'].mode()[0]
axs[0,0].set_title("Distribution of Average Score Semester 1", fontsize=28)
axs[0,0].set_xlabel("Average Score", fontsize=28)
axs[0,0].set_ylabel("", fontsize=28)
axs[0,0].tick_params(axis='x', labelsize=28)
axs[0,0].tick_params(axis='y', labelsize=28)
axs[0,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x / 100:.2f}'))

palette = sns.color_palette()
hue_labels = ['Non-dropout', 'Dropout']
hue_colors = [palette[i] for i in range(len(hue_labels))]
hue_handles = [mlines.Line2D([], [], color=color, marker='o', linestyle='', label=label)
               for color, label in zip(hue_colors, hue_labels)]

for status, color in zip(df['Dropout Status'].unique(), ['red', 'green']):               
    axs[0,0].axvline(mean_avg1, color='red', linestyle='--')
    axs[0,0].axvline(median_avg1, color='green', linestyle='--')
    axs[0,0].axvline(mode_avg1, color='blue', linestyle='--')
axs[0,0].legend(handles=hue_handles, title='Dropout Status', loc='upper right')
axs[0,0].add_artist(axs[0,0].get_legend())
axs[0,0].legend(handles=mean_line + median_line + mode_line, title='Statistics', loc='upper left')

sns.histplot(data=df, x="semester_2_average_score", hue="Dropout Status", kde=True, ax=axs[0,1], stat='percent', common_norm=False)
mean_avg2 = df['semester_2_average_score'].mean()
median_avg2 = df['semester_2_average_score'].median()
mode_avg2 = df['semester_2_average_score'].mode()[0]
axs[0,1].set_title("Distribution of Average Score Semester 2", fontsize=28)
axs[0,1].set_xlabel("Average Score", fontsize=28)
axs[0,1].set_ylabel("", fontsize=28)
axs[0,1].tick_params(axis='x', labelsize=28)
axs[0,1].tick_params(axis='y', labelsize=28)
axs[0,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x / 100:.2f}'))

for status, color in zip(df['Dropout Status'].unique(), ['red', 'green']):
    axs[0,1].axvline(mean_avg2, color='red', linestyle='--')
    axs[0,1].axvline(median_avg2, color='green', linestyle='--')
    axs[0,1].axvline(mode_avg2, color='blue', linestyle='--')
axs[0,1].legend(handles=hue_handles, title='Dropout Status', loc='upper right')
axs[0,1].add_artist(axs[0,1].get_legend())
axs[0,1].legend(handles=mean_line + median_line + mode_line, title='Statistics', loc='upper left')

sns.histplot(data=df, x="semester_3_average_score", hue="Dropout Status", kde=True, ax=axs[0,2], stat='percent', common_norm=False)
mean_avg3 = df['semester_3_average_score'].mean()
median_avg3 = df['semester_3_average_score'].median()
mode_avg3 = df['semester_3_average_score'].mode()[0]
axs[0,2].set_title("Distribution of Average Score Semester 3", fontsize=28)
axs[0,2].set_xlabel("Average Score", fontsize=28)
axs[0,2].set_ylabel("", fontsize=28)
axs[0,2].tick_params(axis='x', labelsize=28)
axs[0,2].tick_params(axis='y', labelsize=28)
axs[0,2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x / 100:.2f}'))

for status, color in zip(df['Dropout Status'].unique(), ['red', 'green']):
    axs[0,2].axvline(mean_avg3, color='red', linestyle='--')
    axs[0,2].axvline(median_avg3, color='green', linestyle='--')
    axs[0,2].axvline(mode_avg3, color='blue', linestyle='--')
axs[0,2].legend(handles=hue_handles, title='Dropout Status', loc='upper right')
axs[0,2].add_artist(axs[0,2].get_legend())
axs[0,2].legend(handles=mean_line + median_line + mode_line, title='Statistics', loc='upper left')

sns.histplot(data=df, x="semester_1_attendance_rate", hue="Dropout Status", kde=True, ax=axs[1,0], stat='percent', common_norm=False)
mean_att1 = df['semester_1_attendance_rate'].mean()
median_att1 = df['semester_1_attendance_rate'].median()
mode_att1 = df['semester_1_attendance_rate'].mode()[0]
axs[1,0].set_title("Distribution of Attendance Rate Semester 1", fontsize=28)
axs[1,0].set_xlabel("Attendance Rate", fontsize=28)
axs[1,0].set_ylabel("", fontsize=28)
axs[1,0].tick_params(axis='x', labelsize=28)
axs[1,0].tick_params(axis='y', labelsize=28)
axs[1,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x / 100:.2f}'))

for status, color in zip(df['Dropout Status'].unique(), ['red', 'green']):
    axs[1,0].axvline(mean_att1, color='red', linestyle='--')
    axs[1,0].axvline(median_att1, color='green', linestyle='--')
    axs[1,0].axvline(mode_att1, color='blue', linestyle='--')
axs[1,0].legend(handles=hue_handles, title='Dropout Status', loc='upper right')
axs[1,0].add_artist(axs[1,0].get_legend())
axs[1,0].legend(handles=mean_line + median_line + mode_line, title='Statistics', loc='upper left')

sns.histplot(data=df, x="semester_2_attendance_rate", hue="Dropout Status", kde=True, ax=axs[1,1], stat='percent', common_norm=False)
mean_att2 = df['semester_2_attendance_rate'].mean()
median_att2 = df['semester_2_attendance_rate'].median()
mode_att2 = df['semester_2_attendance_rate'].mode()[0]
axs[1,1].set_title("Distribution of Attendance Rate Semester 2", fontsize=28)
axs[1,1].set_xlabel("Attendance Rate", fontsize=28)
axs[1,1].set_ylabel("", fontsize=28)
axs[1,1].tick_params(axis='x', labelsize=28)
axs[1,1].tick_params(axis='y', labelsize=28)
axs[1,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x / 100:.2f}'))

for status, color in zip(df['Dropout Status'].unique(), ['red', 'green']):
    axs[1,1].axvline(mean_att2, color='red', linestyle='--')
    axs[1,1].axvline(median_att2, color='green', linestyle='--')
    axs[1,1].axvline(mode_att2, color='blue', linestyle='--')
axs[1,1].legend(handles=hue_handles, title='Dropout Status', loc='upper right')
axs[1,1].add_artist(axs[1,1].get_legend())
axs[1,1].legend(handles=mean_line + median_line + mode_line, title='Statistics', loc='upper left')

sns.histplot(data=df, x="semester_3_attendance_rate", hue="Dropout Status", kde=True, ax=axs[1,2], stat='percent', common_norm=False)
mean_att3 = df['semester_3_attendance_rate'].mean()
median_att3 = df['semester_3_attendance_rate'].median()
mode_att3 = df['semester_3_attendance_rate'].mode()[0]
axs[1,2].set_title("Distribution of Attendance Rate Semester 3", fontsize=28)
axs[1,2].set_xlabel("Attendance Rate", fontsize=28)
axs[1,2].set_ylabel("", fontsize=28)
axs[1,2].tick_params(axis='x', labelsize=28)
axs[1,2].tick_params(axis='y', labelsize=28)
axs[1,2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x / 100:.2f}'))

for status, color in zip(df['Dropout Status'].unique(), ['red', 'green']):
    axs[1,2].axvline(mean_att3, color='red', linestyle='--')
    axs[1,2].axvline(median_att3, color='green', linestyle='--')
    axs[1,2].axvline(mode_att3, color='blue', linestyle='--')
axs[1,2].legend(handles=hue_handles, title='Dropout Status', loc='upper right')
axs[1,2].add_artist(axs[1,2].get_legend())
axs[1,2].legend(handles=mean_line + median_line + mode_line, title='Statistics', loc='upper left')

sns.histplot(data=df, x="semester_1_passed_percent", hue="Dropout Status", kde=True, ax=axs[2,0], stat='percent', common_norm=False)
mean_pass1 = df['semester_1_passed_percent'].mean()
median_pass1 = df['semester_1_passed_percent'].median()
mode_pass1 = df['semester_1_passed_percent'].mode()[0]
axs[2,0].set_title("Distribution of Passed Credit Rate Semester 1", fontsize=28)
axs[2,0].set_xlabel("Passed Credit Rate", fontsize=28)
axs[2,0].set_ylabel("", fontsize=28)
axs[2,0].tick_params(axis='x', labelsize=28)
axs[2,0].tick_params(axis='y', labelsize=28)
axs[2,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x / 100:.2f}'))

for status, color in zip(df['Dropout Status'].unique(), ['red', 'green']):
    axs[2,0].axvline(mean_pass1, color='red', linestyle='--')
    axs[2,0].axvline(median_pass1, color='green', linestyle='--')
    axs[2,0].axvline(mode_pass1, color='blue', linestyle='--')
axs[2,0].legend(handles=hue_handles, title='Dropout Status', loc='upper right')
axs[2,0].add_artist(axs[2,0].get_legend())
axs[2,0].legend(handles=mean_line + median_line + mode_line, title='Statistics', loc='upper left')

sns.histplot(data=df, x="semester_2_passed_percent", hue="Dropout Status", kde=True, ax=axs[2,1], stat='percent', common_norm=False)
mean_pass2 = df['semester_2_passed_percent'].mean()
median_pass2 = df['semester_2_passed_percent'].median()
mode_pass2 = df['semester_2_passed_percent'].mode()[0]
axs[2,1].set_title("Distribution of Passed Credit Rate Semester 2", fontsize=28)
axs[2,1].set_xlabel("Passed Credit Rate", fontsize=28)
axs[2,1].set_ylabel("", fontsize=28)
axs[2,1].tick_params(axis='x', labelsize=28)
axs[2,1].tick_params(axis='y', labelsize=28)
axs[2,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x / 100:.2f}'))

for status, color in zip(df['Dropout Status'].unique(), ['red', 'green']):
    axs[2,1].axvline(mean_pass2, color='red', linestyle='--')
    axs[2,1].axvline(median_pass2, color='green', linestyle='--')
    axs[2,1].axvline(mode_pass2, color='blue', linestyle='--')
axs[2,1].legend(handles=hue_handles, title='Dropout Status', loc='upper right')
axs[2,1].add_artist(axs[2,1].get_legend())
axs[2,1].legend(handles=mean_line + median_line + mode_line, title='Statistics', loc='upper left')

sns.histplot(data=df, x="semester_3_passed_percent", hue="Dropout Status", kde=True, ax=axs[2,2], stat='percent', common_norm=False)
mean_pass3 = df['semester_3_passed_percent'].mean()
median_pass3 = df['semester_3_passed_percent'].median()
mode_pass3 = df['semester_3_passed_percent'].mode()[0]
axs[2,2].set_title("Distribution of Passed Credit Rate Semester 3", fontsize=28)
axs[2,2].set_xlabel("Passed Credit Rate", fontsize=28)
axs[2,2].set_ylabel("", fontsize=28)
axs[2,2].tick_params(axis='x', labelsize=28)
axs[2,2].tick_params(axis='y', labelsize=28)
axs[2,2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x / 100:.2f}'))

for status, color in zip(df['Dropout Status'].unique(), ['red', 'green']):
    axs[2,2].axvline(mean_pass3, color='red', linestyle='--')
    axs[2,2].axvline(median_pass3, color='green', linestyle='--')
    axs[2,2].axvline(mode_pass3, color='blue', linestyle='--')
axs[2,2].legend(handles=hue_handles, title='Dropout Status', loc='upper right')
axs[2,2].add_artist(axs[2,2].get_legend())
axs[2,2].legend(handles=mean_line + median_line + mode_line, title='Statistics', loc='upper left')

# Save the figure
output_path = 'Distribution of Average Score, Attendance Rate and Passed Credit Rate through semesters.png'
fig.savefig(output_path, bbox_inches='tight', dpi=300)
plt.show()