import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

df1, df2, df3 = [], [], []

# Transform the data into normal format
for index, row in combine_data.iterrows():
    sem = ast.literal_eval(row['semester_1'])
    sem_df = pd.DataFrame.from_dict(sem, orient='index').reset_index()\
        .rename(columns={'index': 'subject_code'})
    sem_df['student_code'] = row['student_code']
    sem_df['dropout_status'] = row['dropout_status']
    sem_df['semester'] = 1
    df1.append(sem_df)

for index, row in combine_data.iterrows():
    sem = ast.literal_eval(row['semester_2'])
    sem_df = pd.DataFrame.from_dict(sem, orient='index').reset_index()\
        .rename(columns={'index': 'subject_code'})
    sem_df['student_code'] = row['student_code']
    sem_df['dropout_status'] = row['dropout_status']
    sem_df['semester'] = 2
    df2.append(sem_df)

for index, row in combine_data.iterrows():
    sem = ast.literal_eval(row['semester_3'])
    sem_df = pd.DataFrame.from_dict(sem, orient='index').reset_index()\
        .rename(columns={'index': 'subject_code'})
    sem_df['student_code'] = row['student_code']
    sem_df['dropout_status'] = row['dropout_status']
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

df = merged_df[['dropout_status', 
                'semester_1_attendance_rate', "semester_1_average_score", "semester_1_passed_percent",
                'semester_2_attendance_rate', "semester_2_average_score", "semester_2_passed_percent",
                'semester_3_attendance_rate', "semester_3_average_score", "semester_3_passed_percent"]]

"""
Semester 1
"""
# Scatter Plot for Final Attendance Rate and Average Score
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='semester_1_attendance_rate', y='semester_1_average_score', hue='dropout_status', style='dropout_status', markers=['X', 'o'], data=df, ax=ax)
ax.set_title('Scatter Plot of Attendance Rate vs. Average Score in Semester 1')
ax.set_xlabel('Attendance Rate')
ax.set_ylabel('Average Score')
ax.legend(title='Dropout Status')
# Save the figure
output_path = 'plot_scatterplot_attendance_rate_and_average_score_semester1.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()


# Scatter Plot for Final Attendance Rate and Passed Credit Percent
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='semester_1_attendance_rate', y='semester_1_passed_percent', hue='dropout_status', style='dropout_status', markers=['X', 'o'], data=df, ax=ax)
ax.set_title('Scatter Plot of Attendance Rate vs. Passed Credit Percent in Semester 1')
ax.set_xlabel('Attendance Rate')
ax.set_ylabel('Passed Credit Percent')
ax.legend(title='Dropout Status')
# Save the figure
output_path = 'plot_scatterplot_attendance_rate_and_passed_percent_semester1.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()

# Scatter Plot for Final Average Score and Passed Credit Percent
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='semester_1_average_score', y='semester_1_passed_percent', hue='dropout_status', style='dropout_status', markers=['X', 'o'], data=df, ax=ax)
ax.set_title('Scatter Plot of Average Score vs. Passed Credit Percent in Semester 1')
ax.set_xlabel('Average Score')
ax.set_ylabel('Passed Credit Percent')
ax.legend(title='Dropout Status')
# Save the figure
output_path = 'plot_scatterplot_average_score_and_passed_percent_semester1.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()

"""
Semester 2
"""
# Scatter Plot for Final Attendance Rate and Average Score
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='semester_2_attendance_rate', y='semester_2_average_score', hue='dropout_status', style='dropout_status', markers=['X', 'o'], data=df, ax=ax)
ax.set_title('Scatter Plot of Attendance Rate vs. Average Score in Semester 2')
ax.set_xlabel('Attendance Rate')
ax.set_ylabel('Average Score')
ax.legend(title='Dropout Status')
# Save the figure
output_path = 'plot_scatterplot_attendance_rate_and_average_score_semester2.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()


# Scatter Plot for Final Attendance Rate and Passed Credit Percent
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='semester_2_attendance_rate', y='semester_2_passed_percent', hue='dropout_status', style='dropout_status', markers=['X', 'o'], data=df, ax=ax)
ax.set_title('Scatter Plot of Attendance Rate vs. Passed Credit Percent in Semester 2')
ax.set_xlabel('Attendance Rate')
ax.set_ylabel('Passed Credit Percent')
ax.legend(title='Dropout Status')
# Save the figure
output_path = 'plot_scatterplot_attendance_rate_and_passed_percent_semester2.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()

# Scatter Plot for Final Average Score and Passed Credit Percent
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='semester_2_average_score', y='semester_2_passed_percent', hue='dropout_status', style='dropout_status', markers=['X', 'o'], data=df, ax=ax)
ax.set_title('Scatter Plot of Average Score vs. Passed Credit Percent in Semester 2')
ax.set_xlabel('Average Score')
ax.set_ylabel('Passed Credit Percent')
ax.legend(title='Dropout Status')
# Save the figure
output_path = 'plot_scatterplot_average_score_and_passed_percent_semester2.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()

"""
Semester 3
"""
# Scatter Plot for Final Attendance Rate and Average Score
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='semester_3_attendance_rate', y='semester_3_average_score', hue='dropout_status', style='dropout_status', markers=['X', 'o'], data=df, ax=ax)
ax.set_title('Scatter Plot of Attendance Rate vs. Average Score in Semester 3')
ax.set_xlabel('Attendance Rate')
ax.set_ylabel('Average Score')
ax.legend(title='Dropout Status')
# Save the figure
output_path = 'plot_scatterplot_attendance_rate_and_average_score_semester3.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()


# Scatter Plot for Final Attendance Rate and Passed Credit Percent
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='semester_3_attendance_rate', y='semester_3_passed_percent', hue='dropout_status', style='dropout_status', markers=['X', 'o'], data=df, ax=ax)
ax.set_title('Scatter Plot of Attendance Rate vs. Passed Credit Percent in Semester 3')
ax.set_xlabel('Attendance Rate')
ax.set_ylabel('Passed Credit Percent')
ax.legend(title='Dropout Status')
# Save the figure
output_path = 'plot_scatterplot_attendance_rate_and_passed_percent_semester3.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()

# Scatter Plot for Final Average Score and Passed Credit Percent
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='semester_3_average_score', y='semester_3_passed_percent', hue='dropout_status', style='dropout_status', markers=['X', 'o'], data=df, ax=ax)
ax.set_title('Scatter Plot of Average Score vs. Passed Credit Percent in Semester 3')
ax.set_xlabel('Average Score')
ax.set_ylabel('Passed Credit Percent')
ax.legend(title='Dropout Status')
# Save the figure
output_path = 'plot_scatterplot_average_score_and_passed_percent_semester3.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()



