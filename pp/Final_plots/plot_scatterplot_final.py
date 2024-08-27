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

df_arr = []

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

df = merged_df[['dropout_status', 
                'attendance_rate', 
                'average_score',
                'passed_percent']]

# Scatter Plot for Final Attendance Rate and Average Score
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='attendance_rate', y='average_score', hue='dropout_status', style='dropout_status', markers=['X', 'o'], data=df, ax=ax)
ax.set_title('Scatter Plot of Final Attendance Rate vs. Average Score')
ax.set_xlabel('Attendance Rate')
ax.set_ylabel('Average Score')
ax.legend(title='Dropout Status')

# Save the figure
output_path = 'plot_scatterplot_attendance_rate_and_average_score_final.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()

# Scatter Plot for Final Attendance Rate and Passed Credit Percent
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='attendance_rate', y='passed_percent', hue='dropout_status', style='dropout_status', markers=['X', 'o'], data=df, ax=ax)
ax.set_title('Scatter Plot of Final Attendance Rate vs. Passed Credit Percent')
ax.set_xlabel('Attendance Rate')
ax.set_ylabel('Passed Credit Percent')
ax.legend(title='Dropout Status')

# Save the figure
output_path = 'plot_scatterplot_attendance_rate_and_passed_percent_final.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()

# Scatter Plot for Final Average Score and Passed Credit Percent
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='average_score', y='passed_percent', hue='dropout_status', style='dropout_status', markers=['X', 'o'], data=df, ax=ax)
ax.set_title('Scatter Plot of Final Average Score vs. Passed Credit Percent')
ax.set_xlabel('Average Score')
ax.set_ylabel('Passed Credit Percent')
ax.legend(title='Dropout Status')

# Save the figure
output_path = 'plot_scatterplot_average_score_and_passed_percent_final.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()

