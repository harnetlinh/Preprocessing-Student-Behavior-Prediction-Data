import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re, ast

# Load the dataset
combine_data = pd.read_csv('sampled_students_data_new.csv')

# Create a new column to differentiate dropout and non-dropout students (Encoding as 0 if Dropout and 1 if opposite)
combine_data['dropout_status'] = combine_data['semester_3_status'].apply(lambda x: 0 if x == 'THO' else 1)

# Pattern to split the JSON formatted data
decimal_pattern = re.compile(r"Decimal\('(\d+\.\d+)'\)")

# Apply pattern on selected column
combine_data['semester_3'] = combine_data['semester_3'].apply(
    lambda x: decimal_pattern.sub(r"'\1'", str(x)))

df_array = []

# Transform the data into normal format
for index, row in combine_data.iterrows():
    sem = ast.literal_eval(row['semester_3'])
    sem_df = pd.DataFrame.from_dict(sem, orient='index').reset_index()\
        .rename(columns={'index': 'subject_code'})
    sem_df['student_code'] = row['student_code']
    sem_df['dropout_status'] = row['dropout_status']
    sem_df['semester'] = 3
    df_array.append(sem_df)
combine_df = pd.concat(df_array)

combine_df["total_credit_3"] = combine_df["number_of_credit"] * combine_df["learnt_times"]

combine_df["credit_passed_3"] = (1 / combine_df["learnt_times"] * combine_df["total_credit_3"]).where(combine_df["passed"] == True, 0)

cg_df = combine_df.groupby('student_code').agg({
    'total_credit_3': 'sum',
    'credit_passed_3': 'sum'
}).reset_index()

cg_df["passed_percent_3"] = cg_df["credit_passed_3"] / cg_df["total_credit_3"] * 100

merged_df = pd.merge(combine_data, cg_df, on='student_code')

# Select variables to evaluate the relationship (Average Score Semester, Attedance Rate Semester, Dropout Status)
selected_df = merged_df[["semester_3_average_score", "passed_percent_3", "semester_3_attendance_rate", "dropout_status"]]

# Correlation matrix for Average Score Semester 3 and Attendance Rate Semester 3 versus Dropout Status
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(selected_df.corr(), annot=True, annot_kws={'size': 15}, cmap='crest', linewidths=.1)
ax.set_title('Correlation Matrix Heatmap of Average Score, Attendance Rate, Passed Credit Percentage with Dropout Status in Semester 3.png')
plt.show()

# Save the figure
output_path = 'Correlation Matrix Heatmap of Average Score, Attendance Rate, Passed Credit Percentage with Dropout Status in Semester 3.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()