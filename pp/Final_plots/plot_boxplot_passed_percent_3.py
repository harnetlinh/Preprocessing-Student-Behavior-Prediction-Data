import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re, ast

# Load the dataset
combine_data = pd.read_csv('sampled_students_data_new.csv')

# Create a new column to differentiate dropout and non-dropout students
combine_data['dropout_status'] = combine_data['semester_3_status'].apply(lambda x: "Dropout" if x == 'THO' else "Non-Dropout")

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

combine_df = combine_df.dropna()

# Calculate the percentage of passed credit
combine_df["total_credit"] = combine_df["number_of_credit"] * combine_df["learnt_times"]

combine_df["credit_passed"] = (1 / combine_df["learnt_times"] * combine_df["total_credit"]).where(combine_df["passed"] == True, 0)

cg_df = combine_df.groupby('student_code').agg({
    'total_credit': 'sum',
    'credit_passed': 'sum'
}).reset_index()

cg_df["passed_percent"] = cg_df["credit_passed"] / cg_df["total_credit"] * 100

merged_df = pd.merge(combine_data, cg_df, on='student_code')

# Box Plot for Semester 1
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='dropout_status', y='passed_percent', data=merged_df)
ax.set_title('Boxplot of Passed Credit Percentage for Semester 3')
ax.set_xlabel('Dropout Status')
ax.set_ylabel('Passed Credit Percentage')

# Save the figure
output_path = 'Boxplot of Passed Credit Percentage for Semester 3.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()
