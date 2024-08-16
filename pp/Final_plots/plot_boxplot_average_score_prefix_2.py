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
combine_data['semester_2'] = combine_data['semester_2'].apply(
    lambda x: decimal_pattern.sub(r"'\1'", str(x)))

df_array = []

# Transform the data into normal format
for index, row in combine_data.iterrows():
    sem = ast.literal_eval(row['semester_2'])
    sem_df = pd.DataFrame.from_dict(sem, orient='index').reset_index()\
        .rename(columns={'index': 'subject_code'})
    sem_df['student_code'] = row['student_code']
    sem_df['dropout_status'] = row['dropout_status']
    sem_df['semester'] = 2
    df_array.append(sem_df)
combine_df = pd.concat(df_array)

# Box plot for semester 2
final_df = combine_df[["subject_code", "average_score", "dropout_status"]]
final_df.loc[:, "average_score"] = final_df["average_score"].astype(float)
final_df_reset = final_df.reset_index(drop=True)

fig, ax = plt.subplots(figsize=(14, 6))
sns.boxplot(x='subject_code', y='average_score', hue="dropout_status", data=final_df_reset)
ax.set_title('Boxplot of Average Score In Detail for Semester 2')
ax.set_xlabel('Subject Code')
ax.set_ylabel('Average Score')
ax.legend(title="Dropout Status", bbox_to_anchor=(1, 1), ncol=1)

# Save the figure
output_path = 'box_plot_semester_2_markers.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()