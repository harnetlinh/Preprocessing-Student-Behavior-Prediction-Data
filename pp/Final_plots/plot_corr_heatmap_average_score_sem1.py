import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re, ast

# Load the dataset
combine_data = pd.read_csv('sampled_students_data_new.csv')

# Create a new column to differentiate dropout and non-dropout students
combine_data['dropout_status'] = combine_data['semester_3_status'].apply(lambda x: 0 if x == 'THO' else 1)

# Pattern to split the JSON formatted data
decimal_pattern = re.compile(r"Decimal\('(\d+\.\d+)'\)")

# Apply pattern on selected column
combine_data['semester_1'] = combine_data['semester_1'].apply(
    lambda x: decimal_pattern.sub(r"'\1'", str(x)))

df_array = []

# Transform the data into normal format
for index, row in combine_data.iterrows():
    sem = ast.literal_eval(row['semester_1'])
    sem_df = pd.DataFrame.from_dict(sem, orient='index').reset_index()\
        .rename(columns={'index': 'subject_code'})
    sem_df['student_code'] = row['student_code']
    sem_df['dropout_status'] = row['dropout_status']
    sem_df['semester'] = 1
    df_array.append(sem_df)
combine_df = pd.concat(df_array)

# # Correlation Map Average Score by Prefix only
pivot_df = combine_df.pivot(index=['student_code', 'dropout_status'], columns='subject_code', values=['average_score'])
pivot_df.columns = ['{}_{}'.format(col[0], col[1]) for col in pivot_df.columns]
pivot_df.reset_index(inplace=True)
pivot_df.drop('student_code', axis=1, inplace=True)

fig, ax = plt.subplots(figsize=(60, 50))
sns.heatmap(pivot_df.corr(), annot=True, annot_kws={'size': 36}, cmap='crest', linewidth=.1)
ax.set_title('Correlation Matrix Heatmap of Average Score with Dropout Status in Semester 1', fontsize=42)
ax.tick_params(axis='x', labelsize=28, labelrotation=45)
ax.tick_params(axis='y', labelsize=28, labelrotation=0)
ax.collections[0].colorbar.ax.tick_params(labelsize=28)

# Save the figure
output_path = 'Correlation Matrix Heatmap of Average Score with Dropout Status in Semester 1.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()