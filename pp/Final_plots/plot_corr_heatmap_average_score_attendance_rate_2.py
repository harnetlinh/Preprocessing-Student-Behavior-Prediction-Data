import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
combine_data = pd.read_csv('sampled_students_data_new.csv')

# Create a new column to differentiate dropout and non-dropout students (Encoding as 0 if Dropout and 1 if opposite)
combine_data['dropout_status'] = combine_data['semester_3_status'].apply(lambda x: 0 if x == 'THO' else 1)

# Select variables to evaluate the relationship (Average Score Semester, Attedance Rate Semester, Dropout Status)
selected_df = combine_data[["semester_2_average_score", "semester_2_attendance_rate", "dropout_status"]]

# Correlation matrix for Average Score Semester 2 and Attendance Rate Semester 2 versus Dropout Status
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(selected_df.corr(), annot=True, annot_kws={'size': 15}, cmap='crest', linewidths=.1)
ax.set_title('Correlation Matrix Heatmap of Average Score and Attendance Rate with Dropout Status in Semester 2')
plt.show()

# Save the figure
output_path = 'Correlation Matrix Heatmap of Average Score and Attendance Rate with Dropout Status in Semester 2.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()