import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
combine_data = pd.read_csv('sampled_students_data_new.csv')

# Create a new column to differentiate dropout and non-dropout students
combine_data['dropout_status'] = combine_data['semester_3_status'].apply(lambda x: 'Dropout' if x == 'THO' else 'Non-Dropout')

# Scatter Plot for Semester 1 with different markers
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='semester_2_attendance_rate', y='semester_2_average_score', hue='dropout_status', style='dropout_status', markers=['X', 'o'], data=combine_data, ax=ax)
ax.set_title('Scatter Plot of Attendance Rate vs. Average Score for Semester 2')
ax.set_xlabel('Attendance Rate')
ax.set_ylabel('Average Score')
ax.legend(title='Student Status On Semester 3')

# Save the figure
output_path = 'scatter_plot_semester_2_markers.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()
