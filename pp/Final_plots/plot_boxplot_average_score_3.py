import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
combine_data = pd.read_csv('sampled_students_data_new.csv')

# Create a new column to differentiate dropout and non-dropout students
combine_data['dropout_status'] = combine_data['semester_3_status'].apply(lambda x: 'Dropout' if x == 'THO' else 'Non-Dropout')

# Box Plot for Semester 1
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='dropout_status', y='semester_3_average_score', data=combine_data)
ax.set_title('Boxplot of Average Score for Semester 3')
ax.set_xlabel('Dropout Status')
ax.set_ylabel('Average Score')

# Save the figure
output_path = 'box_plot_semester_3_markers.png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()