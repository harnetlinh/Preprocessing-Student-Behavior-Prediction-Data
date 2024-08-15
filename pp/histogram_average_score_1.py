# Full code to generate the histogram of Average Scores for Semester 1

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
combine_data = pd.read_csv('sampled_students_data_new.csv')

# Create a new column to differentiate dropout and non-dropout students
combine_data['dropout_status'] = combine_data['semester_3_status'].apply(lambda x: 'Dropout' if x == 'THO' else 'Non-Dropout')

# Histogram for Semester 1 Average Scores
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data=combine_data, x='semester_1_average_score', kde=True, hue='dropout_status', ax=ax)
ax.set_title('Distribution of Average Scores (Semester 1)')
ax.set_xlabel('Average Score')
ax.set_ylabel('Frequency')
ax.legend(title='Student Status On Semester 3')

# Save the figure
output_path_histogram = 'histogram_average_score_semester_1.png'
fig.savefig(output_path_histogram, bbox_inches='tight')
plt.show()

output_path_histogram
