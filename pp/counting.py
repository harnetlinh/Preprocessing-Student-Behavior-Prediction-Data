import pandas as pd

# Load the new dataset
file_path_new = 'sampled_students_data_new.csv'  # Update this path to your local file
data_new = pd.read_csv(file_path_new)

# Display the distribution of each label in 'status'
class_distribution = data_new['semester_3_status'].value_counts()
print("Class Distribution:\n", class_distribution)
