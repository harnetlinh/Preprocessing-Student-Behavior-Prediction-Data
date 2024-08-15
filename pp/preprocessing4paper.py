import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the new dataset
file_path_new = 'merged_results.csv'  # Update this path to your local file
data_new = pd.read_csv(file_path_new)

# Step 2: Encode student_code to hide it
le = LabelEncoder()
data_new['student_code'] = le.fit_transform(data_new['student_code'])

# Step 3: Check the class distribution
class_distribution = data_new['semester_3_status'].value_counts()
print("Class Distribution:\n", class_distribution)

# Step 4: Ensure each class has at least 2 members
min_class_size = 2
valid_classes = class_distribution[class_distribution >= min_class_size].index
data_new_filtered = data_new[data_new['semester_3_status'].isin(valid_classes)]

# Step 5: Perform stratified sampling to get 3000 records
sample_size = min(3000, len(data_new_filtered))
sampled_data_new, _ = train_test_split(data_new_filtered, train_size=sample_size, stratify=data_new_filtered['semester_3_status'], random_state=42)

# Step 6: Save the sampled data to a CSV file
output_file_path_new = 'sampled_students_data_new.csv'  # Update this path to where you want to save the file
sampled_data_new.to_csv(output_file_path_new, index=False)

print(f"Sampled data saved to: {output_file_path_new}")
