import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

# Load the dataset
combine_data = pd.read_csv('sampled_students_data_new.csv')

# Create a new column to differentiate dropout and non-dropout students
combine_data['dropout_status'] = combine_data['semester_3_status'].apply(lambda x: 0 if x == 'THO' else 1)

df = combine_data[['student_code', 'dropout_status', 
                   'semester_1_attendance_rate', 'semester_1_average_score',
                   'semester_2_attendance_rate', 'semester_2_average_score',
                   'semester_3_attendance_rate', 'semester_3_average_score']]

df = df.drop('student_code', axis=1)
# Choose between drop all the NaN value or fill it with value 0
df = df.fillna(0)
# Consider scaling the data if needed with MinMaxScaler

# Split the data into training/validation set and test set with ratio of 2/8
X_train_val,X_test,y_train_val,y_test = train_test_split(df.drop(['dropout_status'], axis=1),
                                                        df['dropout_status'], 
                                                        test_size=0.2, random_state=42)

# Split the training/validation set into training set and validation set with ratio of 2/8
X_train,X_val,y_train,y_val = train_test_split(X_train_val, y_train_val,
                                               test_size=0.2, random_state=42)

# Initializing SMOTE
# Change the 'sampling_strategy' to a desire float value if needed - default = 1.0
smote = SMOTE(random_state=42)

# Apply SMOTE for oversampling
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

df_resampled = pd.concat([X_train_res, y_train_res], axis=1)

df_resampled_sem1 = df_resampled[["semester_1_attendance_rate", "semester_1_average_score", "dropout_status"]]

# Correlation Matrix to evaluate after resampling with SMOTE
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(df_resampled_sem1.corr(), annot=True, annot_kws={'size': 15}, cmap='crest', linewidths=.1)
ax.set_title('Correlation Matrix Heatmap of Average Score and Attendance Rate with Dropout Status in Semester 1 (After resampling)')
plt.show()

# Save the figure
output_path = 'Correlation Matrix Heatmap of Average Score and Attendance Rate with Dropout Status in Semester 1 (After resampling).png'
fig.savefig(output_path, bbox_inches='tight')
plt.show()