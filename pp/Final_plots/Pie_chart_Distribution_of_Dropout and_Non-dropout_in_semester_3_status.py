import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = '../sampled_students_data_new.csv'
df = pd.read_csv(file_path)

# Cập nhật nhãn thành Dropout và Non-dropout
df['simplified_status'] = df['semester_3_status'].apply(lambda x: 'Dropout' if x == 'THO' else 'Non-dropout')

# Tính toán tỷ lệ phân bố của Dropout và Non-dropout
simplified_status_counts = df['simplified_status'].value_counts(normalize=True) * 100

# Vẽ biểu đồ chuẩn để đưa vào bài báo khoa học
plt.figure(figsize=(8, 8))

# Tạo biểu đồ tròn với các tùy chỉnh phù hợp cho bài báo khoa học
plt.pie(simplified_status_counts, labels=simplified_status_counts.index, autopct='%1.1f%%',
        startangle=140, colors=['lightcoral', 'skyblue'], textprops={'fontsize': 12})

# Thêm tiêu đề
plt.title('Distribution of Dropout and Non-dropout in semester_3_status', fontsize=16)

# Đảm bảo rằng biểu đồ không bị kéo dài hoặc thu nhỏ
plt.axis('equal')

# Tối ưu hóa không gian xung quanh biểu đồ
plt.tight_layout()

# Lưu biểu đồ dưới dạng file hình ảnh với độ phân giải cao
plt.savefig('Pie_chart_distribution_dropout_non_dropout.png', dpi=300)

# Hiển thị biểu đồ
plt.show()
