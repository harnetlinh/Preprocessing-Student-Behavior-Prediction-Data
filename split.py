# đọc file list_students.csv và tạo ra 3 file list_students_1.csv, list_students_2.csv, list_students_3.csv, chia theo student_code
# mỗi file chứa 1/3 số lượng sinh viên

import csv
import math

def split_file(file_name):
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        students = list(reader)
        # lấy danh sách student_code
        student_codes = [student[0] for student in students]
        # sắp xếp student_codes theo thứ tự giảm dần
        student_codes = sorted(student_codes, reverse=True)
        # đảm bảo item trong student_codes là không trùng nhau
        student_codes = list(dict.fromkeys(student_codes))
        # tính số lượng sinh viên trong mỗi file
        num_students = len(student_codes)
        num_students_per_file = math.ceil(num_students / 3)
        # chia student_codes thành 3 phần
        student_codes_1 = student_codes[:num_students_per_file]
        student_codes_2 = student_codes[num_students_per_file:2*num_students_per_file]
        student_codes_3 = student_codes[2*num_students_per_file:]
        # tạo 3 file mới lấy theo student_codes_1, student_codes_2, student_codes_3 đã chia ở trên
        with open('list_students_1.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            for student in students:
                if student[0] in student_codes_1:
                    writer.writerow(student)
        with open('list_students_2.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            for student in students:
                if student[0] in student_codes_2:
                    writer.writerow(student)
        with open('list_students_3.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            for student in students:
                if student[0] in student_codes_3:
                    writer.writerow(student)

if __name__ == '__main__':
    split_file('list_students.csv')

        