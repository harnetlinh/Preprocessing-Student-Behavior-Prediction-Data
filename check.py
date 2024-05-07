import os
import sys
import numpy as np
import pandas as pd
import mariadb
import sys
import time

terms = [24, 26, 27, 28, 29, 30, 33, 34, 35, 36, 37, 38, 39, 40, 50, 51, 52, 55, 58, 59]

ho_term_campus_terms = {
    26:24,
    28:26,
    29:27,
    30:28,
    31:29,
    34:30,
    38:33,
    43:34,
    44:35,
    45:36,
    46:37,
    47:38,
    48:39,
    49:40,
    50:50,
    51:51,
    52:52,
    55:55,
    58:58,
    59:59
}

def convert_ho_term_campus_term(term):
    if term in ho_term_campus_terms:
        return ho_term_campus_terms[term]
    return term

# hàm lấy kỳ ngay liền trước kỳ hiện tại từ terms và cả terms
def get_previous_term(term):
    index = terms.index(term)
    if index == 0:
        return None
    return terms[index - 1]


# lấy danh sách các kỳ học từ data có kỳ thứ < 3
def get_terms(data):
    res = []
    for i in range(len(data)):
        if int(data[i]["semester"]) <= 3:
            res.append(convert_ho_term_campus_term(data[i]["term_id"]))
    return res

# Lấy điểm trung bình theo môn học
def get_average_score(subject_code, student_code, list_term_ids):

    try:
        conn = mariadb.connect(
            user="hangoclinh",
            password="1111",
            host="localhost",
            port=3306,
            database="ap_hn20240502",
        )
    except mariadb.Error as e:
        print(f"Error connecting to MariaDB Platform: {e}")
        sys.exit(1)
    cur = conn.cursor()
    sql_array_ids = ""
    if len(list_term_ids) > 0:
        sql_array_ids = ", ".join([str(i) for i in list_term_ids])
    query = (
        """
            SELECT t7_course_result.grade, t7_course_result.val, fu_subject.num_of_credit
            FROM t7_course_result
            JOIN fu_user ON fu_user.user_login = t7_course_result.student_login
            JOIN fu_subject ON t7_course_result.psubject_code = fu_subject.subject_code
            WHERE fu_user.user_code = '"""
        + student_code
        + """'
                AND t7_course_result.term_id IN ("""
        + sql_array_ids
        + """)
                AND t7_course_result.skill_code = '"""
        + subject_code
        + """'
            ORDER BY t7_course_result.id DESC
            LIMIT 1
            """
    )
    cur.execute(query)
    # lấy dữ liệu từ câu lệnh sql
    res = cur.fetchall()
    # trả về điểm trung bình duy nhất, nếu không có trả về None
    if len(res) == 0:
        return None
    return res[0][0], int(res[0][1]) > 0, int(res[0][2])


# Lấy số lần học theo môn học
def get_learnt_times(subject_code, student_code, list_term_ids):
    try:
        conn = mariadb.connect(
            user="hangoclinh",
            password="1111",
            host="localhost",
            port=3306,
            database="ap_hn20240502",
        )
    except mariadb.Error as e:
        print(f"Error connecting to MariaDB Platform: {e}")
        sys.exit(1)
    cur = conn.cursor()
    sql_array_ids = ""
    if len(list_term_ids) > 0:
        sql_array_ids = ", ".join([str(i) for i in list_term_ids])
    query = (
        """
            SELECT count(*)
            FROM t7_course_result
            JOIN fu_user ON fu_user.user_login = t7_course_result.student_login
            WHERE fu_user.user_code = '"""
        + student_code
        + """'
                AND t7_course_result.term_id IN ("""
        + sql_array_ids
        + """)
                AND t7_course_result.skill_code = '"""
        + subject_code
        + """'
            """
    )
    cur.execute(query)
    # lấy dữ liệu từ câu lệnh sql
    res = cur.fetchall()
    # trả về điểm trung bình duy nhất, nếu không có trả về None
    if len(res) == 0:
        return None
    return res[0][0]


# convert periods to semester format array
def convert_periods_to_semester(periods):
    semester1 = []
    semester2 = []
    semester3 = []
    all_subjects = []
    for period in periods:
        if int(period["ordering"]) > 3:
            continue
        full_skill_code = period["skill_code"]
        # chỉ lấy 3 ký tự đầu của mã môn học
        skill_code = period["skill_code"][:3]
        # check xem môn học trong periods có môn học nào trùng 3 ký tự với 3 ký tự của mã môn học trong all_subjects không
        # nếu không thì thêm vào all_subjects
        exists_subject = None
        for subject in all_subjects:
            # chỉ check 3 ký tự đầu của cả 2
            if subject[:3] == skill_code:
                exists_subject = subject
        if exists_subject is None:
            all_subjects.append(skill_code)
        else:
            # phần số trong exists_subject
            text_number = exists_subject[3:]
            if text_number == "":
                number = 1
            else:
                number = int(text_number)
            # tăng số lên 1
            number += 1
            # thêm vào all_subjects
            all_subjects.append(f"{skill_code}{number}")
            skill_code = f"{skill_code}{number}"

        match int(period["ordering"]):
            case 1:
                semester1.append((skill_code, full_skill_code))
            case 2:
                semester2.append((skill_code, full_skill_code))
            case 3:
                semester3.append((skill_code, full_skill_code))
            case _:
                continue    
    return semester1, semester2, semester3


# lấy trạng thái học kỳ 3 của sinh viên
def get_semester3_status(list_status):
    # trả về status với semester = 3
    final_status = None
    count_time = 0
    for status in list_status:
        if status["semester"] == 3:
            count_time += 1
            final_status = status["status"]
    return final_status, count_time

# Lấy tỉ lệ điểm danh theo môn học
def get_attendance_rate(subject_code, student_code, list_term_ids):

    try:
        conn = mariadb.connect(
            user="hangoclinh",
            password="1111",
            host="localhost",
            port=3306,
            database="ap_hn20240502",
        )
    except mariadb.Error as e:
        print(f"Error connecting to MariaDB Platform: {e}")
        sys.exit(1)
    cur = conn.cursor()
    sql_array_ids = ""
    if len(list_term_ids) > 0:
        sql_array_ids = ", ".join([str(i) for i in list_term_ids])
    query = (
        """
        SELECT
        (COUNT(fu_attendance.activity_id) * 100.0 / COUNT(fu_activity.id)) AS attendance_rate
        FROM
        fu_group
        JOIN
        fu_group_member ON fu_group_member.groupid = fu_group.id
        JOIN
        fu_user ON fu_user.user_login = fu_group_member.member_login
        JOIN
        fu_activity ON fu_activity.groupid = fu_group.id
        LEFT JOIN
        fu_attendance ON fu_activity.id = fu_attendance.activity_id AND fu_attendance.user_login = fu_user.user_login AND fu_attendance.val = 1
        WHERE
        fu_user.user_code = '"""
        + student_code
        + """'
        AND fu_group.pterm_id IN ("""
        + sql_array_ids
        + """)
        AND fu_group.skill_code = '"""
        + subject_code
        + """'
        AND fu_activity.course_slot < 100
        GROUP BY fu_user.user_code
        """
    )

    # hoàn thành câu lệnh bằng cách thêm các giá trị vào câu lệnh sql
    cur.execute(query)

    # lấy dữ liệu từ câu lệnh sql
    res = cur.fetchall()
    # trả về tỉ lệ điểm danh duy nhất nếu có, nếu không trả về None
    if len(res) == 0:
        return None
    return res[0][0]

def check_if_subject_is_exempted(skill_code, student_code):
    try:
        conn = mariadb.connect(
            user="hangoclinh",
            password="1111",
            host="localhost",
            port=3306,
            database="ap_hn20240502",
        )
    except mariadb.Error as e:
        print(f"Error connecting to MariaDB Platform: {e}")
        sys.exit(1)
    cur = conn.cursor()
    query = "SELECT * FROM mien_giam_tap_trung WHERE user_code = '" + student_code + "' AND skill_code = '" + skill_code + "'"
    cur.execute(query)
    res = cur.fetchall()
    if len(res) > 0:
        return True
    return False

def get_info_elective(student_code, subject_code, curriculum_id, semester, list_term_ids):
    try:
        try:
            conn = mariadb.connect(
                user="hangoclinh",
                password="1111",
                host="localhost",
                port=3306,
                database="ap_hn20240502",
            )
        except mariadb.Error as e:
            print(f"Error connecting to MariaDB Platform: {e}")
            sys.exit(1)
        cur = conn.cursor()
        query_get_list_elective = "SELECT skill_code FROM fu_elective_group JOIN fu_elective_subject ON fu_elective_subject.elective_group_id = fu_elective_group.id WHERE curriculum_id = " + str(curriculum_id) + " AND ki_thu = " + str(semester)
        cur.execute(query_get_list_elective)
        res = cur.fetchall()
        list_elective = []
        for item in res:
            list_elective.append(item[0])
        if subject_code in list_elective:
            sql_array_term_ids = ""
            sql_array_skill_codes = ""
            grade = None
            val = None
            skil_code = None
            if len(list_term_ids) > 0:
                sql_array_term_ids = ", ".join([str(i) for i in list_term_ids])
            if len(list_elective) > 0:

                sql_array_skill_codes = ", ".join([f"'{i}'" for i in list_elective])
            query_get_info_result_of_elective_subject = """
                SELECT
                t7_course_result.grade, t7_course_result.val, t7_course_result.skill_code, fu_subject.num_of_credit
                FROM
                t7_course_result
                JOIN
                fu_user ON fu_user.user_login = t7_course_result.student_login
                JOIN fu_subject ON t7_course_result.psubject_code = fu_subject.subject_code
                WHERE
                fu_user.user_code = '""" + student_code + """'
                AND t7_course_result.term_id IN (""" + sql_array_term_ids + """)
                AND t7_course_result.skill_code IN (""" + sql_array_skill_codes + """)
                ORDER BY
                t7_course_result.id DESC
                LIMIT 1
            """
            cur.execute(query_get_info_result_of_elective_subject)
            res = cur.fetchall()
            if len(res) > 0:
                grade = res[0][0]
                val = res[0][1]
                skil_code = res[0][2]
                number_of_credit = res[0][3]
                query_attendance = (
                    """
                    SELECT
                    (COUNT(fu_attendance.activity_id) * 100.0 / COUNT(fu_activity.id)) AS attendance_rate
                    FROM
                    fu_group
                    JOIN
                    fu_group_member ON fu_group_member.groupid = fu_group.id
                    JOIN
                    fu_user ON fu_user.user_login = fu_group_member.member_login
                    JOIN
                    fu_activity ON fu_activity.groupid = fu_group.id
                    LEFT JOIN
                    fu_attendance ON fu_activity.id = fu_attendance.activity_id AND fu_attendance.user_login = fu_user.user_login AND fu_attendance.val = 1
                    WHERE
                    fu_user.user_code = '"""
                    + student_code
                    + """'
                    AND fu_group.pterm_id IN ("""
                    + sql_array_term_ids
                    + """)
                    AND fu_group.skill_code = '"""
                    + skil_code
                    + """'
                    AND fu_activity.course_slot < 100
                    GROUP BY fu_user.user_code
                    """
                )
                cur.execute(query_attendance)
                res = cur.fetchall()
                attendance_rate = None
                if len(res) > 0:
                    attendance_rate = res[0][0]
                return grade, int(val) > 0, attendance_rate, skil_code, number_of_credit
            return None
        return None
    except Exception as e:
        # print the error line
        _, _, tb = sys.exc_info()
        throw_error = f"Error on line {tb.tb_lineno}, {e}"
        # throw the error
        raise Exception(throw_error)

def get_attendance_rate_of_semester(student_code, list_term_ids, subject_codes):
    try:
        conn = mariadb.connect(
            user="hangoclinh",
            password="1111",
            host="localhost",
            port=3306,
            database="ap_hn20240502",
        )
    except mariadb.Error as e:
        print(f"Error connecting to MariaDB Platform: {e}")
        sys.exit(1)
    cur = conn.cursor()
    sql_array_term_ids = ""  
    if len(list_term_ids) > 0:
        sql_array_term_ids = ", ".join([str(i) for i in list_term_ids])
    sql_array_skill_codes = ""
    if len(subject_codes) > 0:
        sql_array_skill_codes = ", ".join([f"'{i}'" for i in subject_codes])
    query = (
                """
                SELECT
                (COUNT(fu_attendance.activity_id) * 100.0 / COUNT(fu_activity.id)) AS attendance_rate
                FROM
                fu_group
                JOIN
                fu_group_member ON fu_group_member.groupid = fu_group.id
                JOIN
                fu_user ON fu_user.user_login = fu_group_member.member_login
                JOIN
                fu_activity ON fu_activity.groupid = fu_group.id
                LEFT JOIN
                fu_attendance ON fu_activity.id = fu_attendance.activity_id AND fu_attendance.user_login = fu_user.user_login AND fu_attendance.val = 1
                WHERE
                fu_user.user_code = '"""
                + student_code
                + """'
                AND fu_group.pterm_id IN ("""
                + sql_array_term_ids
                + """)
                AND fu_group.skill_code IN ("""
                + sql_array_skill_codes
                + """)
                AND fu_activity.course_slot < 100
                GROUP BY fu_user.user_code
                """
            )
    cur.execute(query)
    res = cur.fetchall()
    if len(res) == 0:
        return None
    return res[0][0]

def get_average_score_of_semester(avg_all_subjects):
    try:
        total_score = 0
        total_credit = 0
        for subject_code, subject_info in avg_all_subjects.items():
            if subject_info['average_score'] is not None and subject_info['number_of_credit'] is not None:
                total_score += subject_info['average_score'] * subject_info['number_of_credit']
                total_credit += subject_info['number_of_credit']
        if total_credit == 0:
            return None
        return total_score / total_credit
    except Exception as e:
        # print the error line
        _, _, tb = sys.exc_info()
        throw_error = f"Error on line {tb.tb_lineno}, {e}"
        # throw the error
        raise Exception(throw_error)
        

# read file period_subjects.csv
df = pd.read_csv("period_subjects.csv")
# transform the data into a dictionary
data = df.to_dict(orient="records")
# get unique ids
ids = df["id"].unique()
# create a dictionary each id with a list of periods
periods = {}
for id in ids:
    _periods = []
    for row in data:
        if row["id"] == id and int(row["ordering"]) <= 3:
            _periods.append(row)
    # create a dictionary with the id and the list of periods
    periods[id] = _periods
# export the data to a json file
# with open("periods.json", "w") as f:

# nhập từ bàn phím tên file list_students.csv
file_name_number = input("Enter file name: ")
# kiểm tra file có tồn tại không
file_name = f"list_students_{file_name_number}.csv"
if not os.path.exists(file_name):
    print("File not found!")
    sys.exit(1)
# create a dictionary with data from the file list_students.csv
df = pd.read_csv(file_name)
data = df.to_dict(orient="records")
student_codes = []
# get unique student_code
student_codes = df["student_code"].unique()
# sắp xếp student_codes theo thứ tự giảm dần
student_codes = sorted(student_codes, reverse=True)
# đảm bảo item trong student_codes là không trùng nhau
student_codes = list(dict.fromkeys(student_codes))
# student_codes = list(dict.fromkeys(student_codes))
# get last 100 student_codes
# student_codes = student_codes[-5:]
# get data from the last 100 student_codes
students = {}
for student_code in student_codes:
    _student = []
    for row in data:
        if row["student_code"] == student_code:
            _student.append(row)
    # create a dictionary with the student_code and the student data
    students[student_code] = {
        "student": _student,
        "periods": periods[_student[0]["curriculum_id"]],
    }

# tạo dữ liệu kết quả
results = []
errors = []
process_time = []

# kiểm tra những sinh viên đã được xử lý và lưu vào file results.csv nếu có thì xóa khỏi student_codes và lưu vào results
if os.path.exists(f"results_{file_name_number}.csv"):
    try:
        # try to read the file, if it is not empty and there are more than 0 rows
        if os.stat(f"results_{file_name_number}.csv").st_size > 0:
            decision = input("File exists, do you want to overwrite or continue? (y/n): ")
            if decision.lower() != "y":
                df = pd.read_csv(f"results_{file_name_number}.csv")
                # check if the file is not empty and there is more than 0 rows
                if not df.empty and len(df) > 0:
                    results = df.to_dict(orient="records")
                    student_codes = [student_code for student_code in student_codes if student_code not in df["student_code"].unique()]
    except Exception as e:
        # get error line number
        _, _, tb = sys.exc_info()
        print("Error line number: ", tb.tb_lineno, "Error: ", e)
        sys.exit(1)
    


# duyệt qua từng sinh viên
for student_code in student_codes:
    # tính thời gian thực thi
    start = time.time()
    end = 0
    try:
        # lấy dữ liệu sinh viên
        student = {}
        student["student_code"] = student_code
        student["semester_1"] = {}
        student["semester_2"] = {}
        student["semester_3"] = {}
        student["semester_3_status"] = None

        # lấy dữ liệu sinh viên
        student_data = students[student_code]
        # lấy dữ liệu kỳ học 1 và 2
        semester1, semester2, semester3 = convert_periods_to_semester(student_data["periods"])
        # lấy trạng thái học kì 3
        semester3_status = get_semester3_status(student_data["student"])
        student["semester_3_status"] = semester3_status[0]
        student["semester_3_count"] = semester3_status[1]

        list_term_ids = get_terms(student_data["student"])
        list_previous_term_ids = [get_previous_term(term) for term in list_term_ids]
        # merge list_term_ids và list_previous_term_ids
        merged_list_term_ids = list(list_term_ids + list_previous_term_ids)
        max_term_id = max(merged_list_term_ids)
        index_max_term_id = terms.index(max_term_id)
        # get toàn bộ term_id từ terms nằm trước term_id lớn nhất
        list_external_term_ids = terms[:index_max_term_id]
        # merge merged_list_term_ids và list_external_term_ids
        merged_list_term_ids = list(merged_list_term_ids + list_external_term_ids)


        # lấy danh sách mã môn học của kỳ học 1
        for info in semester1:
            prefixed_subject_code, full_subject_code = info
            student["semester_1"][prefixed_subject_code] = {}
            student["semester_1"][prefixed_subject_code]["attendance_rate"] = (
                get_attendance_rate(
                    subject_code=full_subject_code,
                    student_code=student_code,
                    list_term_ids=merged_list_term_ids,
                )
            )
            subject_result = (
                get_average_score(
                    subject_code=full_subject_code,
                    student_code=student_code,
                    list_term_ids=merged_list_term_ids,
                )
            )
            if subject_result is None:
                student["semester_1"][prefixed_subject_code]["average_score"] = None
                student["semester_1"][prefixed_subject_code]["learnt_times"] = None
                student["semester_1"][prefixed_subject_code]["passed"] = False
                student["semester_1"][prefixed_subject_code]["number_of_credit"] = None
                if check_if_subject_is_exempted(full_subject_code, student_code):
                    student["semester_1"][prefixed_subject_code]["passed"] = True
                if student["semester_1"][prefixed_subject_code]["passed"] == False:
                    elective_info = get_info_elective(student_code, full_subject_code, student_data["student"][0]["curriculum_id"], 1, merged_list_term_ids)
                    if elective_info is not None:
                        student["semester_1"][prefixed_subject_code]["average_score"] = elective_info[0]
                        student["semester_1"][prefixed_subject_code]["passed"] = elective_info[1]
                        student["semester_1"][prefixed_subject_code]["attendance_rate"] = elective_info[2]
                        elective_subject = elective_info[3]
                        student["semester_1"][prefixed_subject_code]["learnt_times"] = get_learnt_times(
                            elective_subject, student_code, merged_list_term_ids
                        )
                        student["semester_1"][prefixed_subject_code]["number_of_credit"] = elective_info[4]
                        full_subject_code = elective_subject
            else:
                student["semester_1"][prefixed_subject_code]["average_score"] = subject_result[0]
                student["semester_1"][prefixed_subject_code]["passed"] = subject_result[1]
                student["semester_1"][prefixed_subject_code]["number_of_credit"] = subject_result[2]
                student["semester_1"][prefixed_subject_code]["learnt_times"] = get_learnt_times(
                    subject_code=full_subject_code,
                    student_code=student_code,
                    list_term_ids=merged_list_term_ids,
                )
            student["semester_1"][prefixed_subject_code][
                "full_subject_code"
            ] = full_subject_code
        actual_semester1_subject_codes = []
        for subject_code, subject_detail in student["semester_1"].items():
            actual_semester1_subject_codes.append(subject_detail["full_subject_code"])
        semester1_attendance_rate = get_attendance_rate_of_semester(student_code, merged_list_term_ids, actual_semester1_subject_codes)
        student["semester_1_attendance_rate"] = semester1_attendance_rate
        semester1_average_score = get_average_score_of_semester(student["semester_1"])
        student["semester_1_average_score"] = semester1_average_score

        # lấy danh sách mã môn học của kỳ học 2
        for info in semester2:
            prefixed_subject_code, full_subject_code = info
            student["semester_2"][prefixed_subject_code] = {}
            student["semester_2"][prefixed_subject_code]["attendance_rate"] = (
                get_attendance_rate(full_subject_code, student_code, merged_list_term_ids)
            )
            subject_result = get_average_score(
                full_subject_code, student_code, merged_list_term_ids
            )
            if subject_result is None:
                student["semester_2"][prefixed_subject_code]["average_score"] = None
                student["semester_2"][prefixed_subject_code]["learnt_times"] = None
                student["semester_2"][prefixed_subject_code]["passed"] = False
                if check_if_subject_is_exempted(full_subject_code, student_code):
                    student["semester_2"][prefixed_subject_code]["passed"] = True
                if student["semester_2"][prefixed_subject_code]["passed"] == False:
                    elective_info = get_info_elective(student_code, full_subject_code, student_data["student"][0]["curriculum_id"], 2, merged_list_term_ids)
                    if elective_info is not None:
                        student["semester_2"][prefixed_subject_code]["average_score"] = elective_info[0]
                        student["semester_2"][prefixed_subject_code]["passed"] = elective_info[1]
                        student["semester_2"][prefixed_subject_code]["attendance_rate"] = elective_info[2]
                        elective_subject = elective_info[3]
                        student["semester_2"][prefixed_subject_code]["number_of_credit"] = elective_info[4]
                        student["semester_2"][prefixed_subject_code]["learnt_times"] = get_learnt_times(
                            elective_subject, student_code, merged_list_term_ids
                        )
                        full_subject_code = elective_subject
            else:
                student["semester_2"][prefixed_subject_code]["average_score"] = subject_result[0]
                student["semester_2"][prefixed_subject_code]["passed"] = subject_result[1]
                student["semester_2"][prefixed_subject_code]["number_of_credit"] = subject_result[2]
                student["semester_2"][prefixed_subject_code]["learnt_times"] = get_learnt_times(
                    full_subject_code, student_code, merged_list_term_ids
                )
            student["semester_2"][prefixed_subject_code][
                "full_subject_code"
            ] = full_subject_code
        actual_semester2_subject_codes = []
        for subject_code, subject_detail in student["semester_2"].items():
            actual_semester2_subject_codes.append(subject_detail["full_subject_code"])
        semester2_attendance_rate = get_attendance_rate_of_semester(student_code, merged_list_term_ids, actual_semester2_subject_codes)
        student["semester_2_attendance_rate"] = semester2_attendance_rate
        semester2_average_score = get_average_score_of_semester(student["semester_2"])
        student["semester_2_average_score"] = semester2_average_score
            
        for info in semester3:
            prefixed_subject_code, full_subject_code = info
            elective_subject = None
            student["semester_3"][prefixed_subject_code] = {}
            student["semester_3"][prefixed_subject_code]["attendance_rate"] = (
                get_attendance_rate(full_subject_code, student_code, merged_list_term_ids)
            )
            subject_result = get_average_score(
                full_subject_code, student_code, merged_list_term_ids
            )
            if subject_result is None:
                student["semester_3"][prefixed_subject_code]["average_score"] = None
                student["semester_3"][prefixed_subject_code]["learnt_times"] = None
                student["semester_3"][prefixed_subject_code]["passed"] = False
                if check_if_subject_is_exempted(full_subject_code, student_code):
                    student["semester_3"][prefixed_subject_code]["passed"] = True
                if student["semester_3"][prefixed_subject_code]["passed"] == False:
                    elective_info = get_info_elective(student_code, full_subject_code, student_data["student"][0]["curriculum_id"], 3, merged_list_term_ids)
                    if elective_info is not None:
                        student["semester_3"][prefixed_subject_code]["average_score"] = elective_info[0]
                        student["semester_3"][prefixed_subject_code]["passed"] = elective_info[1]
                        student["semester_3"][prefixed_subject_code]["attendance_rate"] = elective_info[2]
                        elective_subject = elective_info[3]
                        student["semester_3"][prefixed_subject_code]["number_of_credit"] = elective_info[4]
                        student["semester_3"][prefixed_subject_code]["learnt_times"] = get_learnt_times(
                            elective_subject, student_code, merged_list_term_ids
                        )
                        full_subject_code = elective_subject
                        
            else:
                student["semester_3"][prefixed_subject_code]["average_score"] = subject_result[0]
                student["semester_3"][prefixed_subject_code]["passed"] = subject_result[1]
                student["semester_3"][prefixed_subject_code]["number_of_credit"] = subject_result[2]
                student["semester_3"][prefixed_subject_code]["learnt_times"] = get_learnt_times(
                    full_subject_code, student_code, merged_list_term_ids
                )
            student["semester_3"][prefixed_subject_code][
                "full_subject_code"
            ] = full_subject_code
        actual_semester3_subject_codes = []
        for subject_code, subject_detail in student["semester_3"].items():
            actual_semester3_subject_codes.append(subject_detail["full_subject_code"])
        semester3_attendance_rate = get_attendance_rate_of_semester(student_code, merged_list_term_ids, actual_semester3_subject_codes)
        student["semester_3_attendance_rate"] = semester3_attendance_rate
        semester3_average_score = get_average_score_of_semester(student["semester_3"])
        student["semester_3_average_score"] = semester3_average_score

        results.append(student)
        # thêm dữ liệu vào file results.csv 
        df = pd.DataFrame(results)
        df.to_csv(f"results_{file_name_number}.csv", index=False)
        end = time.time()
        process_time.append(end - start)
    except Exception as e:
        errors.append(student_code)
        # print the error line
        _, _, tb = sys.exc_info()
        print("Error processing student: ", student_code, "Error line number: ", tb.tb_lineno, "Error: ", e)
        # store the errors note in a file
        with open(f"errors_{file_name_number}.txt", "a") as f:
            f.write(f"{student_code}: on line {tb.tb_lineno}, {e}\n")
        end = time.time()
    # xóa console
    os.system("clear")
    # show tỉ lệ hoàn thành
    print("Progress: ", (len(results) + len(errors)) / len(student_codes) * 100, "%")
    # if end is not defined then print ***
    if np.isnan(end):
        print("Current time per student: *** s")
    else:
        print("Current time per student: ", end - start, "s")
    # trung bình thời gian thực thi
    avg = 0;
    if len(process_time) > 0:
        avg = np.mean(process_time)
    else:
        avg = 0
    print("Average time: ", avg)
    # dự đoán thời gian còn lại
    number_of_students = len(student_codes) - len(results) - len(errors)
    if number_of_students <= 0:
        remaining_time = 0
    else:
        remaining_time = np.mean(process_time) * (len(student_codes) - len(results) - len(errors))
    # nếu remaining_time là NaN thì gán bằng 0
    if np.isnan(remaining_time):
        remaining_time = 0
    # chuyển sang hh:mm:ss
    remaining_time = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
    print("Remaining time: ", remaining_time)
    
    # đóng kết nối


# show thống kê kết quả
os.system("clear")
print("Total students: ", len(student_codes))
print("Total errors: ", len(errors))
print("Total results: ", len(results))
print("Total process time: ", np.sum(process_time))
print("Average process time: ", np.mean(process_time))

