import random

# Ex: lst = [1, 2, 3, 2, 4, 5, 1, 6, 3]
# Output:
# {
#     2: [1, 3],
#     1: [0, 6],
#     3: [2, 8]
# }
def find_duplicates(lst):
    duplicates = {}
    seen = {}
    
    for i, value in enumerate(lst):
        if value in seen:
            if value in duplicates:
                duplicates[value].append(i)
            else:
                duplicates[value] = [seen[value], i]
        else:
            seen[value] = i
    
    return duplicates

# Ex:     list = [1, 3, 5, 7]
#         num_sessions = [2, 2, 2, 2]
# Output: 0
# def count_overlap(list, num_sessions):
#     overlap_cases = 0
#     for i in range(len(list) - 1):
#         if num_sessions[i] == 0:
#             continue
#         for j in range(i + 1, len(list)):
#             if num_sessions[j] == 0:
#                 continue
#             start1, end1 = list[i], list[i] + num_sessions[i]
#             start2, end2 = list[j], list[j] + num_sessions[j]

#             if start1 < end2 and start2 < end1:
#                 overlap_cases += 1
#     return overlap_cases

def count_overlap(session_starts, num_sessions):
    session_ends = [start + num for start, num in zip(session_starts, num_sessions)]
    sorted_intervals = sorted(zip(session_starts, session_ends))
    overlaps = 0
    for i in range(len(sorted_intervals) - 1):
        if sorted_intervals[i][1] > sorted_intervals[i + 1][0]:
            overlaps += 1
    return overlaps

def weeks_overlap(week1, week2):
    """Return True if any week bit is 1 in both week1 and week2."""
    return any(w1 == '1' and w2 == '1' for w1, w2 in zip(week1, week2))

def get_same_semester_courses(course_id, df1, df2):
    semester = None
    for df in [df1, df2]:
        matched_row = df[df['course_id'] == course_id]
        if not matched_row.empty:
            semester = matched_row.iloc[0]['semester']
            break
    if semester is None:
        return []

    all_courses = set()
    for df in [df1, df2]:
        same_semester_rows = df[df['semester'] == semester]
        all_courses.update(same_semester_rows['course_id'].tolist())

    all_courses.discard(course_id)
    return list(all_courses)

def get_lecture_by_course_group(lecture_population, course_id, group_id):
    for lecture in lecture_population:
        if lecture.course_id == course_id and lecture.group_id == group_id:
            return lecture
    return None

def get_lectures_by_courses(lecture_population, course_ids):
    return [lec for lec in lecture_population if lec.course_id in course_ids]

def create_non_conflicting_time(lecture, same_semester_lectures, num_trials=20):
    for _ in range(num_trials):
        day_bin = format(random.randint(2, 7), '04b')
        session_start = random.choice([2, 8])
        session_bin = format(session_start, '04b')
        
        lab_day = day_bin
        lab_session_range = set(range(session_start, session_start + 5))  # sessions the lab will occupy

        conflict = False

        # Check own lecture
        if lecture:
            lec_day = lecture.bitstring[:4]
            lec_session = int(lecture.bitstring[4:8], 2)
            if lec_day == lab_day and lec_session in lab_session_range:
                conflict = True

        # Check other same-semester lectures
        if not conflict:
            for lec in same_semester_lectures:
                lec_day = lec.bitstring[:4]
                lec_session = int(lec.bitstring[4:8], 2)
                if lec_day == lab_day and lec_session in lab_session_range:
                    conflict = True
                    break

        if not conflict:
            return day_bin + session_bin  # safe to use

    return None  # fallback if no conflict-free time found

