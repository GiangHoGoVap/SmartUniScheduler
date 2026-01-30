import random
from model.individual import Individual
from config import CLASS_BLOCK_SIZE

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
            return [int(lab_day, 2), session_start]  # safe to use

    return None  # fallback if no conflict-free time found

def _pretty_table(title: str, data: dict, is_soft=False):
    if not data:
        return
    # longest key → dynamic width
    w = max(len(k) for k in data) + 2          # +2 for a little padding

    print("-" * (w + 12))
    print(title)
    for k, v in data.items():
        if is_soft:
            print(f"{k:<{w}} {v:>9.3f}")
        else:
            print(f"{k:<{w}} {v:>9}")

def convert_to_individuals(flat_individual, class_ids):
    individuals = []
    block_size = CLASS_BLOCK_SIZE  # 1 (day) + 1 (session) + 16 (weeks)
    num_classes = len(class_ids)

    for i in range(num_classes):
        start = i * block_size
        chunk = flat_individual[start:start + block_size]
        
        day = chunk[0]
        session = chunk[1]
        weeks = chunk[2:]

        day_bits = format(day, '04b')
        session_bits = format(session, '04b')
        week_bits = ''.join(str(w) for w in weeks)
        bitstring = day_bits + session_bits + week_bits

        course_id, group_id = class_ids[i].split('-')
        ind = Individual(course_id, group_id, bitstring)
        individuals.append(ind)

    return individuals