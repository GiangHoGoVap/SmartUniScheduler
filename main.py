### --- MARK: Imports ---
import os
import re
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygad

from model.individual import Individual
from model.utils import (
    get_same_semester_courses, get_lecture_by_course_group,
    get_lectures_by_courses, create_non_conflicting_time
)
from model.constraints import (
    ValidDayConstraint, SessionStartConstraint, LunchBreakConstraint,
    MidtermBreakConstraint, CourseDurationConstraint,
    CourseSameSemesterConstraint, CourseSameSemesterLabConstraint,
    LectureBeforeLabConstraint, LabSessionSpacingConstraint,
    ConstraintsManager
)

### --- MARK: Global Constants & Configs ---
SHEET_FILE_NAME = 'Dự kiến SVMT_242 data.xlsx'
SHEET_NAMES = ['Thống kê', 'KHMT', 'KTMT']
CLASS_BLOCK_SIZE = 18
VALID_DAYS = range(2, 7)
VALID_DAYS_LAB = range(2, 8)
VALID_SESSIONS = range(2, 12)

# ROOM, PROGRAM & COLUMN MAPPING DATA...
ROOMS = {
    'CO3093': {
        'CS1': ['C6-102', 'C6-103', 'C6-104', 'C6-509'],
        'CS2': ['H6-603', 'H6-604', 'H6-702', 'H6-703', 'H6-708']
    },
    'CO2013': {
        'CS1': ['C6-102', 'C6-103', 'C6-104', 'C6-509'],
        'CS2': ['H6-701', 'H6-702', 'H6-703', 'H6-707', 'H6-603', 'H6-604']
    },
    'CO2007': {
        'CS1': ['C5-202'],
        'CS2': ['H6-601', 'H6-605']
    },
    'CO2003': {
        'CS1': ['C6-102', 'C6-103', 'C6-104', 'C6-509'],
        'CS2': ['H6-701', 'H6-702', 'H6-703', 'H6-707', 'H6-603', 'H6-604']
    },
    'CO1023': {
        'CS1': ['C5-202', 'C6-105'],
        'CS2': ['H6-601', 'H6-605']
    },
    'CO3053': {
        'CS1': ['C6-105'],
        'CS2': ['H6-601']
    },
    'CO2017': {
        'CS1': ['C6-102', 'C6-103', 'C6-104', 'C6-509'],
        'CS2': ['H6-701', 'H6-702', 'H6-703', 'H6-707', 'H6-603', 'H6-604']
    },
    'CO3009': {
        'CS1': ['C5-202', 'C6-105'],
        'CS2': ['H6-601']
    },
    'CO2037': {
        'CS1': ['C5-202', 'C6-105'],
        'CS2': ['H6-601', 'H6-605']
    },
    'CO1005': {
        'CS1': ['C6-102', 'C6-103', 'C6-104', 'C6-509'],
        'CS2': ['H6-701', 'H6-702', 'H6-703', 'H6-707', 'H6-603', 'H6-604']
    }
}

PROGRAM_ID = {
    "Chương trình giảng dạy bằng tiếng Anh": "CC",
    "Chương trình tiêu chuẩn": "CQ",
    "Chương trình định hướng Nhật Bản": "CN"
}
REVERSE_PROGRAM_ID = {v: k for k, v in PROGRAM_ID.items()}

COLUMN_MAP = {
    "Mã môn học": "course_id",
    "Tên môn học": "course_name",
    "Loại hình lớp": "program_id",
    "Số SV dự kiến": "num_students",
    "Số lượng nhóm": "num_groups",
    "Sỉ số SV max": "max_students",
    "Tổng LT": "num_lectures",
    "Tổng TH": "num_lab_lectures",
    "Số tiết LT": "num_sessions",
    "Số tiết TH": "num_lab_sessions",
    "Học kỳ": "semester",
    "Thực hành": "is_lab"
}

COURSE_ID_TO_NAME = {}
ROOM_TYPE_ID = {}

lab_rooms = []
lecture_population = []

### --- MARK: Data Loading and Preprocessing ---
def read_excel(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name=sheet_name)

def clean_column(data, column_name):
    data[column_name] = data[column_name].str.replace(r'\s+', ' ', regex=True).str.strip()
    return data

def preprocess(df, index):
    global COURSE_ID_TO_NAME, ROOM_TYPE_ID

    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})

    if index == 0:
        df = df.dropna(subset=['Môn học TC/BB'])
        df = clean_column(df, 'program_id')
        COURSE_ID_TO_NAME = df.set_index('course_id')['course_name'].to_dict()
        ROOM_TYPE_ID = df.set_index('course_id')['max_students'].to_dict()

        df = df[['course_id', 'program_id', 'num_students', 'num_groups', 'max_students', 'is_lab']]
        df['program_id'] = df['program_id'].map(PROGRAM_ID)
        df['num_students'] = df['num_students'].astype(int)
        df['num_groups'] = df['num_groups'].astype(int)
        df['max_students'] = df['max_students'].astype(int)
        df['is_lab'] = df['is_lab'].apply(lambda x: 1 if x == 'x' else 0)

    elif index in (1, 2):
        df = df[['course_id', 'num_lectures', 'num_sessions', 'num_lab_lectures', 'num_lab_sessions', 'semester']]
        df = df.astype({
            'num_lectures': int, 'num_sessions': int,
            'num_lab_lectures': int, 'num_lab_sessions': int, 'semester': int
        })
        df['num_weeks'] = (df['num_lectures'] / df['num_sessions']).astype(int)
        df['num_weeks_lab'] = df.apply(
            lambda r: int(r['num_lab_lectures'] / r['num_lab_sessions']) if r['num_lab_sessions'] else 0, axis=1
        )

    return df

### --- MARK: Encoding and Constraint Setup ---
def encode(df):
    lectures, labs = [], []
    for row in df.itertuples():
        for group in range(1, row.num_groups + 1):
            lecture_id = f"{row.course_id}-{row.program_id}{str(group).zfill(2)}"
            lectures.append(lecture_id)
            if row.is_lab:
                for lab_num in range(1, math.ceil(row.max_students / 40) + 1):
                    lab_id = f"{row.course_id}-LAB{lab_num}-{row.program_id}{str(group).zfill(2)}"
                    labs.append(lab_id)
    return lectures, labs

def initialize_constraints(df1, df2, is_lab=False):
    cm = ConstraintsManager()
    cm.add_constraint(MidtermBreakConstraint())

    if is_lab:
        cm.add_constraint(ValidDayConstraint(VALID_DAYS_LAB, True))
        cm.add_constraint(SessionStartConstraint(df1, True))
        cm.add_constraint(SessionStartConstraint(df2, True))
        cm.add_constraint(CourseDurationConstraint(df1, True))
        cm.add_constraint(CourseDurationConstraint(df2, True))
        cm.add_constraint(CourseSameSemesterLabConstraint(df1, df_course_lookup))
        cm.add_constraint(CourseSameSemesterLabConstraint(df2, df_course_lookup))
        cm.add_constraint(LectureBeforeLabConstraint())
        cm.add_constraint(LabSessionSpacingConstraint())
    else:
        cm.add_constraint(ValidDayConstraint(VALID_DAYS))
        cm.add_constraint(SessionStartConstraint(df1))
        cm.add_constraint(SessionStartConstraint(df2))
        cm.add_constraint(LunchBreakConstraint())
        cm.add_constraint(CourseDurationConstraint(df1))
        cm.add_constraint(CourseDurationConstraint(df2))
        cm.add_constraint(CourseSameSemesterConstraint(df1))
        cm.add_constraint(CourseSameSemesterConstraint(df2))

    return cm

### --- MARK: Population Initialization ---
def create_initial_population(pop_size, class_ids, df1, df2):
    population = []
    for _ in range(pop_size):
        individual = []
        for class_id in class_ids:
            course_id, _ = class_id.split('-')
            day = np.random.choice(VALID_DAYS)
            session = random.choice(list(VALID_SESSIONS))
            for df in [df1, df2]:
                if course_id in df['course_id'].values:
                    num_sessions = df[df['course_id'] == course_id]['num_sessions'].values[0]
                    while session + num_sessions - 1 > max(VALID_SESSIONS):
                        session = random.choice(list(VALID_SESSIONS))
            weeks = [1] * 16
            weeks[-9] = 0
            individual.extend([day, session] + weeks)
        population.append(individual)
    return np.array(population)

def create_initial_population_lab(pop_size, class_ids, df1, df2, lecture_population):
    population = []
    for _ in range(pop_size):
        individual = []
        for class_id in class_ids:
            course_id, lab_id, group_id = class_id.split('-')
            same_semester_courses = get_same_semester_courses(course_id, df1, df2)
            own_lecture = get_lecture_by_course_group(lecture_population, course_id, group_id)
            same_semester_lectures = get_lectures_by_courses(lecture_population, same_semester_courses)
            time_bits = create_non_conflicting_time(own_lecture, same_semester_lectures)
            if time_bits is None:
                return None
            num_weeks_lab = 0
            for df in [df1, df2]:
                for _, row in df.iterrows():
                    if row['course_id'] == course_id:
                        num_weeks_lab = row['num_weeks_lab']
                        break
            lecture_start_week = 0
            for lecture in lecture_population:
                lecture_course_id = lecture.course_id
                lecture_group_id = lecture.group_id
                lecture_bitstring = lecture.bitstring
                if lecture_course_id == course_id and lecture_group_id == group_id:
                    lecture_weeks = list(lecture_bitstring[8:])
                    if '1' in lecture_weeks:
                        lecture_start_week = lecture_weeks.index('1')
                        break
            min_start_week = lecture_start_week + 3
            available_weeks = [w for w in range(min_start_week, 16) if w != 7]

            last_part = [0] * 16
            selected_weeks = []

            if not available_weeks:
                return None
            
            first_week = random.choice(available_weeks)
            selected_weeks.append(first_week)
            available_weeks.remove(first_week)

            parity = first_week % 2
            available_weeks = [w for w in available_weeks if w % 2 == parity]
            while len(selected_weeks) < num_weeks_lab and available_weeks:
                week = random.choice(available_weeks)
                selected_weeks.append(week)
                available_weeks.remove(week)
            for week in selected_weeks:
                last_part[week] = 1

            room_type = 'CS1' if group_id.startswith('CC') or group_id.startswith('CN') else 'CS2'
            room_list = ROOMS.get(course_id, {}).get(room_type, [])
            if not room_list:
                raise ValueError(f"No rooms available for course {course_id} with group type {room_type}")

            room_id = random.choice(room_list)
            lab_rooms.append(room_id)

            individual.extend(time_bits + last_part)
        population.append(individual)
    return np.array(population)

### --- MARK: Genetic Representation Conversion ---
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

def decode_solution(solution, class_ids, is_lab=False, room_list=None):
    individuals = []

    for i, class_id in enumerate(class_ids):
        start = i * CLASS_BLOCK_SIZE
        chunk = solution[start:start + CLASS_BLOCK_SIZE]
        day, session, weeks = chunk[0], chunk[1], chunk[2:]

        day_bits = format(day, '04b')
        session_bits = format(session, '04b')
        week_bits = ''.join(str(w) for w in weeks)
        bitstring = day_bits + session_bits + week_bits

        if is_lab:
            course_id, lab_id, group_id = class_id.split('-')
            room = room_list[i] if room_list else None
            full_group_id = f"{lab_id}-{group_id}"
            individuals.append(Individual(course_id, full_group_id, bitstring, "lab", room))
        else:
            course_id, group_id = class_id.split('-')
            individuals.append(Individual(course_id, group_id, bitstring))

    return individuals

### --- MARK: Fitness Evaluation ---
def evaluate_fitness_generic(individuals, constraints_manager, is_lab=False, lecture_population=None):
    scores = [constraints_manager.evaluate(ind) * 0.5 for ind in individuals]

    if is_lab:
        pop_scores = constraints_manager.evaluate_population(individuals, is_lab=True, lecture_population=lecture_population)
    else:
        pop_scores = constraints_manager.evaluate_population(individuals)

    for i in range(len(scores)):
        scores[i] += pop_scores[i] * 0.5
        scores[i] *= 1 / len(individuals)

    return sum(scores)

def fitness_func(ga_instance, solution, solution_idx):
    individuals = decode_solution(solution, chromosomes_df, False)
    return -evaluate_fitness_generic(individuals, constraints_manager, is_lab=False)

def fitness_func_lab(ga_instance, solution, solution_idx):
    individuals = decode_solution(solution, chromosomes_lab_df, True, lab_rooms)
    return -evaluate_fitness_generic(individuals, constraints_manager_lab, is_lab=True, lecture_population=lecture_population)

### --- MARK: Genetic Operators ---
def general_crossover(parents, offspring_size, class_count):
    offspring = []
    for k in range(offspring_size[0]):
        p1, p2 = parents[k % parents.shape[0]], parents[(k + 1) % parents.shape[0]]
        child = []
        for i in range(class_count):
            start = i * CLASS_BLOCK_SIZE
            block = p1[start:start+CLASS_BLOCK_SIZE] if np.random.rand() < 0.5 else p2[start:start+CLASS_BLOCK_SIZE]
            child.extend(block)
        offspring.append(child)
    return np.array(offspring)

def general_mutation(offspring, class_count, is_lab=False):
    for individual in offspring:
        i = np.random.randint(class_count)
        block_start = i * CLASS_BLOCK_SIZE
        choice = np.random.choice(["day", "session", "week"])
        if choice == "day":
            individual[block_start] = np.random.choice(VALID_DAYS) if not is_lab else np.random.choice(VALID_DAYS_LAB)
        elif choice == "session":
            individual[block_start + 1] = np.random.choice(VALID_SESSIONS)
        else:
            bit_idx = np.random.randint(16)
            individual[block_start + 2 + bit_idx] ^= 1
    return offspring

def custom_crossover(parents, offspring_size, ga_instance):
    return general_crossover(parents, offspring_size, NUM_CLASSES)

def custom_crossover_lab(parents, offspring_size, ga_instance):
    return general_crossover(parents, offspring_size, NUM_CLASSES_LAB)

def custom_mutation(offspring, ga_instance):
    return general_mutation(offspring, NUM_CLASSES)

def custom_mutation_lab(offspring, ga_instance):
    return general_mutation(offspring, NUM_CLASSES_LAB, True)

### --- MARK: Output Utilities ---
def decode_solution_to_dataframe(solution, class_ids, is_lab=False, room_list=None):
    decoded_rows = []

    for i, class_id in enumerate(class_ids):
        start = i * CLASS_BLOCK_SIZE
        chunk = solution[start:start + CLASS_BLOCK_SIZE]

        day = chunk[0]
        session = chunk[1]
        weeks = chunk[2:]

        if is_lab:
            course_id, lab_id, group_id = class_id.split('-')
        else:
            course_id, group_id = class_id.split('-')

        # Extract program info
        text, number = re.match(r'([a-zA-Z]+)(\d+)', group_id).groups()
        group_code = group_id.replace('CQ', 'L') if 'CQ' in group_id else group_id

        decoded = {
            'Mã môn học': course_id,
            'Tên môn học': COURSE_ID_TO_NAME.get(course_id, 'Unknown'),
            'Loại hình lớp': REVERSE_PROGRAM_ID.get(text, 'Unknown'),
            'Mã nhóm': group_code,
            'Thứ': day,
            'Tiết BD': session
        }

        if is_lab:
            decoded['Loại LAB'] = lab_id
            decoded['Phòng học'] = room_list[i] if room_list else 'Unknown'
        else:
            decoded['Loại phòng'] = ROOM_TYPE_ID.get(course_id, 'Unknown')

        # Add weeks
        for w_idx, week in enumerate(weeks):
            decoded[f'Tuần {w_idx + 1}'] = 'x' if week else None

        decoded_rows.append(decoded)

    return pd.DataFrame(decoded_rows)

def save_decoded_schedule(df_lecture=None, df_lab=None, output_path='output/result.xlsx'):
    with pd.ExcelWriter(output_path) as writer:
        if df_lecture is not None:
            df_lecture.to_excel(writer, sheet_name='LT', index=False)
        if df_lab is not None:
            df_lab.to_excel(writer, sheet_name='TH', index=False)
    print(f"Schedule saved to {output_path}")

def plot_best_fitness(ga_instance):
    best_fitness_per_gen = ga_instance.best_solutions_fitness

    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_per_gen, marker='o', linestyle='-', color='blue')
    plt.title('Best Fitness Value per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

### --- MARK: Main Execution Pipeline ---
def main():
    # 1. Load and preprocess input
    file_path = os.path.join('data', SHEET_FILE_NAME)
    df, df_khmt, df_ktmt = [read_excel(file_path, s) for s in SHEET_NAMES]
    df_main = preprocess(df, 0)
    df_khmt = preprocess(df_khmt, 1)
    df_ktmt = preprocess(df_ktmt, 2)

    # 2. Encode and setup constraints
    global chromosomes_df, NUM_CLASSES, NUM_CLASSES_LAB, constraints_manager, constraints_manager_lab, chromosomes_lab_df, lecture_population, df_course_lookup
    df_course_lookup = pd.concat([df_khmt, df_ktmt], ignore_index=True).drop_duplicates(subset="course_id")
    chromosomes_df, chromosomes_lab_df = encode(df_main)
    NUM_CLASSES = len(chromosomes_df)
    NUM_CLASSES_LAB = len(chromosomes_lab_df)

    constraints_manager = initialize_constraints(df_khmt, df_ktmt)
    constraints_manager_lab = initialize_constraints(df_khmt, df_ktmt, is_lab=True)

    # 3. Generate populations
    initial_population = create_initial_population(30, chromosomes_df, df_khmt, df_ktmt)
    initial_population_lab = create_initial_population_lab(30, chromosomes_lab_df, df_khmt, df_ktmt, lecture_population)

    # 4. Setup GA for lecture
    ga_instance = pygad.GA(
        num_generations=200,
        sol_per_pop=30,
        num_parents_mating=10,
        fitness_func=fitness_func,
        initial_population=initial_population,
        gene_type=int,
        crossover_type=custom_crossover,
        mutation_type=custom_mutation,
        mutation_probability=0.1,
        allow_duplicate_genes=True,
        stop_criteria="saturate_10"
    )

    # 5. Setup GA for lab
    ga_instance_lab = pygad.GA(
        num_generations=200,
        sol_per_pop=30,
        num_parents_mating=10,
        fitness_func=fitness_func_lab,
        initial_population=initial_population_lab,
        gene_type=int,
        crossover_type=custom_crossover_lab,
        mutation_type=custom_mutation_lab,
        mutation_probability=0.1,
        allow_duplicate_genes=True,
        stop_criteria="reach_0"
    )

    # 6. Run GA (Lecture)
    ga_instance.run()
    plot_best_fitness(ga_instance)
    best_solution, best_fitness, _ = ga_instance.best_solution()
    df_lecture = decode_solution_to_dataframe(best_solution, chromosomes_df, False)
    lecture_population = convert_to_individuals(best_solution, chromosomes_df)

    # 7. Run GA (Lab)
    ga_instance_lab.run()
    plot_best_fitness(ga_instance_lab)
    best_solution_lab, best_fitness_lab, _ = ga_instance_lab.best_solution()
    df_lab = decode_solution_to_dataframe(best_solution_lab, chromosomes_lab_df, True, lab_rooms)

    # Decode best individuals
    best_individuals_lecture = decode_solution(best_solution, chromosomes_df, is_lab=False)
    best_individuals_lab = decode_solution(best_solution_lab, chromosomes_lab_df, is_lab=True, room_list=lab_rooms)

    # Count violations
    lecture_violations = constraints_manager.count_violations(best_individuals_lecture)
    lab_violations = constraints_manager_lab.count_violations(best_individuals_lab, best_individuals_lecture)

    print("-" * 50)
    print("Best fitness (Lecture):", -best_fitness)
    print("Best fitness (Lab):", -best_fitness_lab)

    # 8. Save output
    save_decoded_schedule(df_lecture=df_lecture, df_lab=df_lab)

if __name__ == "__main__":
    main()