import os
import pandas as pd
import numpy as np
import re

import sys
sys.path.append('model')
from GA import GeneticAlgorithm
from constraints import ValidDayConstraint, SessionStartConstraint, LunchBreakConstraint, MidtermBreakConstraint, CourseDurationConstraint, CourseSameSemesterConstraint, ConstraintsManager

SHEET_FILE_NAME = 'Dự kiến SVMT_241 data.xlsx'
SHEET_NAMES = ['Thống kê', 'KHMT', 'KTMT'] 

PROGRAM_ID = {
    "Chương trình giảng dạy bằng tiếng Anh": "CC",
    "Chương trình tiêu chuẩn": "CQ",
    "Chương trình định hướng Nhật Bản": "CN"
}
REVERSE_PROGRAM_ID = {v: k for k, v in PROGRAM_ID.items()}

VI_TO_EN_COLUMN_NAMES = {
    "Mã môn học": "course_id",
    "Tên môn học": "course_name",
    "Loại hình lớp": "program_id",
    "Số lượng nhóm": "num_groups",
    "Sỉ số SV max": "max_students",
    "Tổng LT": "num_lectures",
    "Số tiết": "num_sessions",
    "Học kỳ": "semester"
}

ROOM_TYPE_ID = {}
COURSE_ID_TO_NAME = {}

def read_excel_file(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df

def clean_column_spaces(data, column_name):
    # Use regex to replace multiple spaces with a single space and strip leading/trailing spaces
    data.loc[:, column_name] = data[column_name].str.replace(r'\s+', ' ', regex=True).str.strip()
    return data

def preprocess_data(data, index):
    if index == 0:
        global COURSE_ID_TO_NAME, ROOM_TYPE_ID

        data = data.dropna(subset=['Môn học TC/BB'])
        data = clean_column_spaces(data, 'Loại hình lớp')

        COURSE_ID_TO_NAME = data.set_index('Mã môn học')['Tên môn học'].to_dict()
        ROOM_TYPE_ID = data.set_index('Mã môn học')['Sỉ số SV max'].to_dict()

        replace_columns = { key: val for key, val in VI_TO_EN_COLUMN_NAMES.items() if key in data.columns }
        data = data.rename(columns=replace_columns)
        data = data[['course_id', 'program_id', 'num_groups']]
        data['program_id'] = data['program_id'].map(PROGRAM_ID)
        data['num_groups'] = data['num_groups'].astype(int)
    elif index == 1:
        replace_columns = { key: val for key, val in VI_TO_EN_COLUMN_NAMES.items() if key in data.columns }
        data = data.rename(columns=replace_columns)
        data = data[['course_id', 'num_lectures', 'num_sessions', 'semester']]
        data['num_lectures'] = data['num_lectures'].astype(int)
        data['num_sessions'] = data['num_sessions'].astype(int)
        data['semester'] = data['semester'].astype(int)
        data['num_weeks'] = (data['num_lectures'] / data['num_sessions']).astype(int)
    return data

def encode(data):
    chromosomes = []
    for row in data.itertuples():
        for group in range(1, row.num_groups + 1):
            chromosome = row.course_id + "-" + row.program_id + str(group).zfill(2)
            chromosomes.append(chromosome)
    
    return chromosomes

def decode(best_individuals):
    # best_individuals = {
    #   'CO2003': {
    #       'L01': ['CO2003-L01-11010101100001110000011', 10], 
    #       'CN02': ['CO2003-CN02-11010101100001110000011', 20]
    #   }
    # }
    decoded_population = []

    for course_id, classes in best_individuals.items():
        for group_id, list_obj in classes.items():
            individual = list_obj[0]
            score = list_obj[1]

            parts = individual.split('-')
            if 'CQ' in parts[1]:
                group_code = parts[1].replace('CQ', 'L')
            else:
                group_code = parts[1]
            bitstring = parts[2]

            text, number = re.match(r'([a-zA-Z]+)(\d+)', parts[1]).groups() # text = 'CC', number = '01'
            day = int(bitstring[:3], 2)
            session_start = int(bitstring[3:7], 2)
            available_weeks = list(bitstring[7:])
            
            decoded_individual = {
                'Mã môn học': parts[0],
                'Tên môn học': COURSE_ID_TO_NAME[parts[0]],
                'Loại hình lớp': REVERSE_PROGRAM_ID[text],
                'Mã nhóm': group_code,
                'Thứ': day,
                'Tiết BD': session_start,
                'Loại phòng': ROOM_TYPE_ID[parts[0]],
                'Điểm': score
            }

            for i, week in enumerate(available_weeks):
                decoded_individual[f'Tuần {i+1}'] = 'x' if week == '1' else None
            
            decoded_population.append(decoded_individual)

    return decoded_population

def main():
    file_path = os.path.join('data', SHEET_FILE_NAME)
    
    df0 = read_excel_file(file_path, SHEET_NAMES[0])
    df1 = read_excel_file(file_path, SHEET_NAMES[1])
    df2 = read_excel_file(file_path, SHEET_NAMES[2])
    
    preprocessed_df0 = preprocess_data(df0, 0)
    preprocessed_df1 = preprocess_data(df1, 1)  
    preprocessed_df2 = preprocess_data(df2, 1)
    
    chromosomes_df0 = encode(preprocessed_df0)
    population_size = len(chromosomes_df0)

    # Initialize constraints
    valid_day_constraint = ValidDayConstraint()
    session_start_constraint_khmt = SessionStartConstraint(preprocessed_df1)
    session_start_constraint_ktmt = SessionStartConstraint(preprocessed_df2)
    lunch_break_constraint = LunchBreakConstraint()
    midterm_break_constraint = MidtermBreakConstraint()
    course_duration_constraint_khmt = CourseDurationConstraint(preprocessed_df1)
    course_duration_constraint_ktmt = CourseDurationConstraint(preprocessed_df2)
    course_same_semester_constraint_khmt = CourseSameSemesterConstraint(preprocessed_df1)
    course_same_semester_constraint_ktmt = CourseSameSemesterConstraint(preprocessed_df2)

    # Initialize the Constraints Manager and add constraints
    constraints_manager = ConstraintsManager()
    constraints_manager.add_constraint(valid_day_constraint)
    constraints_manager.add_constraint(session_start_constraint_khmt)
    constraints_manager.add_constraint(session_start_constraint_ktmt)
    constraints_manager.add_constraint(lunch_break_constraint)
    constraints_manager.add_constraint(midterm_break_constraint)
    constraints_manager.add_constraint(course_duration_constraint_khmt)
    constraints_manager.add_constraint(course_duration_constraint_ktmt)
    constraints_manager.add_constraint(course_same_semester_constraint_khmt)
    constraints_manager.add_constraint(course_same_semester_constraint_ktmt)
    
    ga = GeneticAlgorithm(population_size=population_size, crossover_rate=0.8, mutation_rate=0.1, elitism=5, constraints_manager=constraints_manager)
    chromosome_length = 23 # 3 bits for day, 4 bits for session_start, 16 bits for weeks
    max_generations = 500
    population = ga.run(chromosomes_df0, chromosome_length, max_generations, preprocessed_df1, preprocessed_df2)
    decoded_population = decode(population)

    result = pd.DataFrame(decoded_population)
    result.to_excel('output/result.xlsx', index=False)
    
if __name__ == "__main__":
    main()