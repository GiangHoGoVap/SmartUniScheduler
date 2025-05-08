import os
import pandas as pd
import numpy as np
import re
import math

import sys
sys.path.append('model')
from GA import GeneticAlgorithm
from constraints import (ValidDayConstraint, 
                         SessionStartConstraint, 
                         LunchBreakConstraint, MidtermBreakConstraint, 
                         CourseDurationConstraint, 
                         CourseSameSemesterConstraint, CourseSameSemesterLabConstraint, 
                         LectureBeforeLabConstraint, LabSessionSpacingConstraint,
                         ConstraintsManager)

SHEET_FILE_NAME = 'Dự kiến SVMT_242 data.xlsx'
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
    global COURSE_ID_TO_NAME, ROOM_TYPE_ID

    # Rename Vietnamese columns to English equivalents
    replace_columns = {key: val for key, val in VI_TO_EN_COLUMN_NAMES.items() if key in data.columns}
    data = data.rename(columns=replace_columns)

    if index == 0:
        # Drop rows with missing values in the specific column
        data = data.dropna(subset=['Môn học TC/BB'])
        data = clean_column_spaces(data, 'program_id')

        # Generate global mappings
        COURSE_ID_TO_NAME = data.set_index('course_id')['course_name'].to_dict()
        ROOM_TYPE_ID = data.set_index('course_id')['max_students'].to_dict()

        # Filter and clean columns
        data = data[['course_id', 'program_id', 'num_students', 'num_groups', 'max_students', 'is_lab']]
        data['program_id'] = data['program_id'].map(PROGRAM_ID)
        data['num_students'] = data['num_students'].astype(int)
        data['num_groups'] = data['num_groups'].astype(int)
        data['max_students'] = data['max_students'].astype(int)
        data['is_lab'] = data['is_lab'].apply(lambda x: 1 if x == 'x' else 0)

    elif index == 1:
        # Select and process specific columns for schedule details
        data = data[['course_id', 'num_lectures', 'num_sessions', 'num_lab_lectures', 'num_lab_sessions', 'semester']]
        data = data.astype({
            'num_lectures': int,
            'num_sessions': int,
            'num_lab_lectures': int,
            'num_lab_sessions': int,
            'semester': int
        })
        # Compute the number of weeks for lectures and labs
        data['num_weeks'] = (data['num_lectures'] / data['num_sessions']).astype(int)
        data['num_weeks_lab'] = data.apply(
            lambda row: int(row['num_lab_lectures'] / row['num_lab_sessions']) if row['num_lab_sessions'] != 0 else 0,
            axis=1
        )

    return data

def encode(data):
    chromosomes = []  # For lecture sessions
    chromosomes_lab = []  # For lab sessions

    for row in data.itertuples():
        for group in range(1, row.num_groups + 1):
            # Encode the lecture session
            lecture_chromosome = f"{row.course_id}-{row.program_id}{str(group).zfill(2)}"
            chromosomes.append(lecture_chromosome)

            if row.is_lab:
                # Generate 2 labs per group
                for lab_num in range(1, math.ceil(row.max_students / 40) + 1):  
                    lab_chromosome = f"{row.course_id}-LAB{lab_num}-{row.program_id}{str(group).zfill(2)}"
                    chromosomes_lab.append(lab_chromosome)

    return chromosomes, chromosomes_lab


def decode_lecture(best_individuals):
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
            
            day = int(bitstring[:4], 2)
            session_start = int(bitstring[4:8], 2)
            available_weeks = list(bitstring[8:])
            
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

def decode_lab(best_individuals):
    # best_individuals = 
    # {
    #     "CO2013": {
    #         "CC01": {
    #             "LAB1": ["CO2013-LAB1-CC01-01010000010110101111100", 0.011363636363636364],
    #             "LAB2": ["CO2013-LAB2-CC01-01100001000110010010101", 0.011607142857142858]
    #         }
    #     }
    # }
    decoded_population = []
    
    for course_id, classes in best_individuals.items():
        for group_id, labs in classes.items():
            for lab_type, individual_data in labs.items():
                individual = individual_data[0]  # Extract encoded string
                score = individual_data[1]  # Extract fitness score
                
                parts = individual.split('-')
                if 'CQ' in parts[2]:
                    group_code = parts[2].replace('CQ', 'L')
                else:
                    group_code = parts[2]
                bitstring = parts[3]
                text, number = re.match(r'([a-zA-Z]+)(\d+)', parts[2]).groups()
                
                day = int(bitstring[:4], 2)
                session_start = int(bitstring[4:8], 2)
                available_weeks = list(bitstring[8:])
                
                decoded_individual = {
                    'Mã môn học': parts[0],
                    'Tên môn học': COURSE_ID_TO_NAME.get(parts[0], 'Unknown'),
                    'Loại hình lớp': REVERSE_PROGRAM_ID.get(text, 'Unknown'),
                    'Mã nhóm': group_code,
                    'Loại phòng': ROOM_TYPE_ID.get(parts[0], 'Unknown'),
                    'Thứ': day,
                    'Tiết BD': session_start,
                    'Điểm': score,
                    'Loại LAB': lab_type
                }
                
                for i, week in enumerate(available_weeks):
                    decoded_individual[f'Tuần {i+1}'] = 'x' if week == '1' else None
                
                decoded_population.append(decoded_individual)
    
    return decoded_population

def initialize_constraints(preprocessed_khmt, preprocessed_ktmt, is_lab=False):
    constraints_manager = ConstraintsManager()
    constraints_manager.add_constraint(MidtermBreakConstraint())

    if is_lab:
        constraints_manager.add_constraint(ValidDayConstraint(range(2, 9), True))
        constraints_manager.add_constraint(SessionStartConstraint(preprocessed_khmt, True))
        constraints_manager.add_constraint(SessionStartConstraint(preprocessed_ktmt, True))
        constraints_manager.add_constraint(CourseDurationConstraint(preprocessed_khmt, True))
        constraints_manager.add_constraint(CourseDurationConstraint(preprocessed_ktmt, True))
        constraints_manager.add_constraint(CourseSameSemesterLabConstraint(preprocessed_khmt))
        constraints_manager.add_constraint(CourseSameSemesterLabConstraint(preprocessed_ktmt))
        constraints_manager.add_constraint(LectureBeforeLabConstraint())
        constraints_manager.add_constraint(LabSessionSpacingConstraint())
    else:
        constraints_manager.add_constraint(ValidDayConstraint(range(2, 8)))
        constraints_manager.add_constraint(SessionStartConstraint(preprocessed_khmt))
        constraints_manager.add_constraint(SessionStartConstraint(preprocessed_ktmt))
        constraints_manager.add_constraint(LunchBreakConstraint())
        constraints_manager.add_constraint(CourseDurationConstraint(preprocessed_khmt))
        constraints_manager.add_constraint(CourseDurationConstraint(preprocessed_ktmt))
        constraints_manager.add_constraint(CourseSameSemesterConstraint(preprocessed_khmt))
        constraints_manager.add_constraint(CourseSameSemesterConstraint(preprocessed_ktmt))

    return constraints_manager


def run_genetic_algorithm(chromosomes, population_size, constraints_manager, chromosome_length, max_generations, preprocessed_df1, preprocessed_df2, lecture_population=None):
    ga = GeneticAlgorithm(population_size=population_size, 
                          crossover_rate=0.8, 
                          mutation_rate=0.1, 
                          elitism=5, 
                          constraints_manager=constraints_manager)
    return ga.run(chromosomes, chromosome_length, max_generations, preprocessed_df1, preprocessed_df2, lecture_population)


def save_results(decoded_lecture=None, decoded_lab=None, output_path=None):
    if output_path is not None:
        with pd.ExcelWriter(output_path) as writer:
            if decoded_lecture is not None:
                pd.DataFrame(decoded_lecture).to_excel(writer, sheet_name='LT', index=False)
            if decoded_lab is not None:
                pd.DataFrame(decoded_lab).to_excel(writer, sheet_name='TH', index=False)
        print(f"Excel file with two sheets has been saved as '{output_path}'")


def main():
    file_path = os.path.join('data', SHEET_FILE_NAME)
    
    # Read and preprocess data
    df = read_excel_file(file_path, SHEET_NAMES[0])
    df_khmt = read_excel_file(file_path, SHEET_NAMES[1])
    df_ktmt = read_excel_file(file_path, SHEET_NAMES[2])
    
    preprocessed_df = preprocess_data(df, 0)
    preprocessed_df_khmt = preprocess_data(df_khmt, 1)  
    preprocessed_df_ktmt = preprocess_data(df_ktmt, 1)

    chromosomes_df, chromosomes_lab_df = encode(preprocessed_df)
    lecture_population_size = len(chromosomes_df)
    lab_population_size = len(chromosomes_lab_df)
    
    # Initialize constraints
    lecture_constraints_manager = initialize_constraints(preprocessed_df_khmt, preprocessed_df_ktmt, is_lab=False)
    lab_constraints_manager = initialize_constraints(preprocessed_df_khmt, preprocessed_df_ktmt, is_lab=True)

    # Run GeneticAlgorithm for lectures
    chromosome_length = 24  # 4 bits for day, 4 bits for session_start, 16 bits for weeks
    max_generations = 500

    population_lecture = run_genetic_algorithm(chromosomes_df, 
                                               lecture_population_size, 
                                               lecture_constraints_manager, 
                                               chromosome_length, 
                                               max_generations, 
                                               preprocessed_df_khmt, 
                                               preprocessed_df_ktmt)
    decoded_population_lecture = decode_lecture(population_lecture)

    population_lst = [value[0] for sub_dict in population_lecture.values() for value in sub_dict.values()]
    
    # Run GeneticAlgorithm for labs
    population_lab = run_genetic_algorithm(chromosomes_lab_df, 
                                           lab_population_size, 
                                           lab_constraints_manager, 
                                           chromosome_length, 
                                           max_generations, 
                                           preprocessed_df_khmt, 
                                           preprocessed_df_ktmt, 
                                           population_lst)
    decoded_population_lab = decode_lab(population_lab)

    # Save results
    save_results(decoded_population_lecture, decoded_population_lab, 'output/result.xlsx')

if __name__ == "__main__":
    main()