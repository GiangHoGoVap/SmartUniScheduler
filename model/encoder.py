from model.llm_parser import parse_with_llm
from config import (COLUMN_MAP, PROGRAM_ID, slot_of)

import pandas as pd
import math
import config

def read_excel(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name=sheet_name)

def clean_column(data, column_name):
    data[column_name] = data[column_name].str.replace(r'\s+', ' ', regex=True).str.strip()
    return data

def spec_to_slot_params(spec: dict, duration: int):
    s_idx = slot_of[(spec["day"], spec["session_start"])]
    e_idx = slot_of[(spec["day"], spec["session_end"])]
    if spec["fuzzy_type"] == "rectangle":
        return { "shape": "rect", "start_idx": s_idx, "end_idx": e_idx }

    best_start = spec.get("best_start", spec["session_start"])
    best_idx   = slot_of[(spec["day"], best_start)]
    plateau_end_idx = best_idx + duration - 1

    return {
        "shape": "trap",
        "a": max(s_idx - 1, 0),
        "b": best_idx,
        "c": plateau_end_idx,
        "d": min(e_idx + 1, len(slot_of) - 1)
    }

def preprocess(df, index):
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})

    if index == 0:
        df = df.dropna(subset=['Môn học TC/BB'])
        df = clean_column(df, 'program_id')
        config.COURSE_ID_TO_NAME = df.set_index('course_id')['course_name'].to_dict()
        config.ROOM_TYPE_ID = df.set_index('course_id')['max_students'].to_dict()

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

    elif index == 3:
        for row in df.itertuples():
            course_key = f"{row.course_id}-{row.group_id}"
            teacher = row.instructor_id
            phrase = row.description

            duration = course_duration(row.course_id)      
            spec = parse_with_llm(phrase, course_key, duration) 

            config.PREFS.setdefault(teacher, {})[course_key] = spec
            config.COURSE_TEACHER[course_key] = teacher
        
        for teacher, course_dict in config.PREFS.items():
            print(f"Course dict for teacher {teacher}: ", course_dict)
            config.PREFS_slot[teacher] = {
                ck: spec_to_slot_params(sp, course_duration(ck.split('-')[0]))
                for ck, sp in course_dict.items()
            }

    return df

def course_duration(course_id: str) -> int:
    row = config.df_course_lookup.loc[config.df_course_lookup.course_id == course_id]
    return int(row.num_sessions.iloc[0]) if not row.empty else 2

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

