from config import VALID_DAYS, VALID_SESSIONS, ROOMS
from model.utils import get_same_semester_courses, get_lecture_by_course_group, get_lectures_by_courses, create_non_conflicting_time
import numpy as np
import random
import config

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
        success = True
        for class_id in class_ids:
            course_id, lab_id, group_id = class_id.split('-')
            same_semester_courses = get_same_semester_courses(course_id, df1, df2)
            own_lecture = get_lecture_by_course_group(lecture_population, course_id, group_id)
            same_semester_lectures = get_lectures_by_courses(lecture_population, same_semester_courses)
            time_bits = create_non_conflicting_time(own_lecture, same_semester_lectures)
            if time_bits is None:
                success = False
                break
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
            config.lab_rooms.append(room_id)

            individual.extend(time_bits + last_part)
        if success:
            population.append(individual)
    return np.array(population)

def evaluate_fitness_generic(individuals, constraints_manager, is_lab=False, lecture_population=None):
    scores = [constraints_manager.evaluate(ind) for ind in individuals]
    soft_scores = [constraints_manager.evaluate_soft_individual(ind) for ind in individuals]
    
    if is_lab:
        pop_scores = constraints_manager.evaluate_population(individuals, is_lab=True, lecture_population=lecture_population)
    else:
        pop_scores = constraints_manager.evaluate_population(individuals)

    for i in range(len(scores)):
        if is_lab is False:
            if soft_scores[i] is None:    
                scores[i] = 0.5 * scores[i] + 0.5 * pop_scores[i]
            else:
                scores[i] = 0.4 * scores[i] + 0.4 * pop_scores[i] + 0.2 * soft_scores[i]
        else:
            scores[i] = 0.5 * scores[i] + 0.5 * pop_scores[i]
        scores[i] *= 1 / len(individuals)

    return sum(scores)