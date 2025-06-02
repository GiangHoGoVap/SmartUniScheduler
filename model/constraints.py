VALID_DAY = range(2, 8)
VALID_SESSION = range(3, 13)

from model.utils import find_duplicates, count_overlap, weeks_overlap
from model.individual import Individual

class Constraint:
    def __init__(self, name):
        self.name = name
        self.violations = 0

    def evaluate(self, individual: Individual):
        return 0

class ValidDayConstraint(Constraint):
    def __init__(self, valid_days, is_lab_constraint=False):
        super().__init__('Valid day constraint' if not is_lab_constraint else 'Valid day lab constraint')
        self.valid_days = valid_days

    def evaluate(self, individual: Individual):
        # chromosome = '11010111011111111011111'
        self.violations = 0
        day = int(individual.bitstring[:4], 2)
        if day not in self.valid_days:
            self.violations = 1
            individual.add_violation(ValidDayConstraint.__name__, [f"{individual.course_id}-{individual.group_id}"])
        return self.violations 

class SessionStartConstraint(Constraint):
    def __init__(self, course_info, is_lab_constraint=False):
        super().__init__('Session start constraint' if not is_lab_constraint else 'Session start lab constraint')
        self.course_info = course_info
        self.is_lab_constraint = is_lab_constraint

    def evaluate(self, individual: Individual):
        # Extract session start directly from chromosome
        self.violations = 0
        session_start = int(individual.bitstring[4:8], 2)
        
        # Check if it's a lab constraint
        if self.is_lab_constraint:
            # Lab constraints: session start must be 2 or 8
            if session_start not in {2, 8}:
                self.violations = 1
                individual.add_violation(SessionStartConstraint.__name__, [f"{individual.course_id}-{individual.group_id}"])
        else:
            for index, row in self.course_info.iterrows():
                if row['course_id'] == individual.course_id:
                    # Check if session start fits within valid session range
                    if session_start + row['num_sessions'] - 1 not in VALID_SESSION:
                        self.violations = 1
                        individual.add_violation(SessionStartConstraint.__name__, [f"{individual.course_id}-{individual.group_id}"])
        return self.violations

class LunchBreakConstraint(Constraint):
    def __init__(self):
        super().__init__('Lunch break constraint')

    def evaluate(self, individual: Individual):
        self.violations = 0
        session_start = int(individual.bitstring[4:8], 2)
        if session_start == 6:
            self.violations = 1
            individual.add_violation(LunchBreakConstraint.__name__, [f"{individual.course_id}-{individual.group_id}"])
        return self.violations
    
class MidtermBreakConstraint(Constraint):
    def __init__(self):
        super().__init__('Midterm break constraint')

    def evaluate(self, individual: Individual):
        self.violations = 0
        weeks = list(individual.bitstring[8:])
        if weeks[7] == '1':
            self.violations = 1
            individual.add_violation(MidtermBreakConstraint.__name__, [f"{individual.course_id}-{individual.group_id}"])
        return self.violations
    
class CourseDurationConstraint(Constraint):
    def __init__(self, course_info, is_lab_constraint=False):
        super().__init__('Course duration constraint' if not is_lab_constraint else 'Course duration lab constraint')
        self.course_info = course_info
        self.is_lab_constraint = is_lab_constraint

    def evaluate(self, individual: Individual):
        # Extract course duration from chromosome
        self.violations = 0
        weeks = list(individual.bitstring[8:])
        total_duration = sum(int(week) for week in weeks)

        for index, row in self.course_info.iterrows():
            if row['course_id'] == individual.course_id:
                if self.is_lab_constraint:
                    expected_duration = row['num_weeks_lab']
                else:
                    expected_duration = row['num_weeks']
                # self.violations += abs(total_duration - expected_duration) ** 2
                if total_duration != expected_duration:
                    self.violations = 1
                    individual.add_violation(CourseDurationConstraint.__name__, [f"{individual.course_id}-{individual.group_id}"])
        return self.violations

class CourseSameSemesterConstraint(Constraint):
    def __init__(self, course_info):
        super().__init__('CourseSameSemesterConstraint')
        self.course_info = course_info
        self.semester_courses = {semester: set(course_info[course_info['semester'] == semester]['course_id'].tolist())
                                 for semester in range(1, 9)}

    def evaluate_population(self, population):
        self.violations = 0

        for semester in range(1, 9):
            course_set = self.semester_courses[semester]
            course_same_semester = {}

            for ind in population:
                if ind.course_id in course_set:
                    course_same_semester.setdefault(ind.group_id, []).append(ind)

            for group_id, individuals in course_same_semester.items():
                # Group by day
                day_dict = {}
                for ind in individuals:
                    day = int(ind.bitstring[:4], 2)
                    day_dict.setdefault(day, []).append(ind)

                for day, same_day_inds in day_dict.items():
                    # Extract with timing and sort
                    slots = []
                    for ind in same_day_inds:
                        start = int(ind.bitstring[4:8], 2)
                        weeks = tuple(ind.bitstring[8:])
                        duration = self.course_info[self.course_info['course_id'] == ind.course_id]['num_sessions'].values[0]
                        end = start + duration
                        slots.append((start, end, ind, weeks))

                    # Sort by start time
                    slots.sort(key=lambda x: x[0])

                    # Sweep for overlap
                    for i in range(len(slots) - 1):
                        _, end_i, ind_i, weeks_i = slots[i]
                        for j in range(i + 1, len(slots)):
                            start_j, end_j, ind_j, weeks_j = slots[j]
                            if start_j >= end_i:
                                break  # No overlap, can skip due to sorting
                            if weeks_i == weeks_j:
                                # Violation
                                ind_i.add_violation(CourseSameSemesterConstraint.__name__, [ind_j.course_id])
                                ind_j.add_violation(CourseSameSemesterConstraint.__name__, [ind_i.course_id])
                                self.violations += 1

        return self.violations

class CourseSameSemesterLabConstraint(Constraint):
    def __init__(self, course_info):
        super().__init__('Course same semester lab constraint')
        self.course_info = course_info
        # Precompute course lists for each semester
        self.semester_courses = {}
        for semester in range(1, 9):
            self.semester_courses[semester] = set(
                self.course_info[self.course_info['semester'] == semester]['course_id'].tolist()
            )

    def evaluate_population(self, lab_population, lecture_population):
        self.violations = 0  # Reset violations count

        for semester in range(1, 9):
            course_set = self.semester_courses[semester]

            # Group labs and lectures by group_id
            lab_same_semester = {}
            lecture_same_semester = {}

            for lab in lab_population:
                lab_course_id, lab_group_id = lab.course_id, lab.group_id.split('-')[1]
                if lab_course_id in course_set:
                    if lab_group_id not in lab_same_semester:
                        lab_same_semester[lab_group_id] = []
                    lab_same_semester[lab_group_id].append(lab)

            for lecture in lecture_population:
                lecture_course_id, lecture_group_id = lecture.course_id, lecture.group_id
                if lecture_course_id in course_set:
                    if lecture_group_id not in lecture_same_semester:
                        lecture_same_semester[lecture_group_id] = []
                    lecture_same_semester[lecture_group_id].append(lecture)

            # Check violations for labs
            for group_id, labs in lab_same_semester.items():
                lab_days = [int(ind.bitstring[:4], 2) for ind in labs]
                lab_session_starts = [int(ind.bitstring[4:8], 2) for ind in labs]
                lab_weeks = [tuple(ind.bitstring[8:]) for ind in labs]
                lab_course_ids = [ind.course_id for ind in labs]
                lab_room_ids = [ind.room for ind in labs]

                # Detect duplicate days
                day_indices = {}
                for idx, day in enumerate(lab_days):
                    day_indices.setdefault(day, []).append(idx)

                # Process duplicate days
                for day, indices in day_indices.items():
                    if len(indices) > 1:
                        lab_sessions_from_day_duplicates = [lab_session_starts[i] for i in indices]
                        lab_weeks_from_day_duplicates = [lab_weeks[i] for i in indices]
                        lab_course_ids_from_day_duplicates = [lab_course_ids[i] for i in indices]
                        # labs_from_day_duplicates = [labs[i] for i in indices]
                        lab_room_ids_from_day_duplicates = [lab_room_ids[i] for i in indices]

                        num_sessions_from_day_duplicates = [self.course_info[self.course_info['course_id'] == course_id]['num_lab_sessions'].values[0]
                                                            for course_id in lab_course_ids_from_day_duplicates]

                        num_lab_overlap_cases = count_overlap(lab_sessions_from_day_duplicates, num_sessions_from_day_duplicates)
                        if num_lab_overlap_cases > 0:
                            for i in range(len(lab_weeks_from_day_duplicates)):
                                for j in range(i + 1, len(lab_weeks_from_day_duplicates)):
                                    if lab_room_ids_from_day_duplicates[i] == lab_room_ids_from_day_duplicates[j] and weeks_overlap(lab_weeks_from_day_duplicates[i], lab_weeks_from_day_duplicates[j]):
                                        # print(f"Lab {lab_course_ids_from_day_duplicates[i]} overlaps with lab {lab_course_ids_from_day_duplicates[j]} on day {day} in weeks {lab_weeks_from_day_duplicates[i]} and {lab_weeks_from_day_duplicates[j]} at room {lab_room_ids_from_day_duplicates[i]}.")
                                        self.violations += 1

            # Check overlaps between labs and lectures
            for group_id, labs in lab_same_semester.items():
                for lab in labs:
                    lab_course_id, lab_bitstring = lab.course_id, lab.bitstring
                    lab_day = int(lab_bitstring[:4], 2)
                    lab_session_start = int(lab_bitstring[4:8], 2)
                    lab_weeks = tuple(lab_bitstring[8:])

                    if group_id in lecture_same_semester:
                        for lecture in lecture_same_semester[group_id]:
                            lecture_course_id, lecture_bitstring = lecture.course_id, lecture.bitstring
                            lecture_day = int(lecture_bitstring[:4], 2)
                            lecture_session_start = int(lecture_bitstring[4:8], 2)
                            lecture_weeks = tuple(lecture_bitstring[8:])

                            if lab_day == lecture_day:
                                num_sessions = self.course_info[self.course_info['course_id'] == lecture_course_id]['num_sessions'].values[0]
                                num_lab_sessions = self.course_info[self.course_info['course_id'] == lab_course_id]['num_lab_sessions'].values[0]
                                if count_overlap([lab_session_start, lecture_session_start], [num_lab_sessions, num_sessions]) > 0:
                                    self.violations += 1

        return self.violations

class LectureBeforeLabConstraint(Constraint):
    def __init__(self, min_weeks_after=3, penalty_weight=1):
        super().__init__('Lecture-before-laboratory dependency constraint')
        self.min_weeks_after = min_weeks_after  # Minimum number of weeks the lab must start after the lecture
        self.penalty_weight = penalty_weight  # Scaling factor for the penalty

    def evaluate_population(self, lab_population, lecture_population):
        # Initialize a list to store penalties for each lab in the population
        penalties = [0] * len(lab_population)

        # Precompute lecture start weeks for faster lookup
        lecture_start_weeks = {}
        for lecture in lecture_population:
            course_id, group_id, bitstring = lecture.course_id, lecture.group_id, lecture.bitstring
            weeks = list(bitstring[8:])  # Extract weeks from the bitstring
            start_week = weeks.index('1') if '1' in weeks else -1  # Find the first week of the lecture
            lecture_start_weeks[f"{course_id}-{group_id}"] = start_week

        # Check labs and assign penalties
        for i, lab in enumerate(lab_population):
            # Extract lab information
            lab_course_id, lab_group_id, lab_bitstring = lab.course_id, lab.group_id.split('-')[1], lab.bitstring
            lab_weeks = list(lab_bitstring[8:])  # Extract weeks from the bitstring
            lab_start_week = lab_weeks.index('1') if '1' in lab_weeks else -1  # Find the first week of the lab

            # Find the corresponding lecture start week
            lecture_key = f"{lab_course_id}-{lab_group_id}"
            lecture_start_week = lecture_start_weeks.get(lecture_key, -1)

            # Check if the lab starts at least `min_weeks_after` weeks after the lecture
            if lecture_start_week != -1 and lab_start_week != -1:
                actual_gap = lab_start_week - lecture_start_week
                violation_amount = max(0, self.min_weeks_after - actual_gap)
                penalties[i] = violation_amount * self.penalty_weight  # Dynamic penalty
            else:
                # Handle invalid cases (e.g., no '1' in weeks or no corresponding lecture)
                penalties[i] = self.penalty_weight  # Assign a fixed penalty for invalid cases

        return penalties

class LabSessionSpacingConstraint(Constraint):
    def __init__(self):
        super().__init__('Lab session spacing constraint')
        
    def evaluate(self, individual: Individual):
        self.violations = 0

        weeks = list(individual.bitstring[8:])
        scheduled_weeks = [idx for idx, week in enumerate(weeks) if week == '1']

        # Check for consecutive weeks
        for j in range(len(scheduled_weeks) - 1):
            if scheduled_weeks[j + 1] == scheduled_weeks[j] + 1:
                self.violations += 1

        return self.violations

class ConstraintsManager:
    def __init__(self):
        self.constraints = []

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def evaluate(self, individual: Individual):
        total_violations = 0
        for constraint in self.constraints:
            violations = constraint.evaluate(individual)
            if violations is not None:
                total_violations += violations
        return 1 - 1 / (1 + total_violations)

    def evaluate_population(self, population, is_lab=False, lecture_population=None):
        total_penalties = [0] * len(population)  # Initialize a list to store penalties for each individual
        
        for constraint in self.constraints:
            # Handle LectureBeforeLabConstraint separately
            if isinstance(constraint, LectureBeforeLabConstraint):
                if lecture_population is None:
                    raise ValueError("Lecture population is required for LectureBeforeLabConstraint.")
                penalties = constraint.evaluate_population(population, lecture_population)
                if penalties is not None:
                    for i, penalty in enumerate(penalties):
                        total_penalties[i] += penalty
            else:
                # Handle other constraints
                if is_lab and constraint.name == 'Course same semester lab constraint':
                    violations = constraint.evaluate_population(population, lecture_population)
                    if violations is not None:
                        for i in range(len(population)):
                            total_penalties[i] += violations  
                elif not is_lab and constraint.name == 'Course same semester constraint':
                    violations = constraint.evaluate_population(population)
                    if violations is not None:
                        for i in range(len(population)):
                            total_penalties[i] += violations  
                else:
                    continue  # Skip irrelevant constraints
        
        # Calculate the overall fitness score based on penalties
        fitness_scores = [1 - 1 / (1 + penalty**2) for penalty in total_penalties]
        return fitness_scores

    def reset_violations(self):
        for constraint in self.constraints:
            constraint.violations = 0

    def count_violations(self, population, lecture_population=None):
        violations_dict = {}

        print("-" * 50)

        # Handle population constraints
        for idx, constraint in enumerate(self.constraints):
            name = f"{constraint.name}_{idx}"
            violations_dict[name] = 0  # Initialize

            if isinstance(constraint, LectureBeforeLabConstraint):
                if lecture_population is None:
                    raise ValueError("Lecture population is required for LectureBeforeLabConstraint.")
                penalties = constraint.evaluate_population(population, lecture_population)
                if penalties is not None: 
                    violations_dict[name] = sum(penalties)
            elif constraint.name in {'Course same semester constraint', 'Course same semester lab constraint'}:
                is_lab = constraint.name == 'Course same semester lab constraint'
                if is_lab:
                    violations = constraint.evaluate_population(population, lecture_population)
                else:
                    violations = constraint.evaluate_population(population)
                if violations is not None:
                    violations_dict[name] = violations

        # Handle individual constraints
        for idx, constraint in enumerate(self.constraints):
            name = f"{constraint.name}_{idx}"
            if constraint.name not in {'Course same semester constraint', 'Course same semester lab constraint'}:
                for individual in population:
                    course_id = individual.course_id
                    group_id = individual.group_id if lecture_population is None else individual.group_id.split('-')[1]
                    session_info = individual.bitstring

                    violations = constraint.evaluate(individual)
                    if violations:
                        print(f"Course {course_id} violates {constraint.name} with {violations} violations.")
                        violations_dict[name] += violations

        for name, count in violations_dict.items():
            print("Constraint:", name, " - Violations:", count)

        # Return as list of tuples, if needed
        return list(violations_dict.items())
