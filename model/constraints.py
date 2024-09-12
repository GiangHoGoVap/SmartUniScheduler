VALID_DAY = range(2, 8)
VALID_SESSION = range(3, 13)

from model.utils import find_duplicates, count_overlap

class Constraint:
    def __init__(self, name):
        self.name = name
        self.violations = 0

    def evaluate(self, course_id, chromosome):
        """
        Evaluate the constraint on a given chromosome for a given course_id.
        Should return a penalty (for hard constraints) or a fitness score (for soft constraints).
        """
        return 0

class ValidDayConstraint(Constraint):
    def __init__(self):
        super().__init__('Valid day constraint')

    def evaluate(self, course_id, chromosome):
        # chromosome = '11010111011111111011111'
        penalty = 0
        day = int(chromosome[:3], 2)
        if day not in VALID_DAY:
            penalty += 100
            self.violations += 1
        return -penalty 

class SessionStartConstraint(Constraint):
    def __init__(self, course_info):
        super().__init__('Session start constraint')
        self.course_info = course_info

    def evaluate(self, course_id, chromosome):
        penalty = 0
        session_start = int(chromosome[3:7], 2)
        for index, row in self.course_info.iterrows():
            if row['course_id'] == course_id:
                if session_start + row['num_sessions'] - 1 not in VALID_SESSION:
                    penalty += 100
                    # print(f"Course {course_id}, chromosome {chromosome} violates session start constraint")
                    self.violations += 1
        return -penalty

class LunchBreakConstraint(Constraint):
    def __init__(self):
        super().__init__('Lunch break constraint')

    def evaluate(self, course_id, chromosome):
        penalty = 0
        session_start = int(chromosome[3:7], 2)
        if session_start == 6 or session_start == 7:
            penalty += 25
            self.violations += 1
        return -penalty
    
class MidtermBreakConstraint(Constraint):
    def __init__(self):
        super().__init__('Midterm break constraint')

    def evaluate(self, course_id, chromosome):
        penalty = 0
        weeks = list(chromosome[7:])
        if weeks[7] == '1':
            penalty += 25
            self.violations += 1
        return -penalty
    
class CourseDurationConstraint(Constraint):
    def __init__(self, course_info):
        super().__init__('Course duration constraint')
        self.course_info = course_info

    def evaluate(self, course_id, chromosome):
        penalty = 0
        weeks = list(chromosome[7:])
        total_duration = sum([int(week) for week in weeks])
        for index, row in self.course_info.iterrows():
            if row['course_id'] == course_id:
                if total_duration != row['num_weeks']:
                    penalty += 50 * (row['num_weeks'] - total_duration)
                    self.violations += 1
        return -penalty

class CourseSameSemesterConstraint(Constraint):
    def __init__(self, course_info):
        super().__init__('Course same semester constraint')
        self.course_info = course_info

    def evaluate(self, course_id, chromosome):
        pass

    def evaluate_population(self, population):
        penalty = 0
        
        for semester in range(1, 9):
            # semester = 1 --> course_list = ['CO1025', 'CO1023']
            course_list = self.course_info[self.course_info['semester'] == semester]['course_id'].tolist()

            # course_same_semester = {
            #   'L01': ['CO1025-L01-11010101100001110000011', 'CO1023-L01-11010101100001110000011'], 
            # }
            course_same_semester = {}
            for individual in population:
                course_id, group_id, bitstring = individual.split('-')[0], individual.split('-')[1], individual.split('-')[2]
                if course_id in course_list:
                    if group_id not in course_same_semester:
                        course_same_semester[group_id] = []
                    course_same_semester[group_id].append(individual)
            
            for group_id, individuals in course_same_semester.items():
                course_ids = [individual.split('-')[0] for individual in individuals]
                bitstrings = [individual.split('-')[2] for individual in individuals]
                days = [int(bitstring[:3], 2) for bitstring in bitstrings]
                session_starts = [int(bitstring[3:7], 2) for bitstring in bitstrings]
                weeks = [list(bitstring[7:]) for bitstring in bitstrings]

                day_duplicates = find_duplicates(days)
                if day_duplicates:
                    for day, indices in day_duplicates.items():
                        # Day '3' is duplicated at indices: [0, 2]
                        session_starts_from_day_duplicates = [session_starts[i] for i in indices] # [8, 9]
                        weeks_from_day_duplicates = [weeks[i] for i in indices]
                        course_ids_from_day_duplicates = [course_ids[i] for i in indices] # ['CO1025', 'CO1023']
                        
                        num_sessions_from_day_duplicates = []
                        for course_id in course_ids_from_day_duplicates:
                            num_sessions = self.course_info[self.course_info['course_id'] == course_id]['num_sessions'].values[0]
                            num_sessions_from_day_duplicates.append(num_sessions)
                        
                        num_overlap_cases = count_overlap(session_starts_from_day_duplicates, num_sessions_from_day_duplicates)
                        if num_overlap_cases > 0:       
                            for i in range(len(weeks_from_day_duplicates) - 1):
                                for j in range(i + 1, len(weeks_from_day_duplicates)):
                                    if weeks_from_day_duplicates[i] == weeks_from_day_duplicates[j]:
                                        # print(f'Course {course_ids_from_day_duplicates[i]}-{group_id} and {course_ids_from_day_duplicates[j]}-{group_id} have overlapping sessions')
                                        penalty += 25
                                        self.violations += 1
        
        return -penalty

class ConstraintsManager:
    def __init__(self):
        self.constraints = []

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def evaluate(self, course_id, chromosome):
        total_score = 0
        for constraint in self.constraints:
            score = constraint.evaluate(course_id, chromosome)
            if score is not None:
                total_score += score
        return total_score

    def evaluate_population(self, population):
        total_score = 0
        for constraint in self.constraints:
            if constraint.name == 'Course same semester constraint':
                score = constraint.evaluate_population(population)
                if score is not None:
                    total_score += score
        return total_score

    def reset_violations(self):
        for constraint in self.constraints:
            constraint.violations = 0

    def count_violations(self, population):
        # population = ['CO1007-CC01-01101011011111110111111', 'CO2003-CN01-11010111011111111011111']
        self.reset_violations()
        for individual in population:
            parts = individual.split('-')
            for constraint in self.constraints:
                if constraint.name == 'Course same semester constraint':
                    constraint.evaluate_population(population)
                else:
                    constraint.evaluate(parts[0], parts[2])
        return { constraint.name: constraint.violations for constraint in self.constraints }