VALID_DAY = range(2, 8)
VALID_SESSION = range(3, 13)

from model.utils import find_duplicates, count_overlap

class Constraint:
    def __init__(self, name):
        self.name = name
        self.violations = 0

    def evaluate(self, course_id, group_id, chromosome):
        """
        Evaluate the constraint on a given chromosome for a given course_id.
        Should return number of violations (for hard constraints) or a fitness score (for soft constraints).
        """
        return 0

class ValidDayConstraint(Constraint):
    def __init__(self, valid_days, is_lab_constraint=False):
        super().__init__('Valid day constraint' if not is_lab_constraint else 'Valid day lab constraint')
        self.valid_days = valid_days

    def evaluate(self, course_id, group_id, chromosome):
        # chromosome = '11010111011111111011111'
        self.violations = 0
        day = int(chromosome[:4], 2)
        if day not in self.valid_days:
            self.violations = 1
        return self.violations 

class SessionStartConstraint(Constraint):
    def __init__(self, course_info, is_lab_constraint=False):
        super().__init__('Session start constraint' if not is_lab_constraint else 'Session start lab constraint')
        self.course_info = course_info
        self.is_lab_constraint = is_lab_constraint

    def evaluate(self, course_id, group_id, chromosome):
        # Extract session start directly from chromosome
        self.violations = 0
        session_start = int(chromosome[4:8], 2)
        
        # Check if it's a lab constraint
        if self.is_lab_constraint:
            # Lab constraints: session start must be 2 or 8
            if session_start not in {2, 8}:
                self.violations = 1
        else:
            for index, row in self.course_info.iterrows():
                if row['course_id'] == course_id:
                    # Check if session start fits within valid session range
                    if session_start + row['num_sessions'] - 1 not in VALID_SESSION:
                        self.violations = 1
        
        return self.violations

class LunchBreakConstraint(Constraint):
    def __init__(self):
        super().__init__('Lunch break constraint')

    def evaluate(self, course_id, group_id, chromosome):
        self.violations = 0
        session_start = int(chromosome[4:8], 2)
        if session_start == 6:
            print(f"Course {course_id}-{group_id} violates lunch break constraint with session start at 6.")
            self.violations = 1
        return self.violations
    
class MidtermBreakConstraint(Constraint):
    def __init__(self):
        super().__init__('Midterm break constraint')

    def evaluate(self, course_id, group_id, chromosome):
        self.violations = 0
        weeks = list(chromosome[8:])
        if weeks[7] == '1':
            self.violations = 1
        return self.violations
    
class CourseDurationConstraint(Constraint):
    def __init__(self, course_info, is_lab_constraint=False):
        super().__init__('Course duration constraint' if not is_lab_constraint else 'Course duration lab constraint')
        self.course_info = course_info
        self.is_lab_constraint = is_lab_constraint

    def evaluate(self, course_id, group_id, chromosome):
        # Extract course duration from chromosome
        self.violations = 0
        weeks = list(chromosome[8:])
        total_duration = sum(int(week) for week in weeks)

        for index, row in self.course_info.iterrows():
            if row['course_id'] == course_id:
                if self.is_lab_constraint:
                    expected_duration = row['num_weeks_lab']
                else:
                    expected_duration = row['num_weeks']
                # self.violations += abs(total_duration - expected_duration) ** 2
                if total_duration != expected_duration:
                    print(f"Course {course_id} has total duration {total_duration} but expected {expected_duration}")
                    self.violations = 1
        
        return self.violations

class CourseSameSemesterConstraint(Constraint):
    def __init__(self, course_info):
        super().__init__('Course same semester constraint')
        self.course_info = course_info
        # Precompute course lists for each semester
        self.semester_courses = {}
        for semester in range(1, 9):
            self.semester_courses[semester] = set(
                self.course_info[self.course_info['semester'] == semester]['course_id'].tolist()
            )

    def evaluate_population(self, population):
        self.violations = 0

        for semester in range(1, 9):
            course_set = self.semester_courses[semester]
            course_same_semester = {}
            for individual in population:
                parts = individual.split('-')
                course_id, group_id, bitstring = parts[0], parts[1], parts[2]
                if course_id in course_set:
                    if group_id not in course_same_semester:
                        course_same_semester[group_id] = []
                    course_same_semester[group_id].append((course_id, bitstring))

            for group_id, individuals in course_same_semester.items():
                course_ids = [ind[0] for ind in individuals]
                bitstrings = [ind[1] for ind in individuals]
                days = [int(bitstring[:4], 2) for bitstring in bitstrings]
                session_starts = [int(bitstring[4:8], 2) for bitstring in bitstrings]
                weeks = [tuple(bitstring[8:]) for bitstring in bitstrings]  # Convert to tuple for hashability

                # Detect duplicate days
                day_indices = {}
                for idx, day in enumerate(days):
                    if day not in day_indices:
                        day_indices[day] = []
                    day_indices[day].append(idx)

                # Process duplicate days
                for day, indices in day_indices.items():
                    if len(indices) > 1:
                        session_starts_from_day_duplicates = [session_starts[i] for i in indices]
                        weeks_from_day_duplicates = [weeks[i] for i in indices]
                        course_ids_from_day_duplicates = [course_ids[i] for i in indices]

                        num_sessions_from_day_duplicates = [
                            self.course_info[self.course_info['course_id'] == course_id]['num_sessions'].values[0]
                            for course_id in course_ids_from_day_duplicates
                        ]

                        num_overlap_cases = count_overlap(session_starts_from_day_duplicates, num_sessions_from_day_duplicates)
                        if num_overlap_cases > 0:
                            week_set = set()
                            for week in weeks_from_day_duplicates:
                                if week in week_set:
                                    self.violations += 1
                                else:
                                    week_set.add(week)

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
                parts = lab.split('-')
                lab_course_id, lab_group_id, lab_bitstring = parts[0], parts[2], parts[3]
                if lab_course_id in course_set:
                    if lab_group_id not in lab_same_semester:
                        lab_same_semester[lab_group_id] = []
                    lab_same_semester[lab_group_id].append((lab_course_id, lab_bitstring))

            for lecture in lecture_population:
                parts = lecture.split('-')
                lecture_course_id, lecture_group_id, lecture_bitstring = parts[0], parts[1], parts[2]
                if lecture_course_id in course_set:
                    if lecture_group_id not in lecture_same_semester:
                        lecture_same_semester[lecture_group_id] = []
                    lecture_same_semester[lecture_group_id].append((lecture_course_id, lecture_bitstring))

            # Check violations for labs
            for group_id, labs in lab_same_semester.items():
                lab_course_ids = [lab[0] for lab in labs]
                lab_bitstrings = [lab[1] for lab in labs]
                lab_days = [int(bitstring[:4], 2) for bitstring in lab_bitstrings]
                lab_session_starts = [int(bitstring[4:8], 2) for bitstring in lab_bitstrings]
                lab_weeks = [tuple(bitstring[8:]) for bitstring in lab_bitstrings]

                # Detect duplicate days
                day_indices = {}
                for idx, day in enumerate(lab_days):
                    if day not in day_indices:
                        day_indices[day] = []
                    day_indices[day].append(idx)

                # Process duplicate days
                for day, indices in day_indices.items():
                    if len(indices) > 1:
                        lab_sessions_from_day_duplicates = [lab_session_starts[i] for i in indices]
                        lab_weeks_from_day_duplicates = [lab_weeks[i] for i in indices]
                        lab_course_ids_from_day_duplicates = [lab_course_ids[i] for i in indices]

                        num_sessions_from_day_duplicates = [
                            self.course_info[self.course_info['course_id'] == course_id]['num_lab_sessions'].values[0]
                            for course_id in lab_course_ids_from_day_duplicates
                        ]

                        num_lab_overlap_cases = count_overlap(lab_sessions_from_day_duplicates, num_sessions_from_day_duplicates)
                        if num_lab_overlap_cases > 0:
                            week_set = set()
                            for week in lab_weeks_from_day_duplicates:
                                if week in week_set:
                                    self.violations += 1
                                else:
                                    week_set.add(week)

            # Check overlaps between labs and lectures
            for group_id, labs in lab_same_semester.items():
                for lab in labs:
                    lab_course_id, lab_bitstring = lab
                    lab_day = int(lab_bitstring[:4], 2)
                    lab_session_start = int(lab_bitstring[4:8], 2)
                    lab_weeks = tuple(lab_bitstring[8:])

                    if group_id in lecture_same_semester:
                        for lecture in lecture_same_semester[group_id]:
                            lecture_course_id, lecture_bitstring = lecture
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
            parts = lecture.split('-')
            course_id, group_id, bitstring = parts[0], parts[1], parts[2]
            weeks = list(bitstring[8:])  # Extract weeks from the bitstring
            start_week = weeks.index('1') if '1' in weeks else -1  # Find the first week of the lecture
            lecture_start_weeks[f"{course_id}-{group_id}"] = start_week

        # Check labs and assign penalties
        for i, lab in enumerate(lab_population):
            # Extract lab information
            lab_parts = lab.split('-')
            lab_course_id, lab_group_id, lab_bitstring = lab_parts[0], lab_parts[2], lab_parts[3]
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
        
    def evaluate(self, course_id, group_id, chromosome):
        self.violations = 0

        weeks = list(chromosome[8:])
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

    def evaluate(self, course_id, group_id, chromosome):
        total_violations = 0
        for constraint in self.constraints:
            violations = constraint.evaluate(course_id, group_id, chromosome)
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
                    parts = individual.split('-')
                    course_id = parts[0]
                    group_id = parts[1] if len(parts) == 3 else parts[2]
                    session_info = parts[2] if len(parts) == 3 else parts[3]

                    violations = constraint.evaluate(course_id, group_id, session_info)
                    if violations:
                        print(f"Course {course_id} violates {constraint.name} with {violations} violations.")
                        violations_dict[name] += violations

        for name, count in violations_dict.items():
            print("Constraint:", name, " - Violations:", count)

        # Return as list of tuples, if needed
        return list(violations_dict.items())
