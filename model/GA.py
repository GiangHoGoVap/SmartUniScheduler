import random
import matplotlib.pyplot as plt
from model.individual import Individual
from utils import get_same_semester_courses, get_lecture_by_course_group, get_lectures_by_courses, create_non_conflicting_time

VALID_SESSION = range(3, 13)

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

class GeneticAlgorithm:
    def __init__(self, population_size, crossover_rate, mutation_rate, elitism, constraints_manager):
        self.population_size = population_size
        self.initial_cross_rate = crossover_rate
        self.initial_mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.constraints_manager = constraints_manager

    def adapt_parameters(self, generation, max_generations):
        self.crossover_rate = self.initial_cross_rate * (1 - generation / max_generations)
        self.mutation_rate = self.initial_mutation_rate * (generation / max_generations)

    def _generate_population(self, prefix_chromosomes, chromosome_length, preprocessed_df1, preprocessed_df2, is_lab, lecture_population=None):
        population = []  # List to store individuals
        
        for i in range(self.population_size):
            prefix = str(prefix_chromosomes[i])
            
            if not is_lab:
                individual = self.create_high_quality_individual(prefix, chromosome_length, preprocessed_df1, preprocessed_df2)
            else:
                individual = self.create_high_quality_individual_lab(prefix, chromosome_length, preprocessed_df1, preprocessed_df2, lecture_population)
            
            population.append(individual)

        return population

    def init_population(self, prefix_chromosomes, chromosome_length, preprocessed_df1, preprocessed_df2):
        return self._generate_population(prefix_chromosomes, chromosome_length, preprocessed_df1, preprocessed_df2, is_lab=False)

    def init_lab_population(self, prefix_chromosomes, chromosome_length, preprocessed_df1, preprocessed_df2, lecture_population):
        return self._generate_population(prefix_chromosomes, chromosome_length, preprocessed_df1, preprocessed_df2, is_lab=True, lecture_population=lecture_population)

    def create_high_quality_individual(self, prefix_chromosome, chromosome_length, preprocessed_df1, preprocessed_df2):
        first_part = []

        # First 4 bits: day
        first_4_bits = format(random.randint(2, 6), '04b')
        first_part.extend(first_4_bits)

        # Next 4 bits: session start
        next_4_bits = format(random.choice([2, 3, 4, 5, 7, 8, 9, 10, 11]), '04b')
        course_id = prefix_chromosome.split('-')[0]
        group_id = prefix_chromosome.split('-')[1]  # e.g., "L01"

        # Adjust for session validity
        for df in [preprocessed_df1, preprocessed_df2]:
            for _, row in df.iterrows():
                if row['course_id'] == course_id:
                    while int(next_4_bits, 2) + row['num_sessions'] - 1 not in VALID_SESSION:
                        next_4_bits = format(random.choice([2, 3, 4, 5, 7, 8, 9, 10, 11]), '04b')

        first_part.extend(next_4_bits)

        # Week availability: 16 bits with '0' in the middle for test
        week_bits = ['1'] * 16
        week_bits[-9] = '0'

        full_bitstring = ''.join(first_part + week_bits)

        return Individual(course_id, group_id, full_bitstring, individual_type='lecture')

    def create_high_quality_individual_lab(self, prefix_chromosome, chromosome_length, preprocessed_df1, preprocessed_df2, lecture_population):
        course_id, lab_id, group_id = prefix_chromosome.split('-')[:3]

        first_part = []

        same_semester_courses = get_same_semester_courses(course_id, preprocessed_df1, preprocessed_df2)
        own_lecture = get_lecture_by_course_group(lecture_population, course_id, group_id)
        same_semester_lectures = get_lectures_by_courses(lecture_population, same_semester_courses)

        time_bits = create_non_conflicting_time(own_lecture, same_semester_lectures)

        if time_bits is None:
            return None  # Could not find a valid non-overlapping slot

        first_part.extend(time_bits)

        num_weeks_lab = 0
        for df in [preprocessed_df1, preprocessed_df2]:
            for _, row in df.iterrows():
                if row['course_id'] == course_id:
                    num_weeks_lab = row['num_weeks_lab']
                    break

        # Find lecture's starting week
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

        # Weeks available (at least 4 after lecture start)
        min_start_week = lecture_start_week + 3
        available_weeks = [w for w in range(min_start_week, 16) if w != 7]

        last_part = ['0'] * 16
        selected_weeks = []

        if not available_weeks:
            return None  # No valid weeks available

        first_week = random.choice(available_weeks)
        selected_weeks.append(first_week)
        available_weeks.remove(first_week)

        # Enforce parity (even/odd spacing)
        parity = first_week % 2
        available_weeks = [w for w in available_weeks if w % 2 == parity]

        while len(selected_weeks) < num_weeks_lab and available_weeks:
            week = random.choice(available_weeks)
            selected_weeks.append(week)
            available_weeks.remove(week)

        for week in selected_weeks:
            last_part[week] = '1'

        bitstring = ''.join(first_part + last_part)

        room_type = 'CS1' if group_id.startswith('CC') or group_id.startswith('CN') else 'CS2'
        room_list = ROOMS.get(course_id, {}).get(room_type, [])

        if not room_list:
            raise ValueError(f"No rooms available for course {course_id} with group type {room_type}")

        room_id = random.choice(room_list)

        return Individual(course_id=course_id, group_id=f"{lab_id}-{group_id}", bitstring=bitstring, individual_type='lab', room=room_id)

    # Roulette wheel selection
    def roulette_wheel_selection(self, population, scores):
        min_score = min(scores)
        if min_score < 0:
            scores = [score - min_score for score in scores]

        total_fitness = sum(scores)
        random_value = random.uniform(0, total_fitness)
        running_sum = 0
        for i in range(len(population)):
            running_sum += scores[i]
            if running_sum > random_value:
                return population[i]
        return population[-1]
    
    # Tournament selection
    def tournament_selection(self, population, scores, tournament_size):
        # Randomly select individuals for the tournament
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_individuals = [population[i] for i in tournament_indices]
        tournament_scores = [scores[i] for i in tournament_indices]

        # Choose the best individual from the tournament
        best_index = tournament_scores.index(min(tournament_scores))
        return tournament_individuals[best_index]


    def calculate_fitness(self, individual):
        fitness = self.constraints_manager.evaluate(individual)
        return fitness * 0.5

    def _evaluate_population_common(self, population, is_lab=False, lecture_population=None):
        scores = []
        course_group_tracker = {}
        
        for individual in population:
            # Determine key for tracking duplicates
            key = f"{individual.course_id}-{individual.group_id}"

            score = self.calculate_fitness(individual)
            scores.append(score)

            self.constraints_manager.reset_violations()

            # Track course-group combinations to detect duplicates
            if key not in course_group_tracker:
                course_group_tracker[key] = []
            course_group_tracker[key].append(individual)

        # Check for duplicates
        for key, individuals in course_group_tracker.items():
            if len(individuals) > 1:
                print(f"Duplicate course_id and group_id found: {key} with individuals: {individuals}")

        # Calculate population score based on constraints
        if is_lab:
            population_score = self.constraints_manager.evaluate_population(population, is_lab, lecture_population)
        else:
            population_score = self.constraints_manager.evaluate_population(population)
        
        # Update scores with the population score
        for i in range(len(scores)):
            scores[i] += population_score[i] * 0.5
            scores[i] *= 1 / len(population)

        return scores

    def evaluate_population(self, population):
        # Call the common evaluation function for regular population
        return self._evaluate_population_common(population)

    def evaluate_population_lab(self, population, _lecture_population):
        # Call the common evaluation function for lab population
        return self._evaluate_population_common(population, is_lab=True, lecture_population=_lecture_population)

    # Single-point crossover
    def crossover(self, parent1: Individual, parent2: Individual):
        bitstring1 = parent1.bitstring
        bitstring2 = parent2.bitstring
        bitstring_length = len(bitstring1)

        assert len(bitstring1) == len(bitstring2), "Parent bitstrings must be the same length"

        if random.random() < self.crossover_rate:
            # Avoid crossover within the last 16 bits (weeks)
            safe_length = bitstring_length - 16 if parent1.individual_type == 'lecture' else bitstring_length
            crossover_point = random.randint(1, safe_length - 1)

            new_bitstring1 = bitstring1[:crossover_point] + bitstring2[crossover_point:]
            new_bitstring2 = bitstring2[:crossover_point] + bitstring1[crossover_point:]

            child1 = Individual(parent1.course_id, parent1.group_id, new_bitstring1, parent1.individual_type, parent1.room)
            child2 = Individual(parent2.course_id, parent2.group_id, new_bitstring2, parent2.individual_type, parent2.room)
            return child1, child2

        return parent1, parent2

    def _mutate_bitstring(self, bitstring, mutation_rate, exclude_last_bits=0):
        # Mutate the bitstring, excluding the last `exclude_last_bits` bits if needed
        bitstring = list(bitstring)
        length = len(bitstring) - exclude_last_bits

        for i in range(length):
            if random.random() < mutation_rate:
                bitstring[i] = '1' if bitstring[i] == '0' else '0'

        return ''.join(bitstring)

    # Bit-flip mutation
    def mutate(self, individual: Individual, mutation_rate=None):
        if mutation_rate is None:
            mutation_rate = self.mutation_rate

        mutated_bitstring = self._mutate_bitstring(individual.bitstring, mutation_rate, exclude_last_bits=16 if individual.individual_type == 'lecture' else 0)
        mutated_individual = Individual(individual.course_id, individual.group_id, mutated_bitstring, individual.individual_type, individual.room)

        return mutated_individual

    def repair_day(self, individual: Individual):
        bitstring = individual.bitstring
        day = int(bitstring[:4], 2)

        if individual.individual_type == 'lab':
            valid_range = range(2, 8)  # 2 to 7 (inclusive)
        else:
            valid_range = range(2, 7)  # 2 to 6 (inclusive)

        if day not in valid_range:
            repaired_day = format(random.choice(valid_range), '04b')
            repaired_bitstring = repaired_day + bitstring[4:]
            return Individual(individual.course_id, individual.group_id, repaired_bitstring, individual.individual_type, individual.room)
        
        return individual

    def repair_session_start(self, individual: Individual, preprocessed_df1, preprocessed_df2):
        bitstring = individual.bitstring
        session_start = int(bitstring[4:8], 2)

        if individual.individual_type == 'lab':
            # Labs can only start at session 2 or 8
            if session_start not in {2, 8}:
                session_start_bitstring = format(random.choice([2, 8]), '04b')
                repaired_bitstring = bitstring[:4] + session_start_bitstring + bitstring[8:]
                return Individual(individual.course_id, individual.group_id, repaired_bitstring, 'lab', individual.room)
            return individual

        # For lecture individuals: consult preprocessed_df1 and preprocessed_df2
        num_sessions = None
        for df in [preprocessed_df1, preprocessed_df2]:
            row = df[df['course_id'] == individual.course_id]
            if not row.empty:
                num_sessions = row.iloc[0]['num_sessions']
                break

        if num_sessions is None:
            # Cannot find session info; return unchanged
            return individual

        # Check if session + duration fits in valid sessions
        while session_start + num_sessions - 1 not in VALID_SESSION:
            valid_choices = [2, 3, 4, 5, 7, 8, 9, 10, 11]
            if num_sessions == 3:
                valid_choices.remove(5)
            session_start = random.choice(valid_choices)
        
        session_start_bitstring = format(session_start, '04b')
        repaired_bitstring = bitstring[:4] + session_start_bitstring + bitstring[8:]

        return Individual(individual.course_id, individual.group_id, repaired_bitstring, 'lecture', None)

    def repair_individual_lab(self, individual: Individual, preprocessed_df1, preprocessed_df2, lecture_population: list[Individual]):
        course_id = individual.course_id
        group_id = individual.group_id.split('-')[1]  # Extract group ID from lab ID
        bitstring = individual.bitstring

        num_weeks_lab = 0

        # 1. Get num_weeks_lab from preprocessed data
        for df in [preprocessed_df1, preprocessed_df2]:
            row = df[df['course_id'] == course_id]
            if not row.empty:
                num_weeks_lab = row.iloc[0]['num_weeks_lab']
                break

        # 2. Find lecture start week (first '1' in weeks 8-)
        lecture_start_week = 0
        for lec in lecture_population:
            if lec.course_id == course_id and lec.group_id == group_id and lec.individual_type == 'lecture':
                lecture_weeks = list(lec.bitstring[8:])
                if '1' in lecture_weeks:
                    lecture_start_week = lecture_weeks.index('1')
                break

        # 3. Extract current lab weeks
        lab_weeks = list(bitstring[8:])
        scheduled_weeks = [i for i, c in enumerate(lab_weeks) if c == '1']

        # 4. Violation checks
        violates_duration = len(scheduled_weeks) != num_weeks_lab
        violates_start_time = any(w < lecture_start_week + 4 for w in scheduled_weeks)
        violates_spacing = any(scheduled_weeks[i] + 1 == scheduled_weeks[i + 1] for i in range(len(scheduled_weeks) - 1))
        violates_midterm = 7 in scheduled_weeks

        # Check time conflict with own lecture and same semester lectures
        lab_day = bitstring[:4]
        lab_session = bitstring[4:8]
        lab_session_int = int(lab_session, 2)
        lab_session_range = set(range(lab_session_int, lab_session_int + 5))

        same_semester_courses = get_same_semester_courses(course_id, preprocessed_df1, preprocessed_df2)
        own_lecture = get_lecture_by_course_group(lecture_population, course_id, group_id)
        same_semester_lectures = get_lectures_by_courses(lecture_population, same_semester_courses)

        violates_time_conflict = False

        # Check conflict with own lecture
        if own_lecture:
            lec_day = own_lecture.bitstring[:4]
            lec_session = int(own_lecture.bitstring[4:8], 2)
            if lec_day == lab_day and lec_session in lab_session_range:
                for w in scheduled_weeks:
                    if own_lecture.bitstring[8 + w] == '1':
                        violates_time_conflict = True
                        break

        # Check conflict with same semester lectures
        if not violates_time_conflict:
            for lec in same_semester_lectures:
                lec_day = lec.bitstring[:4]
                lec_session = int(lec.bitstring[4:8], 2)
                if lec_day == lab_day and lec_session in lab_session_range:
                    for w in scheduled_weeks:
                        if lec.bitstring[8 + w] == '1':
                            violates_time_conflict = True
                            break
                if violates_time_conflict:
                    break

        if not (violates_duration or violates_start_time or violates_spacing or violates_midterm or violates_time_conflict):
            return individual

        # 5. Begin repair
        min_start_week = lecture_start_week + 3
        available_weeks = [w for w in range(min_start_week, 16) if w != 7]

        if not available_weeks:
            return individual 

        selected_weeks = []
        first_week = random.choice(available_weeks)
        selected_weeks.append(first_week)
        available_weeks.remove(first_week)

        parity = first_week % 2
        available_weeks = [w for w in available_weeks if w % 2 == parity]

        while len(selected_weeks) < num_weeks_lab and available_weeks:
            week = random.choice(available_weeks)
            selected_weeks.append(week)
            available_weeks.remove(week)

        # 6. Repair day+session if violated
        if violates_time_conflict:
            time_bits = create_non_conflicting_time(own_lecture, same_semester_lectures)
            if time_bits is None:
                time_bits = bitstring[:8]  # fallback
        else:
            time_bits = bitstring[:8]

        weeks_part = ['0'] * 16
        for w in selected_weeks:
            weeks_part[w] = '1'
    
        repaired_bitstring = time_bits + ''.join(weeks_part)

        room_type = 'CS1' if group_id.startswith('CC') or group_id.startswith('CN') else 'CS2'
        room_list = ROOMS.get(course_id, {}).get(room_type, [])

        if not room_list:
            raise ValueError(f"No rooms available for course {course_id} with group type {room_type}")

        room_id = random.choice(room_list)

        return Individual(individual.course_id, individual.group_id, repaired_bitstring, 'lab', room_id)

    def run(self, prefix_chromosomes, chromosome_length, max_generations, preprocessed_df1, preprocessed_df2, lecture_population=None):
        if lecture_population is None:
            population = self.init_population(prefix_chromosomes, chromosome_length, preprocessed_df1, preprocessed_df2)
        else:
            population = self.init_lab_population(prefix_chromosomes, chromosome_length, preprocessed_df1, preprocessed_df2, lecture_population)
        
        best_population = {}
        best_population_score = float('+inf')
        fitness_scores = []

        for i in range(max_generations):
            # self.adapt_parameters(i, max_generations)

            if lecture_population is None:
                scores = self.evaluate_population(population)
                violations = self.constraints_manager.count_violations(population)
            else:
                scores = self.evaluate_population_lab(population, lecture_population)
                violations = self.constraints_manager.count_violations(population, lecture_population)

            fitness_score = sum(scores) 
            print(f'Generation {i} - Fitness score: {fitness_score}')
            
            print(f'Generation {i}: {violations}')
            
            if fitness_score < best_population_score:
                # Update the best population across all generations
                for individual in population:
                    if lecture_population is None:
                        score = scores[population.index(individual)]
                        if individual.course_id not in best_population:
                            best_population[individual.course_id] = {}
                        best_population[individual.course_id][individual.group_id] = [individual, score]
                    else:
                        lab_id, group_id = individual.group_id.split('-')
                        score = scores[population.index(individual)]
                        if individual.course_id not in best_population:
                            best_population[individual.course_id] = {}
                        if group_id not in best_population[individual.course_id]:
                            best_population[individual.course_id][group_id] = {}
                        best_population[individual.course_id][group_id][lab_id] = [individual, score]
                    
                print(f'Best Generation {i} - Fitness score: {fitness_score}')
                print(f'Best Generation {i}: {violations}')
                best_population_score = fitness_score

            fitness_scores.append(fitness_score)

            if fitness_score == 0:
                break

            # Sort the population by fitness in ascending order
            sorted_population, sorted_scores = zip(*sorted(zip(population, scores), key=lambda x: x[1]))
            next_generation = []
            next_generation.extend(sorted_population[:self.elitism])

            if lecture_population is None:

                selected_parents = set((ind.course_id, ind.group_id) for ind in next_generation)
                next_gen_set = set((ind.course_id, ind.group_id) for ind in next_generation)

                while len(next_generation) < self.population_size:
                    available_population = [ind for ind in sorted_population[self.elitism:] if (ind.course_id, ind.group_id) not in selected_parents]
                    available_scores = [score for ind, score in zip(sorted_population[self.elitism:], sorted_scores[self.elitism:]) if (ind.course_id, ind.group_id) not in selected_parents]
                    
                    if not available_population:
                        break

                    if len(available_population) < 2:
                        parent1 = parent2 = available_population[0]
                    else:
                        # parent1 = self.roulette_wheel_selection(available_population, available_scores)
                        tournament_size = min(5, len(available_population))
                        parent1 = self.tournament_selection(available_population, available_scores, tournament_size)

                        filtered_population = [ind for ind in available_population if ind != parent1]
                        filtered_scores = [score for ind, score in zip(available_population, available_scores) if ind != parent1]
                        # parent2 = self.roulette_wheel_selection(filtered_population, filtered_scores)
                        tournament_size = min(5, len(filtered_population))
                        parent2 = self.tournament_selection(filtered_population, filtered_scores, tournament_size)
                    
                    child1, child2 = self.crossover(parent1, parent2)
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)

                    child1 = self.repair_day(child1)
                    child2 = self.repair_day(child2)

                    child1 = self.repair_session_start(child1, preprocessed_df1, preprocessed_df2)
                    child2 = self.repair_session_start(child2, preprocessed_df1, preprocessed_df2)

                    if child1 not in next_generation and (child1.course_id, child1.group_id) not in next_gen_set:
                        next_generation.append(child1)
                        next_gen_set.add((child1.course_id, child1.group_id))
                        selected_parents.add((parent1.course_id, parent1.group_id))
                    
                    if child2 not in next_generation and len(next_generation) < self.population_size and (child2.course_id, child2.group_id) not in next_gen_set:
                        next_generation.append(child2)
                        next_gen_set.add((child2.course_id, child2.group_id))
                        selected_parents.add((parent2.course_id, parent2.group_id))

                population = next_generation
            
            else:
                selected_parents = set((ind.course_id, ind.group_id.split('-')[0], ind.group_id.split('-')[1]) for ind in next_generation)
                next_gen_set = set((ind.course_id, ind.group_id.split('-')[0], ind.group_id.split('-')[1]) for ind in next_generation)

                while len(next_generation) < self.population_size:
                    available_population = [ind for ind in sorted_population[self.elitism:] if (ind.course_id, ind.group_id.split('-')[0], ind.group_id.split('-')[1]) not in selected_parents]
                    available_scores = [score for ind, score in zip(sorted_population[self.elitism:], sorted_scores[self.elitism:]) if (ind.course_id, ind.group_id.split('-')[0], ind.group_id.split('-')[1]) not in selected_parents]
                    
                    if not available_population:
                        break

                    if len(available_population) < 2:
                        parent1 = parent2 = available_population[0]
                    else:
                        # parent1 = self.roulette_wheel_selection(available_population, available_scores)
                        tournament_size = min(5, len(available_population))
                        parent1 = self.tournament_selection(available_population, available_scores, tournament_size)

                        filtered_population = [ind for ind in available_population if ind != parent1]
                        filtered_scores = [score for ind, score in zip(available_population, available_scores) if ind != parent1]
                        # parent2 = self.roulette_wheel_selection(filtered_population, filtered_scores)
                        tournament_size = min(5, len(filtered_population))
                        parent2 = self.tournament_selection(filtered_population, filtered_scores, tournament_size)
                    
                    child1, child2 = self.crossover(parent1, parent2)
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)

                    child1 = self.repair_day(child1)
                    child2 = self.repair_day(child2)

                    child1 = self.repair_session_start(child1, preprocessed_df1, preprocessed_df2)
                    child2 = self.repair_session_start(child2, preprocessed_df1, preprocessed_df2)

                    child1 = self.repair_individual_lab(child1, preprocessed_df1, preprocessed_df2, lecture_population)
                    child2 = self.repair_individual_lab(child2, preprocessed_df1, preprocessed_df2, lecture_population)

                    child1_lab_id, child1_group_id = child1.group_id.split('-')
                    child2_lab_id, child2_group_id = child2.group_id.split('-')

                    if child1 not in next_generation and (child1.course_id, child1_lab_id, child1_group_id) not in next_gen_set:
                        next_generation.append(child1)
                        next_gen_set.add((child1.course_id, child1_lab_id, child1_group_id))
                        selected_parents.add((parent1.course_id, parent1.group_id.split('-')[0], parent1.group_id.split('-')[1]))
                    
                    if child2 not in next_generation and len(next_generation) < self.population_size and (child2.course_id, child2_lab_id, child2_group_id) not in next_gen_set:
                        next_generation.append(child2)
                        next_gen_set.add((child2.course_id, child2_lab_id, child2_group_id))
                        selected_parents.add((parent2.course_id, parent2.group_id.split('-')[0], parent2.group_id.split('-')[1]))

                population = next_generation
        
        # Plot the fitness scores over generations
        plt.plot(range(len(fitness_scores)), fitness_scores, marker='o')
        plt.title('Fitness Score Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness Score')
        plt.grid(True)
        plt.show()

        return best_population
