import random
import matplotlib.pyplot as plt

VALID_SESSION = range(3, 13)

class GeneticAlgorithm:
    def __init__(self, population_size, crossover_rate, mutation_rate, elitism, constraints_manager, hill_climbing_prob=0.3):
        self.population_size = population_size
        self.initial_cross_rate = crossover_rate
        self.initial_mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.constraints_manager = constraints_manager
        self.hill_climbing_prob = hill_climbing_prob

    def hill_climbing(self, individual):
        """Applies hill climbing to slightly improve an individual."""
        best_fitness = self.calculate_fitness(individual.split('-')[0], individual.split('-')[2])
        best_individual = individual
        
        for _ in range(4):  # Small number of hill climbing steps
            mutated_individual = self.mutate(best_individual, mutation_rate=0.3)
            fitness = self.calculate_fitness(mutated_individual.split('-')[0], mutated_individual.split('-')[2])
            if fitness < best_fitness:
                best_fitness = fitness
                best_individual = mutated_individual

        return best_individual

    def adapt_parameters(self, generation, max_generations):
        self.crossover_rate = self.initial_cross_rate * (1 - generation / max_generations)
        self.mutation_rate = self.initial_mutation_rate * (generation / max_generations)

    # def _generate_population(self, prefix_chromosomes, chromosome_length, preprocessed_df1, preprocessed_df2, is_lab=False, is_high_quality=False):
    #     population = []  # List to store individuals
        
    #     for i in range(self.population_size):
    #         prefix = str(prefix_chromosomes[i])
            
    #         # Determine individual type (high-quality or random)
    #         if is_high_quality and i < self.population_size // 2:
    #             individual = prefix + "-" + self.create_high_quality_individual(prefix_chromosomes[i], chromosome_length, preprocessed_df1, preprocessed_df2)
    #         elif is_lab:
    #             individual = prefix + "-" + self.create_individual(chromosome_length)
    #         else:
    #             individual = prefix + "-" + self.create_individual(chromosome_length)
            
    #         population.append(individual)

    #     return population

    def _generate_population(self, prefix_chromosomes, chromosome_length, preprocessed_df1, preprocessed_df2, is_lab, lecture_population=None):
        population = []  # List to store individuals
        
        for i in range(self.population_size):
            prefix = str(prefix_chromosomes[i])
            
            # Determine individual type (high-quality or random)
            if is_lab == False:
                individual = prefix + "-" + self.create_high_quality_individual(prefix_chromosomes[i], chromosome_length, preprocessed_df1, preprocessed_df2)
            else:
                individual = prefix + "-" + self.create_high_quality_individual_lab(prefix_chromosomes[i], chromosome_length, preprocessed_df1, preprocessed_df2, lecture_population)
            
            population.append(individual)

        return population

    def init_population(self, prefix_chromosomes, chromosome_length, preprocessed_df1, preprocessed_df2):
        # Initialize population with high-quality individuals first and then random ones
        # population = ['CO1007-CC01-01101011011111110111111', 'CO2003-CN01-11010111011111111011111']
        return self._generate_population(prefix_chromosomes, chromosome_length, preprocessed_df1, preprocessed_df2, is_lab=False)

    def init_lab_population(self, prefix_chromosomes, chromosome_length, preprocessed_df1, preprocessed_df2, lecture_population):
        # Initialize lab population with random individuals
        # population = ['CO1005-LAB1-CC01-01101011011111110111111', 'CO1005-LAB2-CC01-01101011011111110111111']
        return self._generate_population(prefix_chromosomes, chromosome_length, preprocessed_df1, preprocessed_df2, is_lab=True, lecture_population=lecture_population)

    def create_high_quality_individual(self, prefix_chromosome, chromosome_length, preprocessed_df1, preprocessed_df2):
        first_part = []
        
        # First 4 characters from 2 to 7 (in binary form)
        first_4_bits = format(random.randint(2, 7), '04b')
        first_part.extend(first_4_bits)
        
        # Next 4 characters from 2 to 10 (in binary form)
        next_4_bits = format(random.choice([2, 3, 4, 5, 7, 8, 9, 10, 11]), '04b')
        course_id = prefix_chromosome.split('-')[0]

        for index, row in preprocessed_df1.iterrows():
            if row['course_id'] == course_id:
                while int(next_4_bits, 2) + row['num_sessions'] - 1 not in VALID_SESSION:
                    if row['num_sessions'] == 3:
                        next_4_bits = format(random.choice([2, 3, 4, 7, 8, 9, 10, 11]), '04b')
                    else:
                        next_4_bits = format(random.choice([2, 3, 4, 5, 7, 8, 9, 10, 11]), '04b')

        for index, row in preprocessed_df2.iterrows():
            if row['course_id'] == course_id:
                while int(next_4_bits, 2) + row['num_sessions'] - 1 not in VALID_SESSION:
                    if row['num_sessions'] == 3:
                        next_4_bits = format(random.choice([2, 3, 4, 7, 8, 9, 10, 11]), '04b')
                    else:
                        next_4_bits = format(random.choice([2, 3, 4, 5, 7, 8, 9, 10, 11]), '04b')
        
        first_part.extend(next_4_bits)

        # Option 1: Generate random for week bitstring
        # remaining_length = chromosome_length - 16 - len(first_part)
        # first_part.extend(random.choice(['0', '1']) for _ in range(remaining_length))

        # Option 2: Generate bitstring with 1s for the first 15 weeks
        last_part = ['1'] * 16
        last_part[-9] = '0'  

        bitstring = first_part + last_part
        return ''.join(bitstring)

    def create_high_quality_individual_lab(self, prefix_chromosome, chromosome_length, preprocessed_df1, preprocessed_df2, lecture_population):
        first_part = []

        # First 4 characters from 2 to 7 (in binary form) -> Represents the day
        first_4_bits = format(random.randint(2, 7), '04b')
        first_part.extend(first_4_bits)

        # Next 4 characters (session start time) -> Allowed session starts: 2 or 8
        next_4_bits = format(random.choice([2, 8]), '04b')
        first_part.extend(next_4_bits)

        # Extract course_id and group_id
        course_id, lab_id, group_id = prefix_chromosome.split('-')[:3]

        # Default lab duration (weeks)
        num_weeks_lab = 0  

        # Retrieve lab duration from preprocessed_df1 if available
        for index, row in preprocessed_df1.iterrows():
            if row['course_id'] == course_id:
                num_weeks_lab = row['num_weeks_lab']
                break

        # Retrieve lab duration from preprocessed_df2 if available
        for index, row in preprocessed_df2.iterrows():
            if row['course_id'] == course_id:
                num_weeks_lab = row['num_weeks_lab']
                break
        
        # Determine the corresponding lecture start week (Lecture always starts at week 1)
        lecture_start_week = 0
        for lecture in lecture_population:
            lecture_parts = lecture.split('-')
            lecture_course_id, lecture_group_id, lecture_bitstring = lecture_parts[0], lecture_parts[1], lecture_parts[2]
            if lecture_course_id == course_id and lecture_group_id == group_id:
                lecture_weeks = list(lecture_bitstring[8:])
                if '1' in lecture_weeks:
                    lecture_start_week = lecture_weeks.index('1')  # First occurrence of '1' in lecture schedule
                break

        # Generate valid lab schedule ensuring:
        # 1. H8: Lab starts at least 4 weeks after lecture
        # 2. H9: No consecutive week scheduling

        min_start_week = lecture_start_week + 3  # Ensuring lab starts at least 4 weeks after lecture
        available_weeks = [week for week in range(min_start_week, 16) if week != 7]  # Available weeks from min_start_week to week 16 (exlcude week 8)

        # Ensure the lab weeks are spaced apart (H9)
        last_part = ['0'] * 16

        selected_weeks = []

        # Step 1: Select the first week randomly
        first_week = random.choice(available_weeks)
        selected_weeks.append(first_week)
        available_weeks.remove(first_week)

        # Step 2: Determine parity (even/odd) based on the first selection
        if first_week % 2 == 0:
            available_weeks = [w for w in available_weeks if w % 2 == 0]  # Only even weeks
        else:
            available_weeks = [w for w in available_weeks if w % 2 != 0]  # Only odd weeks

        # Step 3: Select the remaining weeks while ensuring non-consecutive selection
        while len(selected_weeks) < num_weeks_lab:
            if not available_weeks:  # Prevent infinite loop
                break

            week = random.choice(available_weeks)
            selected_weeks.append(week)
            available_weeks.remove(week)

        # Assign '1' to selected lab weeks
        for week in selected_weeks:
            last_part[week] = '1'

        # Combine parts into a bitstring
        bitstring = first_part + last_part
        return ''.join(bitstring)

    def create_individual(self, chromosome_length):
        return ''.join(str(random.randint(0, 1)) for _ in range(chromosome_length))

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


    def calculate_fitness(self, course_id, group_id, individual):
        # individual = '11010111011111111011111'
        fitness = self.constraints_manager.evaluate(course_id, group_id, individual)
        return fitness * 0.5

    def _evaluate_population_common(self, population, is_lab=False, lecture_population=None):
        scores = []
        course_group_tracker = {}
        
        for individual in population:
            # Split based on whether it's a regular or lab population
            parts = individual.split('-')
            if is_lab:
                course_id, lab_id, group_id, bitstring = parts[0], parts[1], parts[2], parts[3]
                key = f"{course_id}-{lab_id}-{group_id}"
            else:
                course_id, group_id, bitstring = parts[0], parts[1], parts[2]
                key = f"{course_id}-{group_id}"
            
            score = self.calculate_fitness(course_id, group_id, bitstring)
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

        print(f"Individual score: {max(scores)}")
        
        # Update scores with the population score
        for i in range(len(scores)):
            scores[i] += population_score[i] * 0.5
            scores[i] *= 1 / len(population)

        print(f"Population score: {population_score[0]}")

        return scores

    def evaluate_population(self, population):
        # Call the common evaluation function for regular population
        return self._evaluate_population_common(population)

    def evaluate_population_lab(self, population, _lecture_population):
        # Call the common evaluation function for lab population
        return self._evaluate_population_common(population, is_lab=True, lecture_population=_lecture_population)

    # Single-point crossover
    def crossover(self, parent1, parent2):
        parts = parent1.split('-')
        if len(parts) == 3:
            bitstring_length = len(parts[2])
        else:
            bitstring_length = len(parts[3])

        if random.random() < self.crossover_rate:
            # parent1 = 'CO2003-CN01-11010111011111111011111' || 'CO2003-LAB1-CN01-11010111011111111011111'
            # MARK: - Crossover point is set to 16 to avoid the last 16 bits
            if len(parts) == 3:
                crossover_point = len(parent1) - bitstring_length + random.randint(1, bitstring_length - 16 - 1)
            else:
                crossover_point = len(parent1) - bitstring_length + random.randint(1, bitstring_length - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
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
    def mutate(self, individual, mutation_rate=None):
        if mutation_rate is None:
            mutation_rate = self.mutation_rate

        # Split individual into its components
        parts = individual.split('-')

        if len(parts) == 3:
            # For regular individuals
            parts[2] = self._mutate_bitstring(parts[2], mutation_rate, exclude_last_bits=16)
        else:
            # For lab individuals
            parts[3] = self._mutate_bitstring(parts[3], mutation_rate)

        return '-'.join(parts)

    def repair_day(self, individual):
        course_id, group_id, bitstring = individual.split('-')
        day = int(bitstring[:4], 2)
        if day < 2 or day > 7:
            repaired_bitstring = format(random.randint(2, 7), '04b') + bitstring[4:]
            return f"{course_id}-{group_id}-{repaired_bitstring}"
        return individual

    def repair_day_lab(self, individual):
        course_id, lab_id, group_id, bitstring = individual.split('-')
        day = int(bitstring[:4], 2)
        if day < 2 or day > 7:
            repaired_bitstring = format(random.randint(2, 7), '04b') + bitstring[4:]
            return f"{course_id}-{lab_id}-{group_id}-{repaired_bitstring}"
        return individual

    def repair_session_start(self, individual, preprocessed_df1, preprocessed_df2):
        course_id, group_id, bitstring = individual.split('-')
        session_start = int(bitstring[4:8], 2)
        
        # Check against preprocessed_df1
        for index, row in preprocessed_df1.iterrows():
            if row['course_id'] == course_id:
                session_start_bitstring = bitstring[4:8]
                while session_start + row['num_sessions'] - 1 not in VALID_SESSION:
                    if row['num_sessions'] == 3:
                        session_start_bitstring = format(random.choice([2, 3, 4, 7, 8, 9, 10, 11]), '04b')
                        session_start = int(session_start_bitstring, 2)
                    else:
                        session_start_bitstring = format(random.choice([2, 3, 4, 5, 7, 8, 9, 10, 11]), '04b')
                        session_start = int(session_start_bitstring, 2)
                repaired_bitstring = bitstring[:4] + session_start_bitstring + bitstring[8:]
                return f"{course_id}-{group_id}-{repaired_bitstring}"
        
        # Check against preprocessed_df2
        for index, row in preprocessed_df2.iterrows():
            if row['course_id'] == course_id:
                session_start_bitstring = bitstring[4:8]
                while session_start + row['num_sessions'] - 1 not in VALID_SESSION:
                    if row['num_sessions'] == 3:
                        session_start_bitstring = format(random.choice([2, 3, 4, 7, 8, 9, 10, 11]), '04b')
                        session_start = int(session_start_bitstring, 2)
                    else:
                        session_start_bitstring = format(random.choice([2, 3, 4, 5, 7, 8, 9, 10, 11]), '04b')
                        session_start = int(session_start_bitstring, 2)
                repaired_bitstring = bitstring[:4] + session_start_bitstring + bitstring[8:]
                return f"{course_id}-{group_id}-{repaired_bitstring}"
        
        return individual

    def repair_session_start_lab(self, individual):
        course_id, lab_id, group_id, bitstring = individual.split('-')
        session_start = int(bitstring[4:8], 2)
        if session_start not in {2, 8}:
            session_start_bitstring = format(random.choice([2, 8]), '04b')
            session_start = int(session_start_bitstring, 2)
            repaired_bitstring = bitstring[:4] + session_start_bitstring + bitstring[8:]
            return f"{course_id}-{lab_id}-{group_id}-{repaired_bitstring}"
        return individual

    def repair_individual_lab(self, individual, preprocessed_df1, preprocessed_df2, lecture_population):
        course_id, lab_id, group_id, bitstring = individual.split('-')
        num_weeks_lab = 0

        # Retrieve lab duration from preprocessed_df1 if available
        for index, row in preprocessed_df1.iterrows():
            if row['course_id'] == course_id:
                num_weeks_lab = row['num_weeks_lab']
                break

        # Retrieve lab duration from preprocessed_df2 if available
        for index, row in preprocessed_df2.iterrows():
            if row['course_id'] == course_id:
                num_weeks_lab = row['num_weeks_lab']
                break
        
        # Determine the corresponding lecture start week (Lecture always starts at week 1)
        lecture_start_week = 0
        for lecture in lecture_population:
            lecture_parts = lecture.split('-')
            lecture_course_id, lecture_group_id, lecture_bitstring = lecture_parts[0], lecture_parts[1], lecture_parts[2]
            if lecture_course_id == course_id and lecture_group_id == group_id:
                lecture_weeks = list(lecture_bitstring[8:])
                if '1' in lecture_weeks:
                    lecture_start_week = lecture_weeks.index('1')  # First occurrence of '1' in lecture schedule
                break

        # Extract current lab weeks from the bitstring
        lab_weeks = list(bitstring[8:])
        scheduled_weeks = [idx for idx, week in enumerate(lab_weeks) if week == '1']

        # Flags to check violations
        violates_course_duration = len(scheduled_weeks) != num_weeks_lab
        violates_lecture_before_lab = any(week < lecture_start_week + 4 for week in scheduled_weeks)
        violates_lab_spacing = any(scheduled_weeks[i] + 1 == scheduled_weeks[i + 1] for i in range(len(scheduled_weeks) - 1))
        violates_midterm_break = 7 in scheduled_weeks  # Ensure no lab sessions in midterm week (Week 7)

        # If no violations, return the original individual
        if not (violates_course_duration or violates_lecture_before_lab or violates_lab_spacing or violates_midterm_break):
            return individual  # No changes needed

        # --- Repair Process ---
        min_start_week = lecture_start_week + 3  # Ensuring lab starts at least 4 weeks after lecture
        available_weeks = [w for w in range(min_start_week, 16) if w != 7]  # Available weeks from min_start_week to week 16 (exclude week 7)

        # Step 1: Select the first week randomly
        selected_weeks = []
        first_week = random.choice(available_weeks)
        selected_weeks.append(first_week)
        available_weeks.remove(first_week)

        # Step 2: Determine parity (even/odd) based on the first selection
        if first_week % 2 == 0:
            available_weeks = [w for w in available_weeks if w % 2 == 0]
        else:
            available_weeks = [w for w in available_weeks if w % 2 != 0]
        
        # Step 3: Select the remaining weeks while ensuring non-consecutive selection
        while len(selected_weeks) < num_weeks_lab:
            if not available_weeks:
                break
            week = random.choice(available_weeks)
            selected_weeks.append(week)
            available_weeks.remove(week)

        # Assign '1' to selected lab weeks
        last_part = ['0'] * 16
        for week in selected_weeks:
            last_part[week] = '1'

        bitstring = bitstring[:8] + ''.join(last_part)

        return '-'.join([course_id, lab_id, group_id, bitstring])

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
                        course_id, group_id = individual.split('-')[0], individual.split('-')[1]
                        score = scores[population.index(individual)]
                        if course_id not in best_population:
                            best_population[course_id] = {}
                        best_population[course_id][group_id] = [individual, score]
                    else:
                        course_id, lab_id, group_id = individual.split('-')[0], individual.split('-')[1], individual.split('-')[2]
                        score = scores[population.index(individual)]
                        if course_id not in best_population:
                            best_population[course_id] = {}
                        if group_id not in best_population[course_id]:
                            best_population[course_id][group_id] = {}
                        best_population[course_id][group_id][lab_id] = [individual, score]
                    
                print(f'Best Generation {i} - Fitness score: {fitness_score}')
                print(f'Best Generation {i}: {violations}')
                best_population_score = fitness_score

            fitness_scores.append(fitness_score)

            if fitness_score == 0:
                break

            # Sort the population by fitness in ascending order
            sorted_population, sorted_scores = zip(*sorted(zip(population, scores)))
            next_generation = []
            next_generation.extend(sorted_population[:self.elitism])

            if lecture_population is None:

                selected_parents = set((ind.split('-')[0], ind.split('-')[1]) for ind in next_generation)
                next_gen_set = set((ind.split('-')[0], ind.split('-')[1]) for ind in next_generation)

                while len(next_generation) < self.population_size:
                    available_population = [ind for ind in sorted_population[self.elitism:] if (ind.split('-')[0], ind.split('-')[1]) not in selected_parents]
                    available_scores = [score for ind, score in zip(sorted_population[self.elitism:], sorted_scores[self.elitism:]) if (ind.split('-')[0], ind.split('-')[1]) not in selected_parents]
                    
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

                    child1_course_id, child1_group_id = child1.split('-')[0], child1.split('-')[1]
                    child2_course_id, child2_group_id = child2.split('-')[0], child2.split('-')[1]

                    if child1 not in next_generation and (child1_course_id, child1_group_id) not in next_gen_set:
                        next_generation.append(child1)
                        next_gen_set.add((child1_course_id, child1_group_id))
                        selected_parents.add((parent1.split('-')[0], parent1.split('-')[1]))
                    
                    if child2 not in next_generation and len(next_generation) < self.population_size and (child2_course_id, child2_group_id) not in next_gen_set:
                        next_generation.append(child2)
                        next_gen_set.add((child2_course_id, child2_group_id))
                        selected_parents.add((parent2.split('-')[0], parent2.split('-')[1]))

                population = next_generation
            
            else:
                selected_parents = set((ind.split('-')[0], ind.split('-')[1], ind.split('-')[2]) for ind in next_generation)
                next_gen_set = set((ind.split('-')[0], ind.split('-')[1], ind.split('-')[2]) for ind in next_generation)

                while len(next_generation) < self.population_size:
                    available_population = [ind for ind in sorted_population[self.elitism:] if (ind.split('-')[0], ind.split('-')[1], ind.split('-')[2]) not in selected_parents]
                    available_scores = [score for ind, score in zip(sorted_population[self.elitism:], sorted_scores[self.elitism:]) if (ind.split('-')[0], ind.split('-')[1], ind.split('-')[2]) not in selected_parents]
                    
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

                    child1 = self.repair_day_lab(child1)
                    child2 = self.repair_day_lab(child2)

                    child1 = self.repair_session_start_lab(child1)
                    child2 = self.repair_session_start_lab(child2)

                    child1 = self.repair_individual_lab(child1, preprocessed_df1, preprocessed_df2, lecture_population)
                    child2 = self.repair_individual_lab(child2, preprocessed_df1, preprocessed_df2, lecture_population)

                    child1_course_id, child1_lab_id, child1_group_id = child1.split('-')[0], child1.split('-')[1], child1.split('-')[2]
                    child2_course_id, child2_lab_id, child2_group_id = child2.split('-')[0], child2.split('-')[1], child2.split('-')[2]

                    if child1 not in next_generation and (child1_course_id, child1_lab_id, child1_group_id) not in next_gen_set:
                        next_generation.append(child1)
                        next_gen_set.add((child1_course_id, child1_lab_id, child1_group_id))
                        selected_parents.add((parent1.split('-')[0], parent1.split('-')[1], parent1.split('-')[2]))
                    
                    if child2 not in next_generation and len(next_generation) < self.population_size and (child2_course_id, child2_lab_id, child2_group_id) not in next_gen_set:
                        next_generation.append(child2)
                        next_gen_set.add((child2_course_id, child2_lab_id, child2_group_id))
                        selected_parents.add((parent2.split('-')[0], parent2.split('-')[1], parent2.split('-')[2]))

                population = next_generation
        
        # Plot the fitness scores over generations
        plt.plot(range(len(fitness_scores)), fitness_scores, marker='o')
        plt.title('Fitness Score Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness Score')
        plt.grid(True)
        plt.show()

        return best_population
