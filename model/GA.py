import random
import matplotlib.pyplot as plt

VALID_DAY = range(2, 8)
VALID_SESSION = range(2, 11)

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

    def hill_climbing(self, individual, course_id):
        """Applies hill climbing to slightly improve an individual."""
        best_fitness = self.calculate_fitness(course_id, individual.split('-')[2])
        best_individual = individual
        
        for i in range(1, 5):  # Small number of hill climbing steps
            mutated_individual = self.mutate(best_individual)
            fitness = self.calculate_fitness(course_id, mutated_individual.split('-')[2])
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = mutated_individual

        return best_individual

    def adapt_parameters(self, generation, max_generations):
        self.crossover_rate = self.initial_cross_rate * (1 - generation / max_generations)
        self.mutation_rate = self.initial_mutation_rate * (generation / max_generations)

    def init_population(self, prefix_chromosomes, chromosome_length):
        population = [] # ['CO1007-CC01-01101011011111110111111', 'CO2003-CN01-11010111011111111011111']
        
        # Seed the first portion of the population with high-quality individuals
        for i in range(self.population_size):
            high_quality_individual = str(prefix_chromosomes[i]) + "-" + self.create_high_quality_individual(chromosome_length)
            population.append(high_quality_individual)

        # Fill the rest of the population randomly
        # for i in range(self.population_size // 2, self.population_size):
        #     random_individual = str(prefix_chromosomes[i]) + "-" + self.create_individual(chromosome_length)
        #     population.append(random_individual)

        return population

    def create_high_quality_individual(self, chromosome_length):
        first_part = []
        
        # First 3 characters from 2 to 7 (in binary form)
        first_3_bits = format(random.randint(2, 7), '03b')
        first_part.extend(first_3_bits)
        
        # Next 4 characters from 2 to 10 (in binary form)
        next_4_bits = format(random.randint(2, 10), '04b')
        first_part.extend(next_4_bits)
        
        remaining_length = chromosome_length - 16 - len(first_part)
        first_part.extend(random.choice(['0', '1']) for _ in range(remaining_length))

        last_part = ['1'] * 16
        last_part[-9] = '0'  

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
        best_index = tournament_scores.index(max(tournament_scores))
        return tournament_individuals[best_index]


    def calculate_fitness(self, course_id, individual):
        # individual = '11010111011111111011111'
        return self.constraints_manager.evaluate(course_id, individual) 

    def evaluate_population(self, population):
        scores = []
        course_group_tracker = {}

        for individual in population:
            # 'CO2003-CN01-11010111011111111011111'
            course_id, group_id, bitstring = individual.split('-')[0], individual.split('-')[1], individual.split('-')[2]

            scores.append(self.calculate_fitness(course_id, bitstring))

            key = f"{course_id}-{group_id}"
            if key not in course_group_tracker:
                course_group_tracker[key] = []
            course_group_tracker[key].append(individual)

        # Check for duplicates
        for key, individuals in course_group_tracker.items():
            if len(individuals) > 1:
                print(f"Duplicate course_id and group_id found: {key} with individuals: {individuals}")

        population_score = self.constraints_manager.evaluate_population(population)
        for i in range(len(scores)):
            scores[i] += population_score

        return scores

    # Single-point crossover
    def crossover(self, parent1, parent2):
        parts = parent1.split('-')
        bitstring_length = len(parts[2])

        if random.random() < self.crossover_rate:
            # parent1 = 'CO2003-CN01-11010111011111111011111'
            crossover_point = len(parent1) - bitstring_length + random.randint(1, bitstring_length - 16 - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            return child1, child2
        return parent1, parent2

    # Bit-flip mutation
    def mutate(self, individual):
        # individual = 'CO2003-CN01-11010111011111111011111'
        parts = individual.split('-')
        bitstring = list(parts[2])

        # MARK: Exclude last 16 bits (weeks) from mutation (LT)
        for i in range(len(bitstring) - 16):
            if random.random() < self.mutation_rate:
                bitstring[i] = '1' if bitstring[i] == '0' else '0'

        parts[2] = ''.join(bitstring)
        individual = '-'.join(parts)

        return individual

    def run(self, prefix_chromosomes, chromosome_length, max_generations):
        population = self.init_population(prefix_chromosomes, chromosome_length)
        best_population = {}
        best_population_score = float('-inf')
        fitness_scores = []

        for i in range(max_generations):
            self.adapt_parameters(i, max_generations)
        
            scores = self.evaluate_population(population)
            fitness_score = sum(scores) 
            print(f'Generation {i} - Fitness score: {fitness_score}')
            
            violations = self.constraints_manager.count_violations(population)
            print(f'Generation {i}: {violations}')
            
            if fitness_score > best_population_score:
                # Update the best population across all generations
                for individual in population:
                    course_id, group_id = individual.split('-')[0], individual.split('-')[1]
                    score = scores[population.index(individual)]
                    if course_id not in best_population:
                        best_population[course_id] = {}
                    best_population[course_id][group_id] = [individual, score]
                print(f'Best Generation {i} - Fitness score: {fitness_score}')
                print(f'Best Generation {i}: {violations}')
                best_population_score = fitness_score

            fitness_scores.append(fitness_score)

            # Hill Climbing step
            for idx in range(self.elitism):
                if random.random() < self.hill_climbing_prob:  # Only apply hill climbing with some probability
                    individual = population[idx]
                    course_id = individual.split('-')[0]
                    population[idx] = self.hill_climbing(individual, course_id)

            # Sort the population by fitness in descending order
            sorted_population, sorted_scores = zip(*sorted(zip(population, scores), reverse=True))
            next_generation = []
            next_generation.extend(sorted_population[:self.elitism])

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
                
                child1_course_id, child1_group_id, child1_bitstring = child1.split('-')
                child2_course_id, child2_group_id, child2_bitstring = child2.split('-')

                child1_day = int(child1_bitstring[:3], 2)
                child1_session = int(child1_bitstring[3:7], 2)

                valid_session_choices = [session for session in VALID_SESSION if session != 6]

                if child1_day not in VALID_DAY:
                    child1_day_bits = format(random.choice(VALID_DAY), '03b')
                else:
                    child1_day_bits = child1_bitstring[:3]

                if child1_session not in VALID_SESSION:
                    child1_session_bits = format(random.choice(valid_session_choices), '04b')
                else:
                    child1_session_bits = child1_bitstring[3:7]

                child1_bitstring = child1_day_bits + child1_session_bits + child1_bitstring[7:]
                child1 = f"{child1_course_id}-{child1_group_id}-{child1_bitstring}"

                child2_day = int(child2_bitstring[:3], 2)
                child2_session = int(child2_bitstring[3:7], 2)
                
                if child2_day not in VALID_DAY:
                    child2_day_bits = format(random.choice(VALID_DAY), '03b')
                else:
                    child2_day_bits = child2_bitstring[:3]

                if child2_session not in VALID_SESSION:
                    child2_session_bits = format(random.choice(valid_session_choices), '04b')
                else:
                    child2_session_bits = child2_bitstring[3:7]

                child2_bitstring = child2_day_bits + child2_session_bits + child2_bitstring[7:]
                child2 = f"{child2_course_id}-{child2_group_id}-{child2_bitstring}"

                if child1 not in next_generation and (child1_course_id, child1_group_id) not in next_gen_set:
                    next_generation.append(child1)
                    next_gen_set.add((child1_course_id, child1_group_id))
                    selected_parents.add((parent1.split('-')[0], parent1.split('-')[1]))
                
                if child2 not in next_generation and len(next_generation) < self.population_size and (child2_course_id, child2_group_id) not in next_gen_set:
                    next_generation.append(child2)
                    next_gen_set.add((child2_course_id, child2_group_id))
                    selected_parents.add((parent2.split('-')[0], parent2.split('-')[1]))

            population = next_generation
        
        # Plot the fitness scores over generations
        plt.plot(range(max_generations), fitness_scores, marker='o')
        plt.title('Fitness Score Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness Score')
        plt.grid(True)
        plt.show()

        return best_population