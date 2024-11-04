import matplotlib.pyplot as plt
from itertools import permutations 
from itertools import combinations
from random import shuffle
import random
import numpy as np
import statistics
import pandas as pd
import seaborn as SNs
import os

def initial_population(cities_list, n_population = 250):
    
    """
    Generating initial population of cities randomly selected from all 
    possible permutations  of the given cities
    Input:
    1- Cities list 
    2. Number of population 
    Output:
    Generated lists of cities
    """
    
    population_perms = []
    possible_perms = list(permutations(cities_list))
    random_ids = random.sample(range(0,len(possible_perms)),n_population)
    for i in random_ids:
        population_perms.append(list(possible_perms[i]))
        
    return population_perms

def dist_two_cities(city_1, city_2):
    
    """
    Calculating the distance between two cities  
    Input:
    1- City one name 
    2- City two name
    Output:
    Calculated Euclidean distance between two cities
    """
    
    city_1_coords = city_coords[city_1]
    city_2_coords = city_coords[city_2]
    return np.sqrt(np.sum((np.array(city_1_coords) - np.array(city_2_coords))**2))

def total_dist_individual(individual):
    
    """
    Calculating the total distance traveled by individual, 
    one individual means one possible solution (1 permutation)
    Input:
    1- Individual list of cities 
    Output:
    Total distance traveled 
    """
    
    total_dist = 0
    for i in range(0, len(individual)):
        if(i == len(individual) - 1):
            total_dist += dist_two_cities(individual[i], individual[0])
        else:
            total_dist += dist_two_cities(individual[i], individual[i+1])
    return total_dist

def fitness_prob(population):
    """
    Calculating the fitness probability 
    Input:
    1- Population  
    Output:
    Population fitness probability 
    """
    total_dist_all_individuals = []
    for i in range (0, len(population)):
        total_dist_all_individuals.append(total_dist_individual(population[i]))
        
    max_population_cost = max(total_dist_all_individuals)
    population_fitness = max_population_cost - total_dist_all_individuals
    population_fitness_sum = sum(population_fitness)
    population_fitness_probs = population_fitness / population_fitness_sum
    return population_fitness_probs


def selection(population, fitness_probs):
    """
    Implement a selection strategy based on proportionate roulette wheel
    Selection.
    Input:
    1- population
    2: fitness probabilities 
    Output:
    selected individual
    """
    population_fitness_probs_cumsum = fitness_probs.cumsum()
    bool_prob_array = population_fitness_probs_cumsum < np.random.uniform(0,1,1)
    selected_individual_index = len(bool_prob_array[bool_prob_array == True]) - 1
    return population[selected_individual_index]

def crossover(parent_1, parent_2):
    """
    Implement mating strategy using simple crossover between two parents
    Input:
    1- parent 1
    2- parent 2 
    Output:
    1- offspring 1
    2- offspring 2
    """
    n_cities_cut = len(cities_names) - 1
    cut = round(random.uniform(1, n_cities_cut))
    offspring_1 = []
    offspring_2 = []
    
    offspring_1 = parent_1 [0:cut]
    offspring_1 += [city for city in parent_2 if city not in offspring_1]
    
    
    offspring_2 = parent_2 [0:cut]
    offspring_2 += [city for city in parent_1 if city not in offspring_2]
    
    
    return offspring_1, offspring_2

def mutation(offspring):
    """
    Implement mutation strategy in a single offspring
    Input:
    1- offspring individual
    Output:
    1- mutated offspring individual
    """
    n_cities_cut = len(cities_names) - 1
    index_1 = round(random.uniform(0,n_cities_cut))
    index_2 = round(random.uniform(0,n_cities_cut))

    temp = offspring [index_1]
    offspring[index_1] = offspring[index_2]
    offspring[index_2] = temp
    return(offspring)



def save_route_image_per_gen(shortest_path, generation, total_distance, city_coords, output_dir):
    x_shortest = [city_coords[city][0] for city in shortest_path]
    y_shortest = [city_coords[city][1] for city in shortest_path]
    x_shortest.append(x_shortest[0])
    y_shortest.append(y_shortest[0])

    fig, ax = plt.subplots()
    ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
    plt.legend()

    # Adding shadow paths
    x = [city_coords[city][0] for city in city_coords]
    y = [city_coords[city][1] for city in city_coords]
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            ax.plot([x[i], x[j]], [y[i], y[j]], 'k-', alpha=0.09, linewidth=1)

    plt.title(label=f"TSP Best Route Using GA - Generation {generation + 1}",
              fontsize=25,
              color="k")

    for i, txt in enumerate(shortest_path):
        ax.annotate(str(i + 1) + "- " + txt, (x_shortest[i], y_shortest[i]), fontsize=20)

    plt.suptitle(f"Total Distance: {total_distance:.2f}", fontsize=18, y=0.95)

    fig.set_size_inches(16, 12)    
    plt.savefig(os.path.join(output_dir, f'solution_generation_{generation + 1}.png'))
    plt.close(fig)

def run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per, city_coords):
    population = initial_population(cities_names, n_population)
    fitness_probs = fitness_prob(population)
    
    parents_list = []
    for i in range(0, int(crossover_per * n_population)):
        parents_list.append(selection(population, fitness_probs))

    offspring_list = []    
    for i in range(0, len(parents_list), 2):
        offspring_1, offspring_2 = crossover(parents_list[i], parents_list[i+1])

        mutate_threshold = random.random()
        if mutate_threshold > (1 - mutation_per):
            offspring_1 = mutation(offspring_1)

        mutate_threshold = random.random()
        if mutate_threshold > (1 - mutation_per):
            offspring_2 = mutation(offspring_2)

        offspring_list.append(offspring_1)
        offspring_list.append(offspring_2)

    mixed_offspring = parents_list + offspring_list

    fitness_probs = fitness_prob(mixed_offspring)
    sorted_fitness_indices = np.argsort(fitness_probs)[::-1]
    best_fitness_indices = sorted_fitness_indices[0:n_population]
    best_mixed_offspring = [mixed_offspring[i] for i in best_fitness_indices]

    output_dir = 'imagePerGen'
    os.makedirs(output_dir, exist_ok=True)

    best_distances = []

    for generation in range(n_generations):
        fitness_probs = fitness_prob(best_mixed_offspring)
        parents_list = []
        for i in range(0, int(crossover_per * n_population)):
            parents_list.append(selection(best_mixed_offspring, fitness_probs))

        offspring_list = []    
        for i in range(0, len(parents_list), 2):
            offspring_1, offspring_2 = crossover(parents_list[i], parents_list[i+1])

            mutate_threshold = random.random()
            if mutate_threshold > (1 - mutation_per):
                offspring_1 = mutation(offspring_1)

            mutate_threshold = random.random()
            if mutate_threshold > (1 - mutation_per):
                offspring_2 = mutation(offspring_2)

            offspring_list.append(offspring_1)
            offspring_list.append(offspring_2)

        mixed_offspring = parents_list + offspring_list
        fitness_probs = fitness_prob(mixed_offspring)
        sorted_fitness_indices = np.argsort(fitness_probs)[::-1]
        best_fitness_indices = sorted_fitness_indices[0:int(0.8 * n_population)]
        best_mixed_offspring = [mixed_offspring[i] for i in best_fitness_indices]

        old_population_indices = [random.randint(0, (n_population - 1)) for j in range(int(0.2 * n_population))]
        for i in old_population_indices:
            best_mixed_offspring.append(population[i])
            
        random.shuffle(best_mixed_offspring)

        # Save the best route for the current generation
        shortest_path = best_mixed_offspring[0]
        total_distance = total_dist_individual(shortest_path)
        best_distances.append(total_distance)
        save_route_image_per_gen(shortest_path, generation, total_distance, city_coords, output_dir)
            
    return best_mixed_offspring, best_distances

def greedy_tsp(city_coords):
    n = len(city_coords)
    visited = [False] * n
    path = []
    total_distance = 0

    # Start from the first city
    current_city = 0
    path.append(current_city)
    visited[current_city] = True

    for _ in range(n - 1):
        nearest_city = None
        nearest_distance = float('inf')
        for next_city in range(n):
            if not visited[next_city]:
                distance = np.linalg.norm(np.array(city_coords[current_city]) - np.array(city_coords[next_city]))
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_city = next_city
        path.append(nearest_city)
        visited[nearest_city] = True
        total_distance += nearest_distance
        current_city = nearest_city

    # Return to the starting city
    total_distance += np.linalg.norm(np.array(city_coords[current_city]) - np.array(city_coords[path[0]]))
    path.append(path[0])

    return path, total_distance

# Parameters and data
n_population = 250
crossover_per = 0.8
mutation_per = 0.2
n_generations = 200
x = [36,-5,-20,2.5,33,
     52.5,38,46,65,1.62]
y = [138,120,30,112.5,65,
     5.75,-97,2,-18,15]
cities_names = ["Japan", "Indonesia", "Zimbabwe", "Malaysia", "Afghanistan", 
                "Netherlands", "United States", "France", "Iceland", "Sweden"]
city_coords = dict(zip(cities_names, zip(x, y)))

# Run the genetic algorithm and get the best routes per generation
best_routes_per_generation, best_distances_per_generation = run_ga(cities_names, n_population,
                                    n_generations, crossover_per,
                                    mutation_per, city_coords)

# Run the greedy algorithm
greedy_path, greedy_distance = greedy_tsp(list(city_coords.values()))

# Print the results
print("Genetic Algorithm Best Distance:", best_distances_per_generation[-1])
print("Greedy Algorithm Distance:", greedy_distance)

# Plot the convergence graph and comparison with Greedy Algorithm
generations = list(range(1, n_generations + 1))
plt.plot(generations, best_distances_per_generation, marker='o', label='Genetic Algorithm')
plt.axhline(y=greedy_distance, color='r', linestyle='--', label='Greedy Algorithm')
plt.title('Comparison of Genetic Algorithm and Greedy Algorithm')
plt.xlabel('Generasi')
plt.ylabel('Jarak Total')
plt.legend()
plt.grid(True)
plt.savefig('comparison_graph.png')
plt.show()

