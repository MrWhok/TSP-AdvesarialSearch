import matplotlib.pyplot as plt
from itertools import permutations
from random import shuffle
import random
import numpy as np
import os
import time
import logging

logging.basicConfig(level=logging.INFO)

class GeneticAlgorithm:
    def __init__(self, regions_names, city_coords, n_population, n_generations,selection_per, crossover_per, mutation_per):
        self.regions_names = regions_names
        self.city_coords = city_coords
        self.n_population = n_population
        self.n_generations = n_generations
        self.selection_per = selection_per
        self.crossover_per = crossover_per
        self.mutation_per = mutation_per
        self.output_dir = 'imagePerGen'
        os.makedirs(self.output_dir, exist_ok=True)
        
    def run(self):
        populations_list=self._initial_population()
        best_distances_per_generation=[]
        
        for generation in range(self.n_generations):
            fitness_populations=self._calculate_fitness(populations_list)
            parents_list=self._selection(populations_list,fitness_populations)
        
            offspring_list=[]
            for i in range(0,len(parents_list),2):
                offspring_1,offspring2=self._crossover(parents_list[i],parents_list[i+1])
                
                tresshold=random.random()
                if tresshold<self.mutation_per:
                    offspring_1=self._mutation(offspring_1)
                tresshold=random.random()
                if tresshold<self.mutation_per:
                    offspring2=self._mutation(offspring2)
                offspring_list.extend([offspring_1,offspring2])
                
            shortest_path, populations_list=self._create_new_population(parents_list,offspring_list)
            total_distance = self.__total_distance(shortest_path)
            best_distances_per_generation.append(total_distance)
            self.__save_route_image_per_gen(shortest_path, generation, total_distance)
        
        return best_distances_per_generation
            
    
    #Made random initial populations
    def _initial_population(self):
        population_list=[]
        all_possible_perm=list(permutations(self.regions_names))
        random_possible_perm_index=random.sample(range(0,len(all_possible_perm)),self.n_population) #randomly get indexes of possible permutations
        # print(random_possible_perm)
        for i in random_possible_perm_index:
            population_list.append(list(all_possible_perm[i]))
            
        return population_list
    
    
    #Calculate fitness of each population
    def _calculate_fitness(self,population_list):
        total_distane_each_population=[self.__total_distance(i) for i in population_list]
        
        max_population_cost=max(total_distane_each_population)
        population_fitness=max_population_cost-np.array(total_distane_each_population)
        population_fitness_sum=sum(population_fitness)
        
        # if population_fitness_sum == 0:
        #     population_fitness_sum = 1e-10
            
        fitness_populations_final=population_fitness/population_fitness_sum  #higher fitness mean lower distance

        # print("population fitness:",population_fitness)
        # print(fitness_populations)
        
        return fitness_populations_final
    
    
    # Selection of parents using rank selction method
    def _selection(self,population_list,fitness_list):
            sorted_population = [x for _, x in sorted(zip(fitness_list, population_list))]
            # logging.info(sorted_population)
            rank_probs = np.array(range(1, len(population_list) + 1)) / sum(range(1, len(population_list) + 1))
            # logging.info(rank_probs)
            selected = random.choices(sorted_population, weights=rank_probs, k=int(self.selection_per * len(population_list)))
            # logging.info(selected)
            return selected
        
    
    #Crossover between two parents
    def _crossover(self,parent_1,parent_2):
        total_region=len(self.regions_names)-1
        cut=random.randint(1,total_region)
        offspring_1=parent_1[:cut]+[city for city in parent_2 if city not in parent_1[:cut]]
        offspring_2=parent_2[:cut]+[city for city in parent_1 if city not in parent_2[:cut]]
        return offspring_1,offspring_2
    
    
    #Mutation of an offspring by swapping two regions
    def _mutation(self,offspring):
        total_region=len(self.regions_names)-1
        index_1=random.randint(0,total_region)
        index_2=random.randint(0,total_region)
        offspring[index_1],offspring[index_2]=offspring[index_2],offspring[index_1]
        return offspring
    
    
    #Create new population
    def _create_new_population(self,parent_list,offspring_list):
        mixed_offspring=parent_list+offspring_list
        fitness_probs=self._calculate_fitness(mixed_offspring)
        sorted_fitness_indices = np.argsort(fitness_probs)[::-1]
        
        best_fitness_indices = sorted_fitness_indices[:int(0.8 * self.n_population)]
        populations_list = [mixed_offspring[i] for i in best_fitness_indices]
        
        old_population_indices = random.sample(range(self.n_population), int(0.2 * self.n_population))
        populations_list.extend([mixed_offspring[i] for i in old_population_indices])
        
        shortest_path = populations_list[0]
        random.shuffle(populations_list)
        return shortest_path,populations_list

    
    
    #Calculate total distance of each population
    def __total_distance(self,population):
        total_distance=0
        for i in range(0,len(population)):
            if i==len(population)-1:
                total_distance+=self.__distance_2_regions(population[i],population[0])
            else:
                total_distance+=self.__distance_2_regions(population[i],population[i+1])
        return total_distance
    
    
    #Calculate distance between two regions (Using Euclidean distance)
    def __distance_2_regions(self,city1,city2):
        city1_coords=self.city_coords[city1]
        city2_coords=self.city_coords[city2]
        return np.sqrt(np.sum((np.array(city1_coords)-np.array(city2_coords))**2))
    
    
    # Save route image per generation
    def __save_route_image_per_gen(self, shortest_path, generation, total_distance):
        x_shortest = [self.city_coords[city][0] for city in shortest_path]
        y_shortest = [self.city_coords[city][1] for city in shortest_path]
        x_shortest.append(x_shortest[0])
        y_shortest.append(y_shortest[0])

        fig, ax = plt.subplots()
        ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
        plt.legend()

        x = [self.city_coords[city][0] for city in self.city_coords]
        y = [self.city_coords[city][1] for city in self.city_coords]
        
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                ax.plot([x[i], x[j]], [y[i], y[j]], 'k-', alpha=0.09, linewidth=1)

        plt.title(label=f"TSP Best Route Using GA - Generation {generation + 1}", fontsize=25, color="k")
        
        for i, txt in enumerate(shortest_path):
            ax.annotate(str(i + 1) + "- " + txt, (x_shortest[i], y_shortest[i]), fontsize=20)

        plt.suptitle(f"Total Distance: {total_distance:.2f}", fontsize=18, y=0.95)
        
        fig.set_size_inches(16, 12)
        
        plt.savefig(os.path.join(self.output_dir, f'solution_generation_{generation + 1}.png'))
        plt.close(fig)

n_population = 250
selection_per = 0.8
crossover_per = 0.8
mutation_per = 0.29
n_generations = 30
x = [36, -5, -20, 2.5, 33, 52.5, 38, 46, 65, 1.62]
y = [138, 120, 30, 112.5, 65, 5.75, -97, 2, -18, 15]
regions_names = ["Japan", "Indonesia", "Zimbabwe", "Malaysia", "Afghanistan", "Netherlands", "United States", "France", "Iceland", "Sweden"]
city_coords = dict(zip(regions_names, zip(x, y)))

# Run the genetic algorithm
ga = GeneticAlgorithm(regions_names, city_coords, n_population, n_generations,selection_per, crossover_per, mutation_per)
start_time_ga = time.time()
best_distances_per_generation = ga.run()
end_time_ga = time.time()
time_taken_ga = end_time_ga - start_time_ga

print("Genetic Algorithm Best Distance:", best_distances_per_generation[-1])
print("Time taken by Genetic Algorithm:", time_taken_ga, "seconds")
print("Best Distance per Generation:",best_distances_per_generation)
