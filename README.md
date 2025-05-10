# Optimizing Machine Learning Models with Genetic Algorithm Based Hyperparameter Tuning

This project explores the use of Genetic Algorithms (GA) to optimize hyperparameters for Machine Learning models, specifically:
  1. Random Forest for the Titanic dataset.
  2. Support Vector Machine (SVM) for the Breast Cancer dataset.

## Genetic Algorithm 
We applied Genetic Algorithm to search for optimal hyperparameters of machine learning models by simulating the process of natural selection. The main steps include:

### Initialization
  - A population of individuals (candidate solutions) is randomly generated.
  - Each individual encodes a set of hyperparameters (e.g., C, gamma for SVM).

### Fitness Evaluation
  - The fitness of each individual is evaluated using model accuracy on a validation set (e.g., k-fold cross-validation).

### Selection
  - The best-performing individuals are selected for reproduction.

  - Selection methods used: tournament selection or roulette wheel.

### Crossover (Recombination)
  - Pairs of individuals are combined to produce new offspring.
  Example: One-point or uniform crossover.

### Mutation
  - Some genes (hyperparameter values) are randomly altered to maintain diversity in the population.
  For instance, C might be increased/decreased slightly.

### Replacement
  - The new generation replaces the old one based on elitism or generational replacement strategy.

### Termination
  - The algorithm stops after a fixed number of generations or when no significant improvement is observed for a certain number of iterations.

## Python Code Sample
```python
def __init__(self, fitness_func, param_ranges, population_size=20, generations=50): 
        self.fitness_func = fitness_func 
        self.param_ranges = param_ranges 
        self.population_size = population_size 
        self.generations = generations 

def initialize_population(self): 
        population = [] 
        for _ in range(self.population_size): 
            individual = [np.random.uniform(low, high) for low, high in self.param_ranges] 
            population.append(individual) 
        return np.array(population) 

def select_parents(self, population, fitness): 
        sorted_idx = np.argsort(fitness) 
        return population[sorted_idx][:2] 
 
def crossover(self, parent1, parent2): 
        crossover_point = np.random.randint(len(parent1)) 
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:])) 
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:])) 
        return child1, child2 

def mutate(self, individual): 
        mutation_idx = np.random.randint(len(individual)) 
        individual[mutation_idx] = np.random.uniform(*self.param_ranges[mutation_idx]) 
        return individual 

def run(self): 
        population = self.initialize_population() 
        for generation in range(self.generations): 
            fitness = np.array([self.fitness_func(ind) for ind in population]) 
            parents = self.select_parents(population, fitness) 
            next_population = [] 
            for _ in range(self.population_size // 2): 
                child1, child2 = self.crossover(parents[0], parents[1]) 
                next_population.extend([self.mutate(child1), self.mutate(child2)]) 
            population = np.array(next_population) 
        best_individual = population[np.argmin([self.fitness_func(ind) for ind in population])] 
        return best_individual
```
## Result 
### Random Forest (Titanic)
| Model        | Accuracy   |
| ------------ | ---------- |
| Before GA    | 0.7692     |
| **After GA** | **0.8112** |

<img src="https://github.com/user-attachments/assets/8de7ea5a-0bff-4480-aed1-153a619ed217" alt="image" width="500"/>

### SVM (Breast Cancer)
| Model        | Accuracy   |
| ------------ | ---------- |
| Before GA    | 0.8947     |
| **After GA** | **0.9825** |

<img width="500" alt="image" src="https://github.com/user-attachments/assets/a0036673-cb8a-4edc-bb48-79e81a29d2f4" />

## Conclusion
Applying Genetic Algorithm for hyperparameter tuning significantly improved the performance of both models. These results demonstrate the effectiveness of GA in automating the hyperparameter search process, especially in non-convex search spaces where grid or random search may be inefficient.








