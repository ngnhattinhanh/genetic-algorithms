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
## Diagram
<img width="450" height="442" alt="image" src="https://github.com/user-attachments/assets/54f6df2a-2db7-4182-bcb8-72d6b1416da8" />

## Pseudocode
```python
Begin
    k = 0
    P(k) = form_initial_population (N)
    fs(k) = evaluate_population(P(k))
    while not (termination_criteria)
        k = k + 1
        M(k) = select_parent(P(k-1), fs(k-1))
        P(k) = crossover(M(k), Pc)
        P(k) = mutation(P(k), Pm)
        fs(k) = evaluate_population(P(k))
        [better, worse, average] = save_solution(P(k), fs(k))
    end
End
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








