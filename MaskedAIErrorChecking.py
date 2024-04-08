import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from ucimlrepo import fetch_ucirepo

adult = fetch_ucirepo(id=2)
y = adult.data.targets
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]
y = y.apply(lambda x: 1 if x.strip() == '>50K' else 0)

X = adult.data.features

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(exclude=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ],
    remainder='passthrough',
    sparse_threshold=0
)

X_train_prepared = preprocessor.fit_transform(X_train)
X_test_prepared = preprocessor.transform(X_test)

# 10-20-1 NN, so d for input-to-hidden is 10*20=200
d = X_train_prepared.shape[1]
population_size = 2 * (10 * 20)

mutation_rate = 1 / d

suid_last_digit = 5  # My SUID : 413276285, last digit is 5
crossover_method = suid_last_digit % 4 # SUID modulo case

backprop_weights = np.random.randn(d)
backprop_mask = np.random.choice([0, 1], size=d, p=[0.9, 0.1])
backprop_weights *= backprop_mask

population = np.array([backprop_weights if i == 0 else np.random.randn(d) for i in range(population_size)])
for individual in population[1:]:
    mask = np.random.choice([0, 1], size=d, p=[0.5, 0.5])
    individual *= mask  # Apply the mask

def two_point_crossover(parent_a, parent_b):
    crossover_points = np.sort(np.random.choice(range(1, d - 1), 2, replace=False))
    child = np.concatenate([parent_a[:crossover_points[0]], parent_b[crossover_points[0]:crossover_points[1]], parent_a[crossover_points[1]:]])
    return child

def mutate(individual, mutation_rate):
    zero_indices = np.where(individual == 0)[0]
    if len(zero_indices) > 0:
        mutation_idx = np.random.choice(zero_indices)
        individual[mutation_idx] = np.random.randn() * mutation_rate
    return individual

def fitness_function(weights):
    predictions = sigmoid(X_train_prepared @ weights) > 0.5
    return -accuracy_score(y_train, predictions)  # Increase accuracy as to minimize its negative value

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Main genetic algorithm loop
best_individual = population[0]
best_fitness = fitness_function(best_individual)
test_accuracies = []  # Store test accuracies from each generation
last_improvement_generation = 0  # Track the last generation where an improvement was made

for generation in range(100):
    offspring = [two_point_crossover(population[np.random.randint(0, population_size)],
                                     population[np.random.randint(0, population_size)])
                 for _ in range(population_size - 1)]
    offspring = [mutate(child, mutation_rate) for child in offspring]
    offspring.append(best_individual.copy())

    fitness_scores = np.array([fitness_function(ind) for ind in offspring])
    best_current_fitness = np.min(fitness_scores)
    
    if best_current_fitness < best_fitness:
        best_fitness = best_current_fitness
        best_individual = offspring[np.argmin(fitness_scores)].copy()
        last_improvement_generation = generation  # Update last improvement
    elif generation - last_improvement_generation >= 5:
        # If no improvement for 5 consecutive generations, flag optima trap warning
        print("[LOCAL OPTIMA TRAP DETECTED and AVOIDED] Retest")
        last_improvement_generation = generation  
        population = np.array([np.random.randn(d) for _ in range(population_size)])  

    population = np.array(offspring)

    if generation % 10 == 0:  # Print progress to user
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

test_predictions = sigmoid(X_test_prepared @ best_individual) > 0.5
test_accuracy = accuracy_score(y_test, test_predictions)
test_accuracies.append(test_accuracy)  
cm = confusion_matrix(y_test, test_predictions)  # Calculate confusion matrix
print(f"Confusion Matrix:\n{cm}")
print(f"True Positives: {cm[1, 1]}")
print(f"True Negatives: {cm[0, 0]}")
print(f"False Positives: {cm[0, 1]}")
print(f"False Negatives: {cm[1, 0]}")
print(f"Final Test Accuracy: {test_accuracy}")
highest_overall_accuracy = max(test_accuracies)
print(f"Highest Test Accuracy Across All Trials: {highest_overall_accuracy}")