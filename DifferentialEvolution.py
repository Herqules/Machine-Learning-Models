import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from ucimlrepo import fetch_ucirepo


def sigmoid(x):
    # Clip input to prevent overflow
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))



def fitness_function(weights, X, y):
    predictions = sigmoid(X @ weights) > 0.5
    return -accuracy_score(y, predictions)  # We want to maximize accuracy, so we minimize its negative


def differential_evolution(X_train, y_train, population_size, F, CR, generations):
    d = X_train.shape[1]  # Number of features
    population = np.array([np.random.randn(d) for _ in range(population_size)])
    
    best_individual = population[0]
    best_fitness = fitness_function(best_individual, X_train, y_train)
    test_accuracies = []

    for generation in range(generations):
        for i in range(population_size):
            target = population[i]
            indices = [idx for idx in range(population_size) if idx != i]
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant_vector = a + F * (b - c)
            trial_vector = np.where(np.random.rand(d) < CR, mutant_vector, target)
            
            if fitness_function(trial_vector, X_train, y_train) < fitness_function(target, X_train, y_train):
                population[i] = trial_vector
                if fitness_function(trial_vector, X_train, y_train) < best_fitness:
                    best_fitness = fitness_function(trial_vector, X_train, y_train)
                    best_individual = trial_vector

        if generation % 10 == 0:
            print(f"Generation {generation}: Best Fitness = {best_fitness}")

    return best_individual


# Fetching dataset...
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

# Parameters for Differential Evolution
F = 0.8  # Differential weight
CR = 0.9  # Crossover probability
generations = 100  # Number of generations
population_size = 40  # Assuming population size of 40 for demonstration

# Run Differential Evolution
best_weights = differential_evolution(X_train_prepared, y_train, population_size, F, CR, generations)

# Evaluate the best individual's performance on the test set
test_predictions = sigmoid(X_test_prepared @ best_weights) > 0.5
test_accuracy = accuracy_score(y_test, test_predictions)
cm = confusion_matrix(y_test, test_predictions)

print(f"Confusion Matrix:\n{cm}")
print(f"True Positives: {cm[1, 1]}")
print(f"True Negatives: {cm[0, 0]}")
print(f"False Positives: {cm[0, 1]}")
print(f"False Negatives: {cm[1, 0]}")
print(f"Final Test Accuracy: {test_accuracy}")
