import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from ucimlrepo import fetch_ucirepo

# Download the dataset from the UCI ML Repository (in our case, the Adult dataset)
print("Fetching dataset...")
adult = fetch_ucirepo(id=2)
print("Dataset fetched successfully.")
y = adult.data.targets
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]  # Using the first column as the target
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
    sparse_threshold=0  # Ensures the output is a dense array
)

X_train_prepared = preprocessor.fit_transform(X_train)
X_test_prepared = preprocessor.transform(X_test)


d = X_train_prepared.shape[1]  # Dimensionality of the input
population_size = 4 + 1 * d  # (4+d) ES variant with last digit of SUID modulo 4: 1 modulo 4 = 1

# Initialize the population: array of shape (population_size, d)
population = np.random.randn(population_size, d)
mutation_rate = 1 / (d ** 0.5)

def sigmoid(x):
    clip_x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-clip_x))

def fitness_function(weights):
    try:
        # Stable sigmoid prediction
        predictions = sigmoid(X_train_prepared.dot(weights))
        predictions = np.round(predictions)
        return mean_squared_error(y_train, predictions)
    except FloatingPointError:
        # Return a large number to represent a bad fitness score in case of numerical issues
        return np.inf



for generation in range(257):
    # Recombination and mutation
    offspring = []
    for _ in range(population_size):
        parents = population[np.random.choice(population_size, 2, replace=False)]
        child = np.mean(parents, axis=0)  # Arithmetic recombination
        child += np.random.randn(d) * mutation_rate  # Mutation
        offspring.append(child)
    offspring = np.array(offspring)
    
    # Fitness evaluation
    fitness_scores = np.array([fitness_function(ind) for ind in offspring])
    
    # Selection
    best_indices = np.argsort(fitness_scores)[:population_size]  # Select the best individuals
    population = offspring[best_indices]
    
    # Mutation rate adaptation using the 1/5 rule
    if generation % 5 == 0:
        success_rate = np.mean(fitness_scores < np.median(fitness_scores))
        if success_rate > 0.2:
            mutation_rate /= 0.85
        elif success_rate < 0.2:
            mutation_rate *= 0.85
    
    # Record and report statistics if needed
    if generation in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        print(f"Generation {generation}: Best fitness score: {fitness_scores[best_indices[0]]}")

# Evaluate the best individual from the last generation
best_weights = population[0]
predictions = sigmoid(X_test_prepared.dot(best_weights))  # Use the stable sigmoid function
predictions = np.round(predictions)
test_score = mean_squared_error(y_test, predictions)
print(f"Overall fitness score (MSE): {test_score}")
