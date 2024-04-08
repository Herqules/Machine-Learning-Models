import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from ucimlrepo import fetch_ucirepo
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


print("Fetching dataset...")
adult = fetch_ucirepo(id=2)
print("Dataset fetched successfully.")

X = adult.data.features
y = adult.data.targets
print(f"Dataset shape: {X.shape[0]} rows and {X.shape[1]} columns. This indicates that we have data on {X.shape[0]} individuals, each described by 14 attributes, alongside their income classification.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("Data split into training and test sets to ensure a fair evaluation of the model's performance.")

X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=200, stratify=y_test, random_state=42)
print("Test set further refined to ensure it contains an equal representation of both income classes, facilitating a fair model evaluation.")

class_1_indices = y_train[y_train == ">50K"].index
class_2_indices = y_train[y_train == "<=50K"].index

np.random.seed(42)
selected_class_1_indices = np.random.choice(class_1_indices, 600, replace=False)  # Selecting a larger number of individuals with income >50K
selected_class_2_indices = np.random.choice(class_2_indices, 200, replace=False)  # Selecting fewer individuals with income <=50K

# Create the unbalanced training set
X_train_unbalanced = X_train.loc[np.concatenate([selected_class_1_indices, selected_class_2_indices])]
y_train_unbalanced = y_train.loc[np.concatenate([selected_class_1_indices, selected_class_2_indices])]
print("Training data unbalanced successfully, challenging the model to learn from an uneven distribution of data.")

categorical_cols = X_train_unbalanced.select_dtypes(include=['object', 'category']).columns
numerical_cols = X_train_unbalanced.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])


X_train_unbalanced_preprocessed = preprocessor.fit_transform(X_train_unbalanced)

X_test_preprocessed = preprocessor.transform(X_test)

# Define the neural network architecture
model = Sequential()
model.add(Dense(units=24, input_dim=X_train_unbalanced.shape[1], activation='sigmoid'))  # 24 nodes in the hidden layer
model.add(Dense(units=1, activation='sigmoid'))  # Output layer for binary classification

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
# history = model.fit(X_train_unbalanced_preprocessed, y_train_unbalanced, epochs=..., batch_size=...) # commented out, couldnt make it run in the Venv with data

print("\nFirst few rows of the unbalanced training features:")
print(X_train_unbalanced.head())
print("\nFirst few rows of the unbalanced training labels:")
print(y_train_unbalanced.head())

# line to space out the output
print("\n")

def introduce_artificial_errors(y, error_rate=0.05):
    np.random.seed(42)
    num_errors = int(len(y) * error_rate)
    error_indices = np.random.choice(y.index, size=num_errors, replace=False)
    
    y_errors = y.copy()
    for idx in error_indices:
        current_label = y_errors.loc[idx]
        new_label = "<=50K" if current_label == ">50K" else ">50K"
        y_errors.loc[idx] = new_label
    return y_errors
