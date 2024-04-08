import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from ucimlrepo import fetch_ucirepo

# Fetch the dataset
print("Fetching dataset...")
adult = fetch_ucirepo(id=2)
print("Dataset fetched successfully.")
y = adult.data.targets
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]  # Using the first column as the target
y = y.apply(lambda x: 1 if x.strip() == '>50K' else 0)

X = adult.data.features

# Split the data
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

# Model definition
model = Sequential([
    Dense(64, input_shape=(X_train_prepared.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Model training
print("Training model...")
history = model.fit(X_train_prepared, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
print("Evaluating model...")
test_loss, test_accuracy = model.evaluate(X_test_prepared, y_test, verbose=1)
print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

# Print a human wistful goodbye and goodnight message
print("Thank you for interacting with me. I'm off to bed. Please come again soon, it's dark and lonely here.")