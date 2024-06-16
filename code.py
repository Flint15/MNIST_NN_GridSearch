# Import necessary libraries
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load digits
digits = load_digits()

# Features and labels
X = digits.data # Input features (image data)
y = digits.target # Input labels (digits classes)

# Split data into training and testing sets
# Here, we use 20% of the data for trainnig and the rest for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=1)

# Initialize Multy-Layer Perceptron classifier with a maximum of 1000 iteration
clf = MLPClassifier(max_iter=1000, random_state=1)

#Define param_grid to search over for hyperparameter tuning
param_grid = {
    'hidden_layer_sizes': [100, 150, 200, (100, 50), (100, 100)], # Various configuration for hidden layers
    'activation': ['identity', 'logistic', 'tanh', 'relu'],       # Different activation function to try
    'alpha': [0.0001, 0.001, 0.1]                                 # Regularization parameter
}

# Initialize GridSearchCV, with 5-fold cross-validation
grid_search = GridSearchCV(clf, param_grid, cv=5)

# Fit the model with hyperparameter tuning
grid_search.fit(X_train, y_train)

# Get the model with best parameters, founds by Grid search
best_model = grid_search.best_estimator_

# Make prediction on test data
y_pred = best_model.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate a detailed classification report
report = classification_report(y_test, y_pred, target_names=[str(name) for name in digits.target_names])

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
