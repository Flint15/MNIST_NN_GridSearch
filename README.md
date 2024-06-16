# Getting started

## Prerequisites

* scikit-learn

## Instalation

`pip install scikit-learn`

## Usage

### Import necessary libraries
```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
```

`from sklearn.datasets import load_digits` Import MNIST dataset

`from sklearn.model_selection import train_test_split, GridSearchCV` Import train_test_split and GridSearchCV function

`from sklearn.neural_network import MLPClassifier` Imprt Multy-Layer Perceptron Classifier

`from sklearn.metrics import accuracy_score, classification_report` Import metrics to evaluate model performance

### Load, assignment and split

```python
digits = load_digits()

X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=1)
```

### Initialize a classifier

```python
clf = MLPClassifier(max_iter=1000, random_state=1)
```

### Define the parameter grid

```python
param_grid = {
    'hidden_layer_sizes': [100, 150, 200, (100, 50), (100, 100)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'alpha': [0.0001, 0.001, 0.1]
}
```

### Initialize GridSearchCV

```python
grid_search = GridSearchCV(clf, param_grid, cv=5)
```

### Fit the model

```python
grid_search.fit(X_train, y_train)
```

### Get the best model and make a prediction

```python
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
```

### Evaluate the performance

```python
accuracy = accuracy_score(y_test, y_pred)

report = classification_report(y_test, y_pred, target_names=[str(name) for name in digits.target_names])
```
