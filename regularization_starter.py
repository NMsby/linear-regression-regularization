import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ONLY use pandas for initial data loading
# All other operations should use NumPy

# Part 1: Data Preparation
def load_and_preprocess_data(file_path):
    """
    Load the dataset and perform preprocessing

    Args:
        file_path: Path to the dataset CSV file

    Returns:
        X_train: Training features
        y_train: Training target values
        X_test: Testing features
        y_test: Testing target values
    """
    # Load data using pandas
    df = pd.read_csv(file_path)

    # Select features for prediction (select at least 5 relevant features)
    # For example: bedrooms, bathrooms, sqft_living, floors, condition, garde, etc.
    features = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'grade', 'sqft_above', 'view']
    X = df[features].values
    y = df['price'].values

    # Handle any missing values (if necessary)
    # Check for NaN values
    if np.isnan(X).any():
        # Replace NaNs with feature means
        for col in range(X.shape[1]):
            mask = np.isnan(X[:, col])
            X[mask, col] = np.mean(X[~mask, col])

    # Normalize/standardize features
    # Compute mean and standard deviation for each feature
    feature_means = np.mean(X, axis=0)
    feature_stds = np.std(X, axis=0)

    # Standardize features (z-score normalization)
    X_normalized = (X - feature_means) / feature_stds

    # Split data into training (80%) and testing (20%) sets
    # Set random seed for reproducibility
    np.random.seed(42)

    # Shuffle data (create random indices)
    indices = np.random.permutation(X.shape[0])
    split_idx = int(X.shape[0] * 0.8)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    # Split the data
    X_train = X_normalized[train_indices]
    y_train = y[train_indices]
    X_test = X_normalized[test_indices]
    y_test = y[test_indices]

    return X_train, y_train, X_test, y_test


# Part 2: Basic Linear Regression with Gradient Descent
def predict(X, weights, bias):
    """
    Make predictions using the linear model: y = X*w + b

    Args:
        X: Features
        weights: Model weights
        bias: Model bias

    Returns:
        Predicted values
    """
    return np.dot(X, weights) + bias


def compute_cost(X, y, weights, bias):
    """
    Compute the Mean Squared Error cost function

    Args:
        X: Features
        y: Target values
        weights: Model weights
        bias: Model bias

    Returns:
        Mean Squared Error
    """
    m = X.shape[0]  # Number of samples

    # Calculate predictions
    y_pred = predict(X, weights, bias)

    # Calculate the error (difference between predictions and actual values)
    error = y_pred - y

    # Calculate MSE: J(w) = (1/2m) * Σ(y_pred - y_actual)²
    cost = (1 / (2 * m)) * np.sum(error ** 2)

    return cost


def gradient_descent(X, y, learning_rate, num_iterations):
    """
    Implement gradient descent algorithm for linear regression

    Args:
        X: Features
        y: Target values
        learning_rate: Learning rate alpha
        num_iterations: Number of iterations to run

    Returns:
        weights: Optimized weights
        bias: Optimized bias
        cost_history: History of cost values during optimization
        weights_history: History of weights during optimization
        bias_history: History of bias during optimization
    """
    # Initialize parameters
    m, n = X.shape  # m = number of samples, n = number of features
    weights = np.zeros(n)
    bias = 0

    # Arrays to store history
    cost_history = np.zeros(num_iterations)
    weights_history = np.zeros((num_iterations, n))
    bias_history = np.zeros(num_iterations)

    # Implement gradient descent algorithm
    for i in range(num_iterations):
        # Calculate predictions
        y_pred = predict(X, weights, bias)

        # Calculate errors
        error = y_pred - y

        # Calculate gradients
        # ∂J/∂w = (1/m) * X^T * (y_pred - y)
        dw = (1 / m) * np.dot(X.T, error)
        # ∂J/∂b = (1/m) * Σ(y_pred - y)
        db = (1 / m) * np.sum(error)

        # Update parameters
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db

        # Store parameters and cost
        weights_history[i] = weights
        bias_history[i] = bias
        cost_history[i] = compute_cost(X, y, weights, bias)

    return weights, bias, cost_history, weights_history, bias_history


# Part 3: RIDGE Regression (L2 Regularization)
def compute_cost_ridge(X, y, weights, bias, lambda_param):
    """
    Compute the Mean Squared Error cost function with L2 regularization

    Args:
        X: Features
        y: Target values
        weights: Model weights
        bias: Model bias
        lambda_param: Regularization parameter

    Returns:
        Mean Squared Error with L2 regularization
    """
    m = X.shape[0]  # Number of samples

    # Calculate predictions
    y_pred = predict(X, weights, bias)

    # Calculate the error (difference between predictions and actual values)
    error = y_pred - y

    # Calculate MSE with L2 regularization:
    # J(w) = (1/2m) * Σ(y_pred - y_actual)² + (λ/2m) * Σw²
    cost = (1 / (2 * m)) * np.sum(error ** 2) + (lambda_param / (2 * m)) * np.sum(weights ** 2)

    return cost


def gradient_descent_ridge(X, y, learning_rate, num_iterations, lambda_param):
    """
    Implement gradient descent algorithm for RIDGE regression

    Args:
        X: Features
        y: Target values
        learning_rate: Learning rate alpha
        num_iterations: Number of iterations to run
        lambda_param: Regularization parameter

    Returns:
        weights: Optimized weights
        bias: Optimized bias
        cost_history: History of cost values during optimization
        weights_history: History of weights during optimization
        bias_history: History of bias during optimization
    """
    # Initialize parameters
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0

    # Arrays to store history
    cost_history = np.zeros(num_iterations)
    weights_history = np.zeros((num_iterations, n))
    bias_history = np.zeros(num_iterations)

    # Implement gradient descent algorithm with RIDGE regularization
    for i in range(num_iterations):
        # Calculate predictions
        y_pred = predict(X, weights, bias)

        # Calculate errors
        error = y_pred - y

        # Calculate gradients with L2 regularization term
        # ∂J/∂w = (1/m) * X^T * (y_pred - y) + (λ/m) * w
        dw = (1 / m) * np.dot(X.T, error) + (lambda_param / m) * weights

        # Bias is not regularized
        # ∂J/∂b = (1/m) * Σ(y_pred - y)
        db = (1 / m) * np.sum(error)

        # Update parameters
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db

        # Store parameters and cost
        weights_history[i] = weights
        bias_history[i] = bias
        cost_history[i] = compute_cost_ridge(X, y, weights, bias, lambda_param)

    return weights, bias, cost_history, weights_history, bias_history


# Part 4: LASSO Regression (L1 Regularization)
def compute_cost_lasso(X, y, weights, bias, lambda_param):
    """
    Compute the Mean Squared Error cost function with L1 regularization

    Args:
        X: Features
        y: Target values
        weights: Model weights
        bias: Model bias
        lambda_param: Regularization parameter

    Returns:
        Mean Squared Error with L1 regularization
    """
    m = X.shape[0]  # Number of samples

    # Calculate predictions
    y_pred = predict(X, weights, bias)

    # Calculate the error (difference between predictions and actual values)
    error = y_pred - y

    # Calculate MSE with L1 regularization:
    # J(w) = (1/2m) * Σ(y_pred - y_actual)² + (λ/m) * Σ|w|
    cost = (1 / (2 * m)) * np.sum(error ** 2) + (lambda_param / m) * np.sum(np.abs(weights))

    return cost


def gradient_descent_lasso(X, y, learning_rate, num_iterations, lambda_param):
    """
    Implement gradient descent algorithm for LASSO regression

    Args:
        X: Features
        y: Target values
        learning_rate: Learning rate alpha
        num_iterations: Number of iterations to run
        lambda_param: Regularization parameter

    Returns:
        weights: Optimized weights
        bias: Optimized bias
        cost_history: History of cost values during optimization
        weights_history: History of weights during optimization
        bias_history: History of bias during optimization
    """
    # Initialize parameters
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0

    # Arrays to store history
    cost_history = np.zeros(num_iterations)
    weights_history = np.zeros((num_iterations, n))
    bias_history = np.zeros(num_iterations)

    # Implement gradient descent algorithm with LASSO regularization
    for i in range(num_iterations):
        # Calculate predictions
        y_pred = predict(X, weights, bias)

        # Calculate errors
        error = y_pred - y

        # Calculate gradient of MSE part
        dw_mse = (1 / m) * np.dot(X.T, error)

        # Calculate LASSO regularization term (subgradient)
        # ∂J/∂w = (1/m) * X^T * (y_pred - y) + (λ/m) * sign(w)
        # sign(w) is +1 for w > 0, -1 for w < 0, and 0 for w = 0
        dw_lasso = (lambda_param / m) * np.sign(weights)

        # Combine MSE gradient and LASSO regularization
        dw = dw_mse + dw_lasso

        # Bias is not regularized
        # ∂J/∂b = (1/m) * Σ(y_pred - y)
        db = (1 / m) * np.sum(error)

        # Update parameters
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db

        # Store parameters and cost
        weights_history[i] = weights
        bias_history[i] = bias
        cost_history[i] = compute_cost_lasso(X, y, weights, bias, lambda_param)

    return weights, bias, cost_history, weights_history, bias_history


# Part 5: Visualization and Analysis Functions
def plot_cost_history(cost_history, title):
    """
    Plot the cost history over iterations

    Args:
        cost_history: History of cost values
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cost_history)), cost_history, linewidth=2)
    plt.title(title, fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cost (MSE)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Highlighting initial and final costs
    plt.annotate(f'Initial cost: {cost_history[0]:.2f}',
                 xy=(0, cost_history[0]),
                 xytext=(50, 20),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle='->'))

    plt.annotate(f'Final cost: {cost_history[-1]:.2f}',
                 xy=(len(cost_history) - 1, cost_history[-1]),
                 xytext=(-50, -20),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle='->'))

    plt.tight_layout()
    plt.show()


def plot_coefficients(feature_names, basic_weights, ridge_weights, lasso_weights):
    """
    Plot the coefficients from different models for comparison

    Args:
        feature_names: Names of the features
        basic_weights: Weights from basic linear regression
        ridge_weights: Weights from RIDGE regression
        lasso_weights: Weights from LASSO regression
    """
    plt.figure(figsize=(14, 8))
    x = np.arange(len(feature_names))
    width = 0.25

    # Create the bars
    plt.bar(x - width, basic_weights, width, label='Basic', color='blue', alpha=0.7)
    plt.bar(x, ridge_weights, width, label='RIDGE', color='green', alpha=0.7)
    plt.bar(x + width, lasso_weights, width, label='LASSO', color='red', alpha=0.7)

    # Customize the plot
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Coefficient Value', fontsize=12)
    plt.title('Comparison of Model Coefficients', fontsize=14)
    plt.xticks(x, feature_names, rotation=45, ha='right', fontsize=10)
    plt.legend(fontsize=10)

    # Add a horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Add grid lines for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


def plot_predictions(X_test, y_test, basic_weights, basic_bias,
                     ridge_weights, ridge_bias, lasso_weights, lasso_bias):
    """
    Plot the predictions from different models against actual values

    Args:
        X_test: Test features
        y_test: Test target values
        basic_weights, basic_bias: Parameters for basic linear regression
        ridge_weights, ridge_bias: Parameters for RIDGE regression
        lasso_weights, lasso_bias: Parameters for LASSO regression
    """
    # Generate predictions for all models
    basic_pred = predict(X_test, basic_weights, basic_bias)
    ridge_pred = predict(X_test, ridge_weights, ridge_bias)
    lasso_pred = predict(X_test, lasso_weights, lasso_bias)

    # Sample a subset of test data for clearer visualization (50 points)
    sample_size = 50
    indices = np.random.choice(len(y_test), sample_size, replace=False)

    # Sort indices by actual values for better visualization
    indices = indices[np.argsort(y_test[indices])]

    # Get the sampled data
    actual = y_test[indices]
    basic_sampled = basic_pred[indices]
    ridge_sampled = ridge_pred[indices]
    lasso_sampled = lasso_pred[indices]

    # Create the plot
    plt.figure(figsize=(14, 8))

    # Plot actual values
    plt.plot(range(sample_size), actual, 'o-', label='Actual', linewidth=2, color='black')

    # Plot predictions
    plt.plot(range(sample_size), basic_sampled, 's--', label='Basic Linear Regression', alpha=0.7)
    plt.plot(range(sample_size), ridge_sampled, '^--', label='RIDGE Regression', alpha=0.7)
    plt.plot(range(sample_size), lasso_sampled, 'd--', label='LASSO Regression', alpha=0.7)

    plt.title('Model Predictions vs Actual Values', fontsize=14)
    plt.xlabel('Sample Index (sorted by actual value)', fontsize=12)
    plt.ylabel('House Price', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_model_comparison(basic_mse, ridge_mse, lasso_mse):
    """
    Plot a comparison of MSE values from different models

    Args:
        basic_mse: MSE from basic linear regression
        ridge_mse: MSE from RIDGE regression
        lasso_mse: MSE from LASSO regression
    """
    models = ['Basic Linear\nRegression', 'RIDGE\nRegression', 'LASSO\nRegression']
    mse_values = [basic_mse, ridge_mse, lasso_mse]

    # Create color map based on performance (lower is better)
    colors = ['red', 'orange', 'green']
    color_indices = np.argsort(np.argsort(mse_values))
    bar_colors = [colors[idx] for idx in color_indices]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, mse_values, color=bar_colors, alpha=0.7)

    # Add value labels on top of each bar
    for bar, value in zip(bars, mse_values):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 value + max(mse_values) * 0.01,
                 f'{value:.2f}',
                 ha='center', va='bottom',
                 fontweight='bold')

    plt.title('Mean Squared Error Comparison', fontsize=14)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def evaluate_model(X, y, weights, bias, model_name):
    """
    Evaluate the model on the provided data

    Args:
        X: Features
        y: Target values
        weights: Model weights
        bias: Model bias
        model_name: Name of the model for printing

    Returns:
        Mean Squared Error
    """
    y_pred = predict(X, weights, bias)
    mse = np.mean((y_pred - y) ** 2)
    print(f"{model_name} MSE: {mse:.4f}")
    return mse


# Main execution
if __name__ == "__main__":
    # File path to the dataset
    file_path = "kc_house_data.csv"  # Update with the correct path

    print("Loading and preprocessing data...")
    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_and_preprocess_data(file_path)

    # Get feature names for later visualization
    # Matches features selected in load_and_preprocess_data
    feature_names = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'grade', 'sqft_above', 'view']

    # Hyperparameters
    learning_rate = 0.01
    num_iterations = 1000
    ridge_lambda = 0.1
    lasso_lambda = 0.01

    print("Training Basic Linear Regression model...")
    basic_weights, basic_bias, basic_cost_history, _, _ = gradient_descent(
        X_train, y_train, learning_rate, num_iterations
    )

    print("Training RIDGE Regression model...")
    ridge_weights, ridge_bias, ridge_cost_history, _, _ = gradient_descent_ridge(
        X_train, y_train, learning_rate, num_iterations, ridge_lambda
    )

    print("Training LASSO Regression model...")
    lasso_weights, lasso_bias, lasso_cost_history, _, _ = gradient_descent_lasso(
        X_train, y_train, learning_rate, num_iterations, lasso_lambda
    )

    # Evaluate models on the test set
    print("\nEvaluation on Test Set:")
    basic_mse = evaluate_model(X_test, y_test, basic_weights, basic_bias, "Basic Linear Regression")
    ridge_mse = evaluate_model(X_test, y_test, ridge_weights, ridge_bias, "RIDGE Regression")
    lasso_mse = evaluate_model(X_test, y_test, lasso_weights, lasso_bias, "LASSO Regression")

    # Plot cost history
    plot_cost_history(basic_cost_history, "Basic Linear Regression Cost History")
    plot_cost_history(ridge_cost_history, "RIDGE Regression Cost History")
    plot_cost_history(lasso_cost_history, "LASSO Regression Cost History")

    # Plot coefficients for comparison
    plot_coefficients(feature_names, basic_weights, ridge_weights, lasso_weights)

    # Plot predictions comparison
    plot_predictions(X_test, y_test, basic_weights, basic_bias,
                     ridge_weights, ridge_bias, lasso_weights, lasso_bias)

    # Plot MSE comparison
    plot_model_comparison(basic_mse, ridge_mse, lasso_mse)

    # Check feature selection by LASSO
    print("\nFeature Selection by LASSO:")
    for i, (name, coef) in enumerate(zip(feature_names, lasso_weights)):
        if abs(coef) > 1e-10:  # Non-zero coefficients
            print(f"Feature {name}: {coef:.6f}")
        else:
            print(f"Feature {name}: 0 (eliminated)")
