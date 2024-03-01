import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        return None

def visualize_relationships(data):
    sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=5, aspect=0.7, kind='reg', plot_kws={'line_kws':{'color':'red'}}, diag_kind=None)
    plt.show()

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

def plot_predictions(y_test, y_pred):
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    perfect_line = np.linspace(min_val, max_val, 100)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, c='blue', alpha=0.5, label='Actual vs Predicted')
    plt.plot(perfect_line, perfect_line, color='red', linestyle='--', label='Perfect Prediction')
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title(f'Actual vs Predicted Sales (R^2 = {r2_score(y_test, y_pred):.2f})')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Load the dataset
    data = load_data('advertising.csv')
    if data is None:
        return

    # Display the first few rows of the dataset
    print("First few rows of the dataset:")
    print(data.head())

    # Basic statistics of the dataset
    print("\nBasic statistics of the dataset:")
    print(data.describe())

    # Check for missing values
    print("\nMissing values in the dataset:")
    print(data.isnull().sum())

    # Visualize the relationships between features and target variable (Sales)
    print("\nVisualizing the relationships between features and target variable:")
    visualize_relationships(data)

    # Splitting the data into train and test sets
    X = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the linear regression model
    model = train_model(X_train, y_train)

    # Evaluating the model
    mse, r2 = evaluate_model(model, X_test, y_test)
    print('\nModel Evaluation:')
    print('Mean Squared Error:', mse)
    print('R^2 Score:', r2)

    # Plotting predicted vs actual values
    plot_predictions(y_test, model.predict(X_test))

if __name__ == "__main__":
    main()
