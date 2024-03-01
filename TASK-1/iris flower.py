import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
iris_df = pd.read_csv('IRIS.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(iris_df.head())

# Summary statistics of the dataset
print("\nSummary statistics of the dataset:")
print(iris_df.describe())

# Pairplot to visualize relationships between features
print("\nPairplot to visualize relationships between features:")
# Adjusting the pairplot to include more details
sns.pairplot(iris_df, hue='species', markers=['o', 's', 'D'], palette='husl')
plt.suptitle('Pairplot of Iris Dataset', y=1.02)
plt.show()

# Splitting the dataset into features (X) and labels (y)
X = iris_df.drop('species', axis=1)
y = iris_df['species']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
