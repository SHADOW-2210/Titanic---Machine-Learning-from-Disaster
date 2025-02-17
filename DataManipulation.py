import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
try:
    titanic = pd.read_csv("train.csv")
except FileNotFoundError:
    print("Error: train.csv not found. Please download the dataset and place it in the same directory.")
    exit()

# Data Exploration and Cleaning
print(titanic.head())  # Display the first few rows
print(titanic.info())  # Get information about the data types and missing values
print(titanic.describe()) # Descriptive statistics

# Handle missing values (example: fill NaN in 'Age' with the median age)
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace = True)

# Feature Engineering (example: create a 'FamilySize' feature)
titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch'] + 1

# Data Analysis and Visualization

# Survival rate by gender
sns.countplot(x='Survived', hue='Sex', data=titanic)
plt.title('Survival Rate by Gender')
plt.show()

# Survival rate by passenger class
sns.countplot(x='Survived', hue='Pclass', data=titanic)
plt.title('Survival Rate by Passenger Class')
plt.show()

# Age distribution
sns.histplot(titanic['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

# Survival rate by family size
sns.countplot(x='Survived', hue='FamilySize', data=titanic)
plt.title('Survival Rate by Family Size')
plt.show()

# Correlation matrix
correlation_matrix = titanic.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# Grouping and Aggregation (example: average age and survival rate by passenger class)
grouped_data = titanic.groupby('Pclass').agg({'Age': 'mean', 'Survived': 'mean'})
print("\nAverage Age and Survival Rate by Passenger Class:")
print(grouped_data)

# Further analysis and visualization can be added here...

# Save cleaned data (optional)
titanic.to_csv("cleaned_titanic.csv", index=False)