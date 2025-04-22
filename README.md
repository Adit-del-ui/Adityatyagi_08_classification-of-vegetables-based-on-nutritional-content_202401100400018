# Adityatyagi_08_classification-of-vegetables-based-on-nutritional-content_202401100400018
 Vegetable Classification by Nutritional Content

[Python](https://img.shields.io/badge/Python-3.8%2B-blue)
[Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0.2-orange)
[Pandas](https://img.shields.io/badge/Pandas-1.4.0-red)

A machine learning project that classifies vegetables into categories (root, fruit, leafy) based on their nutritional content using Random Forest.

## Dataset
`vegetables.csv` contains:
- 3 nutritional features:
  - `vitamin_a` (percentage)
  - `vitamin_c` (percentage)
  - `fiber` (grams)
- Target variable:
  - `type` (root/fruit/leafy)

## Requirements
Python 3.8+
pandas>=1.4.0
scikit-learn>=1.0.2
matplotlib>=3.5.0
seaborn>=0.11.2


## Project Structure
vegetable-classification/
├── vegetables.csv # Dataset
├── vegetable_classifier.ipynb # Jupyter notebook
└── README.md


## Implementation Steps

1. **Data Loading & Exploration**
   - Check for missing values
   - Analyze class distribution

2. **Data Preparation**
   - Separate features (nutritional values) and labels (vegetable type)
   - Split into train/test sets (80/20 ratio)

3. **Model Training**
   - Random Forest Classifier with 100 trees
   - Default hyperparameters

4. **Evaluation**
   - Accuracy metric
   - Classification report (precision/recall/F1)
   - Confusion matrix visualization

5. **Feature Analysis**
   - Importance ranking of nutritional features

## Results
- Achieved accuracy: XX.XX% (replace with your actual result)
- Most important feature: `vitamin_c` (replace if different)
- Confusion matrix shows best performance on [class] classification
CODE:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from Drive
df = pd.read_csv('/content/drive/MyDrive/vegetables.csv')

# Show first few rows
print("First 5 rows:")
print(df.head())

# Check if there are missing values
print("\nMissing values:")
print(df.isnull().sum())

# Show class distribution
print("\nVegetable Types Distribution:")
print(df['type'].value_counts())




# Features (X) and Labels (y)
X = df.drop('type', axis=1)
y = df['type']

# Split into Train and Test sets (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc*100:.2f}%")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
![image](https://github.com/user-attachments/assets/9bf3149c-7a8a-4682-8bb9-efd6d8d2861a)
# Feature Importance Plot
importances = model.feature_importances_
feature_names = X.columns

# Create DataFrame
feat_importance = pd.DataFrame({'Features': feature_names, 'Importance': importances})
feat_importance = feat_importance.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Features', data=feat_importance, palette="viridis")
plt.title('Feature Importance')
plt.show()


![image](https://github.com/user-attachments/assets/fd639c61-1031-49a7-92a7-9f958d108dd6)

