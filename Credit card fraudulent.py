import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Load the dataset
data = pd.read_csv(r'C:\Users\KIIT\Downloads\creditcard.csv.zip')

# Explore the dataset
print(data.head())
print(data.info())
print(data.describe())

# Check for class imbalance
print(data['Class'].value_counts())
sns.countplot(x='Class', data=data)
plt.title('Class Distribution')
plt.show()

# Separate features and target variable
X = data.drop('Class', axis=1)
y = data['Class']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance using SMOTE
oversample = SMOTE()
X_resampled, y_resampled = oversample.fit_resample(X_scaled, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train a logistic regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Train a random forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Evaluate the logistic regression model
lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_report = classification_report(y_test, lr_predictions, target_names=['Genuine', 'Fraudulent'])

print("Logistic Regression Model")
print(f'Accuracy: {lr_accuracy}')
print('Classification Report:')
print(lr_report)

# Evaluate the random forest classifier
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_report = classification_report(y_test, rf_predictions, target_names=['Genuine', 'Fraudulent'])

print("Random Forest Classifier")
print(f'Accuracy: {rf_accuracy}')
print('Classification Report:')
print(rf_report)

# Plot confusion matrix for random forest
rf_conf_matrix = confusion_matrix(y_test, rf_predictions)
sns.heatmap(rf_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Genuine', 'Fraudulent'], yticklabels=['Genuine', 'Fraudulent'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')
plt.show()
