import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load dataset
dataset = pd.read_excel('fruit_data.xlsx')
print(dataset.head())
print(dataset.info())
print(f"Is dataset empty? {dataset.empty}")

# Encode categorical data
en = LabelEncoder()
dataset['name'] = en.fit_transform(dataset['name'])

# Feature and target variables
x = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].values   # Target

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
print(f"x_train = {len(x_train)}")
print(f"x_test = {len(x_test)}")
print(f"y_train = {len(y_train)}")
print(f"y_test = {len(y_test)}")

# Standardize features
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Initialize and train classifier
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predictions and evaluation
y_pred = classifier.predict(x_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification Report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save predictions to Excel
ydata = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
ydata.to_excel('data_Fruit_actualpred.xlsx', index=False)

# Save trained model using pickle
filename = 'NaiveBayes_Fruit.sav'
pickle.dump(classifier, open(filename, 'wb'))
