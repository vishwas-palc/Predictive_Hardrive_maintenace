import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils import resample

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import OneClassSVM
import joblib


# Get a list of all CSV files in the directory
file_paths = glob.glob('/train data/train data_2022/*.csv')

# Initialize an empty list to store DataFrames
dfs = []


# Load and concatenate all CSV files
for file_path in file_paths:
    df = pd.read_csv(file_path)
    dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)
# Impute missing values using mean for numeric columns
combined_df.fillna(-1, inplace=True)

# Print the first few rows of the DataFrame after imputation
combined_df.tail()
selected_columns = [
    "smart_198_raw",
    "smart_197_raw",
    "smart_187_raw",

    "smart_5_raw",
        "failure"
]
"""
selected_columns = [
    "smart_5_raw",
    "smart_197_raw",
        "failure"
]
"""
df_failure = combined_df[selected_columns]
X = df_failure.drop("failure", axis=1)

y = df_failure["failure"]



# Split into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load or train the One-Class SVM model
#load_saved_model = False  # Change this to True if you want to load a saved model

"""if load_saved_model:
    loaded_model = joblib.load('/content/sample_data/trained model/one_class_svm_model_2021.pkl')
else:"""
one_class_svm_model = OneClassSVM(nu=0.009)
one_class_svm_model.fit(X_train)
joblib.dump(one_class_svm_model, '/trained model/one_class_svm_model.pkl')  # Save the trained model

"""# Predict anomalies on the test data
if load_saved_model:
    anomaly_predictions = loaded_model.predict(X_test)
else:
    anomaly_predictions = one_class_svm_model.predict(X_test)

# Convert the predictions to binary format for easy comparison with the ground truth
binary_predictions = [1 if pred == -1 else 0 for pred in anomaly_predictions]

# Calculate metrics for anomaly detection
accuracy = accuracy_score(y_test, binary_predictions)
precision = precision_score(y_test, binary_predictions)
recall = recall_score(y_test, binary_predictions)
f1 = f1_score(y_test, binary_predictions)

print("Anomaly Detection Accuracy:", accuracy)
print("Anomaly Detection Precision:", precision)
print("Anomaly Detection Recall:", recall)
print("Anomaly Detection F1-score:", f1)"""
