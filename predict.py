import numpy as np
import csv
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
import numpy as np
from sklearn.svm import OneClassSVM
# Load the .npy file
npy_file_path = 'tain_data_array.npy'
data = np.load(npy_file_path)

# Define the CSV file path
csv_file_path = '/data/train/tain_data_array.csv'

# Write the data to the CSV file
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(data)

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)
df_failure=df.fillna(-1, inplace=True)
X = df_failure.drop("failure", axis=1)

y = df_failure["failure"]



# Split into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the One-Class SVM model
#one_class_svm_model = OneClassSVM(nu=0.0006)  # Adjust nu parameter as needed
one_class_svm_model = OneClassSVM(nu=0.00007)
one_class_svm_model.fit(X_train)

# Predict anomalies on the test data
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
print("Anomaly Detection F1-score:", f1)
