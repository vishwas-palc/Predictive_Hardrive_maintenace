import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pymongo
import json
from bson import ObjectId



# MongoDB server connection parameters
host = '172.27.1.162'  # Change this to your MongoDB server host
port = 27017        # Change this to your MongoDB server port
database_name = 'SMART'
collection_name = 'sda'
username = 'mongoadmin'
password = 'bdung'

# Connect to the MongoDB server with authentication
client = pymongo.MongoClient(host, port, username=username, password=password)

# Print information about the MongoDB connection
print(f'Connected to MongoDB server at {host}:{port} as user {username}')

# Access the specified database and collection
db = client[database_name]
collection = db[collection_name]

# Fetch data from the collection and convert ObjectId to strings
data = list(collection.find())
for document in data:
    document['_id'] = str(document['_id'])

# Check the document count in the collection
document_count = collection.count_documents({})  # Count all documents
print(f'Total documents in collection: {document_count}')

# Close the MongoDB connection
client.close()

one_class_svm_model = joblib.load('trained model/one_class_svm_model.pkl')

# Create the directory if it doesn't exist
output_directory = 'testing results/2021/'
os.makedirs(output_directory, exist_ok=True)

# Add the "failure" column to the data
for document in data:
    document['failure'] = 1

# Create a DataFrame from the modified data
df_year_latest = pd.json_normalize(data)
selected_latest_year = [
    "smart_198_raw",
    "smart_197_raw",
    "smart_187_raw",
    "smart_5_raw"
]
year = 2023
df_year_latest.fillna(-1, inplace=True)
df_year_failure = df_year_latest[selected_latest_year]
x_year = df_year_failure

anomaly_predictions_year = one_class_svm_model.predict(x_year)

# Create a DataFrame containing only '_id' and 'predicted_failure' columns
result_df = pd.DataFrame({'_id': df_year_latest['_id'], 'predicted_failure': anomaly_predictions_year})

# Rename the 'predicted_failure' column values
result_df['predicted_failure'] = result_df['predicted_failure'].apply(lambda x: 'not critical' if x == 1 else 'critical')

# Save the result DataFrame as a CSV file (optional)
result_csv_file = os.path.join(output_directory, f'predicted_failure_{year}.csv')
result_df.to_csv(result_csv_file, index=False)

# Show the result DataFrame
print(result_df)

# Create a plot to visualize the 'predicted_failure' column
plt.figure(figsize=(8, 6))
sns.countplot(x='predicted_failure', data=result_df, palette='viridis')
plt.xlabel('Predicted Failure')
plt.ylabel('Count')
plt.title(f'Predicted Failures in {year}')
plt.show()


"""years = list(range(2013, 2024))  # Update the range according to your years

# Load the saved One-Class SVM model
one_class_svm_model = joblib.load('trained model/one_class_svm_model.pkl')

# Create the directory if it doesn't exist
output_directory = 'testing results/2021/'
os.makedirs(output_directory, exist_ok=True)

for year in years:
    file_paths = glob.glob(f'test data/test data_{year}/*.csv')

    dfs_year = []

    # Load and concatenate all CSV files for the current year
    for file_path in file_paths:
        df_year = pd.read_csv(file_path)
        dfs_year.append(df_year)

    df_year_latest = pd.concat(dfs_year, ignore_index=True)

    selected_latest_year = [
        "smart_198_raw",
        "smart_197_raw",
        "smart_187_raw",
        "smart_5_raw"
    ]

    df_year_latest.fillna(-1, inplace=True)
    df_year_failure = df_year_latest[selected_latest_year]
    x_year = df_year_failure

    anomaly_predictions_year = one_class_svm_model.predict(x_year)

    df_year_latest['predicted_failure'] = [1 if pred == -1 else 0 for pred in anomaly_predictions_year]
    y_year_prediction = df_year_latest['predicted_failure']
    y_test=df_year_latest['failure']
    accuracy = accuracy_score(y_test, y_year_prediction)
    precision = precision_score(y_test, y_year_prediction)
    recall = recall_score(y_test, y_year_prediction)
    f1 = f1_score(y_test, y_year_prediction)
    
    print(f"Anomaly Detection of year {year} Accuracy:", accuracy)
    print(f"Anomaly Detection of year {year} Precision:", precision)
    print(f"Anomaly Detection of year {year} Recall:", recall)
    print(f"Anomaly Detection of year {year} F1-score:", f1)

    pivot_table = df_year_latest.pivot_table(index='failure', columns='predicted_failure', aggfunc='size', fill_value=0)
    sns.set_palette("viridis")

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt="d", cmap="viridis")

    plt.xlabel('Predicted Column')
    plt.ylabel('Target Column')
    plt.title(f'Heat Map of Target {year} Column vs. Predicted Column')

    # Save the plot as an SVG file
    svg_filename = os.path.join(output_directory, f'heatmap_{year}.svg')
    plt.savefig(svg_filename, format='svg', bbox_inches='tight')

    plt.show()"""