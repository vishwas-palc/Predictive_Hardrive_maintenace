import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
years = list(range(2013, 2024))  # Update the range according to your years

# Load the saved One-Class SVM model
one_class_svm_model = joblib.load('/trained model/one_class_svm_model.pkl')

# Create the directory if it doesn't exist
output_directory = '/testing results/2016/'
os.makedirs(output_directory, exist_ok=True)

for year in years:
    file_paths = glob.glob(f'/test data/test data_{year}/*.csv')

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
    
    print("Anomaly Detection Accuracy:", accuracy)
    print("Anomaly Detection Precision:", precision)
    print("Anomaly Detection Recall:", recall)
    print("Anomaly Detection F1-score:", f1)

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

    plt.show()
