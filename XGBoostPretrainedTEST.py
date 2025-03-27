import wfdb
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report

# Load the saved model
best_model = joblib.load("xgboost_stress_model.pkl")
print("Model loaded successfully!")

# Define file path
path = r'C:\Users\DJOA0\OneDrive\Desktop\SpO2HR'

# Select new test subjects (Subjects 17-20, for example)
new_test_subjects = range(16, 21)
new_test_data = []

# Load new test data
for i in new_test_subjects:
    record_name = f"Subject{i}_SpO2HR"
    record = wfdb.rdrecord(os.path.join(path, record_name))
    
    df = pd.DataFrame(record.p_signal, columns=record.sig_name)
    new_test_data.append(df)

# Combine new test data
new_test_df = pd.concat(new_test_data, ignore_index=True)

# Define stress classification function
def stress_label(hr, spo2):
    if hr > 110 and spo2 < 95:
        return 2  # High Stress
    elif 100 <= hr <= 110 and 95 <= spo2 <= 98:
        return 1  # Moderate Stress
    else:
        return 0  # Resting state

# Apply stress labels to the new test data
new_test_df['stress'] = [stress_label(hr, spo2) for hr, spo2 in zip(new_test_df['hr'], new_test_df['SpO2'])]

# Extract features
def extract_features(df):
    df['HR_std'] = df['hr'].rolling(window=5).std().fillna(0)
    df['SpO2_var'] = df['SpO2'].rolling(window=5).var().fillna(0)
    df['HR_diff'] = df['hr'].diff().fillna(0)
    return df[['SpO2', 'hr', 'HR_std', 'SpO2_var', 'HR_diff']]

X_new_test = extract_features(new_test_df)
y_new_test = new_test_df['stress']

# Standardize features (Use the same scaler as before)
scaler = StandardScaler()
X_new_test = scaler.fit_transform(X_new_test)

# Make predictions with the pretrained model
y_new_pred = best_model.predict(X_new_test)

# Evaluate Performance
f1 = f1_score(y_new_test, y_new_pred, average='weighted')
print(f"F1 Score: {f1:.2f}")
print(classification_report(y_new_test, y_new_pred))
