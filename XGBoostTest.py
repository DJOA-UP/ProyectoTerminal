import wfdb
import numpy as np
import pandas as pd
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Define file path
path = r'C:\Users\DJOA0\OneDrive\Desktop\SpO2HR'

# Load data from multiple subjects
num_subjects = 20  
train_subjects = 16  
test_subjects = 4    

train_data, test_data = [], []

for i in range(1, num_subjects + 1):
    record_name = f"Subject{i}_SpO2HR"
    record = wfdb.rdrecord(os.path.join(path, record_name))
    
    df = pd.DataFrame(record.p_signal, columns=record.sig_name)

    # Assign to training or testing set
    if i <= train_subjects:
        train_data.append(df)
    else:
        test_data.append(df)

# Combine all data
train_df = pd.concat(train_data, ignore_index=True)
test_df = pd.concat(test_data, ignore_index=True)

print(train_df.columns)

# Define stress classification
def stress_label(hr, spo2):
    if hr > 110 and spo2 < 95:
        return 2  # High Stress
    elif 100 <= hr <= 110 and 95 <= spo2 <= 98:
        return 1  # Moderate Stress
    else:
        return 0  # resting state

# Apply stress labels
train_df['stress'] = [stress_label(hr, spo2) for hr, spo2 in zip(train_df['hr'], train_df['SpO2'])]
test_df['stress'] = [stress_label(hr, spo2) for hr, spo2 in zip(test_df['hr'], test_df['SpO2'])]

# Feature Engineering
def extract_features(df):
    df['HR_std'] = df['hr'].rolling(window=5).std().fillna(0)
    df['SpO2_var'] = df['SpO2'].rolling(window=5).var().fillna(0)
    df['HR_diff'] = df['hr'].diff().fillna(0)
    return df[['SpO2', 'hr', 'HR_std', 'SpO2_var', 'HR_diff']]

X_train = extract_features(train_df)
X_test = extract_features(test_df)
y_train = train_df['stress']
y_test = test_df['stress']

# Handle Class Imbalance
smote = SMOTE()
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Feature Scaling
scaler = StandardScaler()
X_train_bal = scaler.fit_transform(X_train_bal)
X_test = scaler.transform(X_test)

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(XGBClassifier(), param_grid, scoring='f1_weighted', cv=5)
grid_search.fit(X_train_bal, y_train_bal)

print("Best parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate Performance
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Improved F1 Score: {f1:.2f}")

print(classification_report(y_test, y_pred))
