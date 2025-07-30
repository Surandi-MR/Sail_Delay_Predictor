import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# 1. Load and prepare training data
df = pd.read_csv("C:\\Users\\RISINA\\Desktop\\Thushan\\Analyzing Sail Delays (1).csv")
df.drop(columns=['Sail ID', 'Transfer Date', 'Ship Date'], inplace=True)

# Drop rows with missing values in features or target
df = df.dropna(subset=['Market Segment', 'Department', 'Sail Luff', 'Sail Hours', 'Graphics', 'Lead time', 'Delay status'])
X = df.drop(columns=['Delay status'])
y = df['Delay status']

categorical_cols = ['Market Segment', 'Department', 'Graphics']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# 2. Preprocess
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

X_encoded = preprocessor.fit_transform(X)

# 3. Balance data with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_encoded, y)

# 4. Train model
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, stratify=y_resampled, test_size=0.2, random_state=42
)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# 5. Function to predict for Excel file
def predict_from_excel():
    try:
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if not file_path:
            return
        file_label.config(text=f"Selected file: {file_path.split('/')[-1]}")

        input_df = pd.read_excel(file_path)

        required_cols = ['Market Segment', 'Department', 'Sail Luff', 'Sail Hours', 'Graphics', 'Lead time']
        if not all(col in input_df.columns for col in required_cols):
            messagebox.showerror("Error", f"Missing one or more required columns:\n{required_cols}")
            return

        initial_rows = input_df.shape[0]
        input_data = input_df[required_cols].dropna()
        removed_rows = initial_rows - input_data.shape[0]

        if input_data.empty:
            messagebox.showerror("Error", "All rows have missing values. No predictions made.")
            return