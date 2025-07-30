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