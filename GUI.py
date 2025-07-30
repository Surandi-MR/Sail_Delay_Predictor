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
        
         # Predict
        input_encoded = preprocessor.transform(input_data)
        predictions = rf.predict(input_encoded)
        statuses = ["Delayed" if pred == 1 else "On Time" for pred in predictions]

        # Add predictions to rows that passed dropna
        input_df = input_df.loc[input_data.index]
        input_df["Predicted Status"] = statuses

        # Save
        save_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                 filetypes=[("Excel files", "*.xlsx")],
                                                 title="Save Predicted Output As")
        if save_path:
            input_df.to_excel(save_path, index=False)
            msg = f"Predictions saved to:\n{save_path}"
            if removed_rows > 0:
                msg += f"\n\nNote: {removed_rows} row(s) with missing values were skipped."
            messagebox.showinfo("Success", msg)

    except Exception as e:
        messagebox.showerror("Error", str(e))

        # 6. GUI
root = tk.Tk()
root.title("Sail Delay Predictor")
root.geometry("500x300")
root.resizable(False, False)

frame = tk.Frame(root, padx=20, pady=20)
frame.pack(fill="both", expand=True)

tk.Label(frame, text="⛵ Sail Delay Prediction Tool", font=("Arial", 16, "bold")).pack(pady=10)

tk.Label(frame, text="Upload an Excel (.xlsx) file with sail data.\nRequired columns:\n"
                     "Market Segment, Department, Sail Luff, Sail Hours, Graphics, Lead time",
         justify="center", font=("Arial", 10)).pack(pady=5)

tk.Button(frame, text="📂 Upload Excel File & Predict", command=predict_from_excel,
          font=("Arial", 11), bg="#4CAF50", fg="white", padx=10, pady=5).pack(pady=10)

file_label = tk.Label(frame, text="", font=("Arial", 9), fg="gray")
file_label.pack(pady=5)

tk.Label(frame, text="Prediction results will be saved as a new Excel file.",
         font=("Arial", 9), fg="blue").pack(pady=5)

root.mainloop()
