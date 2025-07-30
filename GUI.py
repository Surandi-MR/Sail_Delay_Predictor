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