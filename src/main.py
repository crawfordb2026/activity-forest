import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load both data sets
train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

# filter out both activity and subject cols
X_train = train_df.drop['subject', 'Activity']
X_test = train_df.drop['subject', 'Activity']

# store only activity column for validation
y_train = train_df['Activity']
y_test = test_df['Activity']

scaler = StandardScaler()
scaler.fit(X_train)

# Z-score all features
X_train_scaled = scaler.transform(X_train) # fit to training data only
X_test_scaled = scaler.transform(X_test)

encoder = LabelEncoder()
encoder.fit(y_train) # fit to training data only

# create encoded matricies
y_train_encoded = encoder.transform(y_train)
y_test_encoded = encoder.transform(y_test)

X_train_split, X_test_split, y_train_val, y_test_val = train_test_split(
    X_train_scaled, 
    y_train_encoded, 
    random_state=42, 
    test_size=0.2
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit











