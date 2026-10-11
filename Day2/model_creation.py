
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model():
    data_path = Path(__file__).resolve().parent / 'Data' / 'day2_sample.csv'
    data = pd.read_csv(data_path)

   
    X = data[['temperature', 'pressure', 'vibration', 'rpm', 'current']]
    y = data['state']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

   
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    return model

train_model()
