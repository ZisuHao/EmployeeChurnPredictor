import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("importFeatureRFdf.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

model = RandomForestClassifier(random_state = 42)
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
