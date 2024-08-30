from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

df = pd.read_csv('data/iris.csv')


X = df.drop(columns='species')
y = df['species']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

with open("decision_tree.pkl", "wb") as f:
    pickle.dump(model, f)