import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

df = pd.read_csv('dataset-path')

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna('S', inplace=True)

X = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked']]
y = df['Survived']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']),
        ('cat', OneHotEncoder(), ['Sex', 'Embarked'])
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

with open('titanic_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved successfully.")
print("ColumnTransformer categories:")
print(model.named_steps['preprocessor'].transformers_[1][1].categories_)
