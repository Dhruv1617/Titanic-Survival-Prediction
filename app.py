from flask import Flask, render_template, request
import pickle
import pandas as pd

with open('titanic_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    
    input_data = pd.DataFrame([{
        'Pclass': int(data['Pclass']),
        'Age': float(data['Age']),
        'SibSp': int(data['SibSp']),
        'Parch': int(data['Parch']),
        'Fare': float(data['Fare']),
        'Sex': data['Sex'],  
        'Embarked': data['Embarked']  
    }])


    print("Prediction Input Data:")
    print(input_data)

    prediction = model.predict(input_data)[0]
    result = "Survived" if prediction == 1 else "Not Survived"
    return render_template('result.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
