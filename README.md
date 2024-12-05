# Titanic-Survival-Prediction

This project predicts the survival chances of passengers on the Titanic based on their characteristics such as age, gender, class, and more. It involves training a machine learning model and providing a user-friendly interface for predictions.

---

## Features
- Preprocessing of Titanic dataset using Scikit-learn.
- Machine learning model trained using Random Forest Classifier.
- Web interface built with Flask for user interaction.
- Model persistence using Pickle for deployment.

--- 

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/Titanic-Survival-Prediction/titanic-survival-prediction.git
   cd titanic-survival-prediction
   ``
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn flask

3. Run the Model:
   ```bash
   python model.py

4. Run the Flask app:
   ```bash
   python app.py

5. Open in browser:
   ```arduino
   http://127.0.0.1:5000/
