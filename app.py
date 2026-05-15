from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

# Load the trained model
with open('malaria_drug_recommender_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the saved TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Load the saved label encoders
with open('le_gender.pkl', 'rb') as f:
    le_gender = pickle.load(f)
with open('le_issues.pkl', 'rb') as f:
    le_issues = pickle.load(f)

# Load the label encoder for the target (drug classes)
with open('le_drug.pkl', 'rb') as f:
    le_drug = pickle.load(f)

unique_issues = ['asthma', 'diabetes', 'none', 'ulcer']

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        age = request.form.get('AGE')
        gender = request.form.get('GENDER')
        weight = request.form.get('WEIGHT(kg)')
        temp = request.form.get('TEMPERATURE(degreec)')
        issues = request.form.get('Underlying Issues')
        symptoms = request.form.getlist('SYMPTOMS')
        symptoms_str = ', '.join(symptoms)
        # Encode categorical features using the saved encoders
        try:
            gender_encoded = le_gender.transform([gender.lower()])[0]
        except Exception:
            gender_encoded = 0  # fallback
        try:
            issues_encoded = le_issues.transform([issues.lower()])[0]
        except Exception:
            issues_encoded = 0  # fallback
        # TF-IDF features
        tfidf_features = tfidf.transform([symptoms_str]).toarray()[0]
        # Combine all features in the same order as training
        input_features = [
            float(age),
            gender_encoded,
            float(weight),
            float(temp),
            issues_encoded
        ]
        final_features = np.concatenate([np.array(input_features, dtype=float), tfidf_features])
        final_features = final_features.reshape(1, -1)
        try:
            pred = model.predict(final_features)
            drug = le_drug.inverse_transform([pred[0]])[0]
            if symptoms:
                prediction = f"Based on the symptoms you selected ({', '.join(symptoms)}), the model recommends this drug: {drug}."
            else:
                prediction = f"The model recommends this drug: {drug}."
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template('index.html', prediction=prediction, unique_issues=unique_issues)

@app.route('/predict', methods=['POST'])
def predict():
    return home()

if __name__ == '__main__':
    app.run(debug=True)
