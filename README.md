# Malaria Drug Recommender System

This project is a machine learning-based system for recommending malaria drugs based on patient symptoms and clinical data. It includes data preprocessing, feature engineering, model training, and a web application for predictions.

## Features
- Cleans and preprocesses malaria patient data
- Extracts and encodes symptoms using TF-IDF
- Handles class imbalance with SMOTE
- Compares multiple models (Random Forest, Decision Tree, XGBoost)
- Saves the best model for deployment
- Provides a Flask web app for user interaction

## Project Structure
```
├── app.py                        # Flask web application
├── think.ipynb                   # Data analysis and model training notebook
├── cleaned_malaria.csv           # Cleaned dataset
├── malaria_drug_recommender_model.pkl  # Trained model
├── tfidf_vectorizer.pkl          # Saved TF-IDF vectorizer
├── le_gender.pkl                 # Label encoder for gender
├── le_issues.pkl                 # Label encoder for underlying issues
├── le_drug.pkl                   # Label encoder for drug classes
├── static/
│   └── style.css                 # CSS for the web app
├── templates/
│   └── index.html                # HTML template for the web app
└── README.md                     # Project documentation
```

## Setup Instructions
1. **Clone the repository**
2. **Install dependencies**
   - Python 3.8+
   - Install required packages:
     ```bash
     pip install -r requirements.txt
     ```
   - Or manually:
     ```bash
     pip install flask scikit-learn imbalanced-learn xgboost pandas numpy matplotlib tabulate
     ```
3. **Run the notebook**
   - Open `think.ipynb` and run all cells to preprocess data and train models.
   - The best model and encoders will be saved as `.pkl` files.
4. **Start the web app**
   ```bash
   python app.py
   ```
   - Visit `http://localhost:5000` in your browser.

## Usage
- Enter patient details and symptoms in the web app to get a drug recommendation.
- The app uses the best-performing model (Random Forest, Decision Tree, or XGBoost) as determined in the notebook.

## Notebooks
- `think.ipynb` contains all data exploration, feature engineering, model training, and evaluation steps.
- Feature importance and model comparison are visualized in the notebook.

## License
This project is for educational and research purposes.
