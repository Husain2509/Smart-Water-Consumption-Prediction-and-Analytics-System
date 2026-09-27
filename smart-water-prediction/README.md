# 💧 Smart Water Consumption Prediction & Analytics System

## 1) Project Title
Smart Water Consumption Prediction and Analytics System

## 2) Project Description
This is an end-to-end Data Science + Machine Learning project that predicts hourly water consumption from weather and time-related inputs and serves predictions through a Streamlit dashboard.

## 3) Problem Statement
Predict the target variable **Consumption** using historical records and build an explainable ML pipeline suitable for a college final-year project.

## 4) Objectives
- Inspect and validate raw dataset quality (types, nulls, duplicates, ranges).
- Perform feature engineering from Date + Time.
- Train and compare multiple regression models.
- Select and save the best model.
- Provide a user-friendly Streamlit dashboard.

## 5) Dataset Description
Expected CSV file: `data/waterconsumption_noNaN_fixedTimechange.csv`

Expected columns:
- Date
- Time
- Consumption (target)
- Temperature
- Rain
- Sun

## 6) Features
Model input features:
- Temperature
- Rain
- Sun
- Hour
- DayOfWeek
- Month
- DayOfYear
- IsWeekend
- Quarter

## 7) Target Variable
- Consumption

## 8) Technologies Used
- Python 3.10+
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Streamlit

## 9) Project Architecture
```text
smart-water-prediction/
│
├── data/
│   └── waterconsumption_noNaN_fixedTimechange.csv
├── notebooks/
│   └── water_consumption_analysis.ipynb
├── models/
│   ├── best_model.pkl
│   └── scaler.pkl (only if best model needs scaling)
├── reports/
│   ├── data_profile.json
│   ├── model_comparison.csv
│   ├── best_model_predictions.csv
│   ├── rmse_comparison.png
│   └── actual_vs_predicted.png
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── predict.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 10) Installation Instructions
```bash
cd smart-water-prediction
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 11) How to Train the Model
```bash
python src/train_model.py
```

Optional custom data path:
```bash
python src/train_model.py --data-path data/waterconsumption_noNaN_fixedTimechange.csv
```

## 12) How to Run Streamlit
```bash
streamlit run app.py
```

## 13) Model Evaluation
The pipeline compares:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. XGBoost Regressor (if installed)

Metrics:
- MAE: Mean Absolute Error
- MSE: Mean Squared Error
- RMSE: Root Mean Squared Error
- R²: Goodness of fit

Model comparison is saved in `reports/model_comparison.csv` and sorted by RMSE.

## 14) Screenshots (Placeholders)
- Dashboard Home: *(add screenshot)*
- Prediction Tab: *(add screenshot)*
- EDA Tab: *(add screenshot)*
- Model Performance Tab: *(add screenshot)*

## 15) Future Improvements
- Add lag-based forecasting features for strict time-series forecasting mode.
- Add hyperparameter tuning with time-series cross-validation.
- Add model monitoring and drift alerts.

## 16) Conclusion
This project demonstrates a complete and explainable regression workflow from raw dataset inspection to deployment-ready prediction UI with Streamlit.

## Academic Viva Notes
- **Why regression?** Consumption is continuous numeric output.
- **Why chronological split?** Prevents training on future information.
- **Why avoid leakage?** Leakage gives unrealistically high performance and poor real-world predictions.
- **Why compare multiple models?** Different algorithms capture patterns differently; best model should be evidence-based.
- **How Streamlit connects to ML?** App loads `models/best_model.pkl`, collects input, enforces feature order, and calls prediction function.
