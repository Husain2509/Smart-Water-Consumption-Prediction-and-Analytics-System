# 💧 Smart Water Consumption Prediction & Analytics System

## 1) Project Title
Smart Water Consumption Prediction and Analytics System

## 2) Project Description
This project is an end-to-end Data Science + Machine Learning system that predicts hourly water consumption from weather and time-based features. It includes data preprocessing, exploratory analysis, model training/comparison, model persistence, and a Streamlit dashboard for prediction and analytics.

## 3) Problem Statement
Given historical water consumption and environmental data (`Date`, `Time`, `Temperature`, `Rain`, `Sun`), build a robust regression pipeline that predicts `Consumption` for each hour.

## 4) Objectives
- Inspect and validate raw data quality before modeling.
- Engineer meaningful time-based features.
- Compare multiple regression algorithms.
- Select and persist the best model.
- Build an interactive Streamlit app for prediction + analytics.

## 5) Dataset Description
Expected dataset file:
- `data/waterconsumption_noNaN_fixedTimechange.csv`

Expected columns:
- `Date`
- `Time`
- `Consumption` (target)
- `Temperature`
- `Rain`
- `Sun`

The training script automatically inspects:
- shape, head, tail
- column names and dtypes
- missing values
- duplicate rows and duplicate timestamps (`Date + Time`)
- descriptive statistics
- unique values where relevant

## 6) Features
Model input features:
- `Temperature`
- `Rain`
- `Sun`
- `Hour`
- `DayOfWeek`
- `Month`
- `DayOfYear`
- `IsWeekend`
- `Quarter`

Additional engineered fields available in the processed dataset:
- `Year`, `Day`, `WeekOfYear`, and `DateTime`

## 7) Target Variable
- `Consumption`

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
│   ├── water_consumption_analysis.ipynb
│   └── water_consumption_analysis.py
├── models/
│   ├── best_model.pkl                 (generated)
│   ├── scaler.pkl                     (generated only if needed)
│   ├── feature_columns.json           (generated)
│   ├── model_comparison.csv           (generated)
│   └── best_model_predictions.csv     (generated)
├── reports/
│   └── figures/                       (generated EDA/model plots)
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── predict.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 10) Installation Instructions
```bash
cd smart-water-prediction
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

## 11) How to Train the Model
1. Place the dataset at:
   - `smart-water-prediction/data/waterconsumption_noNaN_fixedTimechange.csv`
2. Run:
```bash
python src/train_model.py
```

What this does:
- inspects the dataset first
- preprocesses and engineers date/time features
- performs chronological 80/20 split (older -> train, newer -> test)
- trains 5 regression models
- evaluates MAE, MSE, RMSE, R²
- selects best model by lowest RMSE
- saves model + metadata + plots

## 12) How to Run Streamlit
```bash
streamlit run app.py
```

If the model is not trained yet, the app shows a clear warning and how to generate artifacts.

## 13) Model Evaluation
Models compared:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. XGBoost Regressor

Metrics:
- **MAE**: average absolute error.
- **MSE**: average squared error.
- **RMSE**: square root of MSE; same scale as target.
- **R²**: proportion of target variance explained by model.

A comparison table is saved to `models/model_comparison.csv` and visualized in Streamlit.

## 14) Screenshots (Placeholders)
- `![Prediction Tab](screenshots/prediction-tab.png)`
- `![Data Overview Tab](screenshots/data-overview-tab.png)`
- `![EDA Tab](screenshots/eda-tab.png)`
- `![Model Performance Tab](screenshots/model-performance-tab.png)`

## 15) Future Improvements
- Add lag/rolling features for forecasting scenarios.
- Add hyperparameter tuning via time-series cross-validation.
- Add model retraining schedules and drift monitoring.
- Add deployment with containerization and CI/CD.

## 16) Conclusion
This project provides a complete, explainable, and student-friendly workflow from raw data inspection to production-style app usage. It avoids data leakage, respects temporal ordering, compares multiple models fairly, and serves predictions through Streamlit.

---

## Academic Viva Notes
- **Why regression?** The target (`Consumption`) is continuous numeric.
- **Why chronological split?** Random shuffling leaks future patterns into training for time-indexed data.
- **Why avoid leakage?** Leakage inflates scores and fails in real-world future predictions.
- **Why compare multiple models?** Different algorithms capture patterns differently; selection should be evidence-based.
- **Why final model selected?** Best test performance (lowest RMSE, with supporting MAE/R²).
- **How Streamlit connects to ML model?** App loads saved `best_model.pkl` and `feature_columns.json`, validates input order/types, then predicts.
