# Employee Salary Increment Prediction System

A machine learning project that estimates employee salary increment percentages using compensation information, performance indicators, experience, skills assessments, and employee review text.

The project compares regression algorithms to evaluate their ability to predict salary increment percentages.

## Features

- Predicts employee salary increment percentages.
- Uses employee salary, experience, job level, and market benchmark features.
- Analyses performance ratings, KPI scores, training, awards, and leadership assessments.
- Processes self-reviews, manager feedback, and achievement descriptions using NLP.
- Uses TF-IDF and Truncated SVD for text feature extraction.
- Compares CatBoost, Linear Regression, and Random Forest.
- Evaluates models using MAE, RMSE, and R².
- Supports visual comparison of model performance through Streamlit and Matplotlib.

## Tech Stack

- **Language:** Python
- **Machine Learning:** scikit-learn, CatBoost
- **NLP:** TF-IDF, Truncated SVD
- **Data Processing:** Pandas, NumPy
- **Visualisation:** Matplotlib
- **Web Application:** Streamlit
- **Model Persistence:** Joblib

## Dataset

The salary increment dataset includes features such as:

- Department, job role, and job level
- Years at the company and experience
- Current annual salary
- Market salary benchmark
- Previous performance rating and KPI score
- Training score and awards
- Leadership and skills assessment scores
- Attendance percentage
- Employee self-review
- Manager feedback
- Achievement descriptions
- Promotion during the review cycle

**Target variable:** `Increment_Percentage`

The model predicts the increment percentage as a regression problem.

Derived post-decision values such as `Increment_Amount_LPA` and `New_Annual_Salary_LPA` should be excluded from model inputs because they are calculated using the target or the resulting salary.

## Project Structure

```text
HR-Increment/
├── data/
│   └── salary_increment.csv
├── models/
├── outputs/
├── increment_linear_app.py
├── increment_train.py
├── increment_linear_regression.py
├── increment_random_forest.py
└── README.md
```

*Adjust the tree to match the files in your repository.*

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone <your-repository-url>
cd HR-Increment
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
python -m pip install pandas numpy scikit-learn catboost streamlit matplotlib joblib
```

## Run the Application

Launch the salary increment prediction interface:

```bash
python -m streamlit run increment_linear_app.py
```

## Train the Models

Train the CatBoost regression model:

```bash
python increment_train.py
```

Train Linear Regression:

```bash
python increment_linear_regression.py
```

Train Random Forest:

```bash
python increment_random_forest.py
```

## Model Evaluation

Previously reported regression results:

| Metric | CatBoost | Linear Regression | Random Forest |
|---|---:|---:|---:|
| MAE | 1.4848 | 1.3932 | 1.5664 |
| RMSE | 1.8400 | 1.7327 | 1.9316 |
| R² | 0.3097 | 0.3879 | 0.2393 |

### Interpretation

- **Linear Regression** achieved the lowest MAE and RMSE among the three reported models.
- Linear Regression also achieved the highest R².
- CatBoost performed better than Random Forest on the reported regression metrics.
- Random Forest had the highest prediction errors among these three models.

MAE and RMSE are expressed in the target variable's units, here percentage points of salary increment if the target is recorded as a percentage.

## Future Improvements

- Evaluate additional regression algorithms.
- Improve the quality and representativeness of salary review data.
- Analyse prediction errors across departments and job levels.
- Add prediction intervals to communicate uncertainty.
- Validate the model on future review cycles.
- Provide explanations for predicted increments.

## Disclaimer

This project is intended for educational and experimental purposes. Salary increment predictions should not be used as the sole basis for compensation decisions. Real deployments require reliable data, transparent compensation policies, appropriate privacy safeguards, and human review.
