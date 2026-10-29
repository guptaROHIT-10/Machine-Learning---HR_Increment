
# ============================================================
# SALARY INCREMENT PREDICTION
# Random Forest Regressor + Hyperparameter Tuning + NLP
# Dataset: data/salary_increment.csv
# ============================================================

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "salary_increment.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_PATH = MODEL_DIR / "salary_increment_random_forest.joblib"
VECTORIZER_PATH = MODEL_DIR / "salary_increment_rf_tfidf.joblib"
SVD_PATH = MODEL_DIR / "salary_increment_rf_svd.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "salary_increment_rf_preprocessor.joblib"
METADATA_PATH = MODEL_DIR / "salary_increment_rf_metadata.json"

METRICS_PATH = OUTPUT_DIR / "salary_increment_rf_metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "salary_increment_rf_test_predictions.csv"
TUNING_PATH = OUTPUT_DIR / "salary_increment_rf_tuning_results.csv"

TARGET_COL = "Increment_Percentage"

TEXT_COLS = [
    "Employee_Self_Review",
    "Manager_Feedback_Pre_Review",
    "Achievements_Text_Pre_Review",
]

DROP_COLS = [
    "Employee_ID",
    "Salary_Review_Date",
    "Increment_Amount_LPA",
    "New_Annual_Salary_LPA",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
N_SVD_COMPONENTS = 30
N_ITER = 12


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).lower().split())


def make_text_corpus(frame):
    return [
        " ".join(clean_text(row[col]) for col in TEXT_COLS)
        for _, row in frame.iterrows()
    ]


def calculate_metrics(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


# ============================================================
# 3. LOAD DATASET
# ============================================================

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place salary_increment.csv inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [TARGET_COL] + TEXT_COLS
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    for col in TEXT_COLS:
        df[col] = df[col].fillna("").astype(str)

    print("\n========== DATASET ==========")
    print("Shape:", df.shape)
    print("Target:", TARGET_COL)
    print("Rows:", len(df))


    # ========================================================
    # 4. SAME 60:20:20 SPLIT AS OTHER MODELS
    # ========================================================

    indices = np.arange(len(df))

    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    relative_validation_size = VALIDATION_SIZE / (1 - TEST_SIZE)

    train_indices, validation_indices = train_test_split(
        train_val_indices,
        test_size=relative_validation_size,
        random_state=RANDOM_STATE,
    )

    train_df = df.iloc[train_indices].copy().reset_index(drop=True)
    validation_df = (
        df.iloc[validation_indices].copy().reset_index(drop=True)
    )
    test_df = df.iloc[test_indices].copy().reset_index(drop=True)

    y_train = train_df[TARGET_COL].astype(float)
    y_validation = validation_df[TARGET_COL].astype(float)
    y_test = test_df[TARGET_COL].astype(float)

    print("\n========== DATA SPLITS ==========")
    print("Training:", len(train_df))
    print("Validation:", len(validation_df))
    print("Testing:", len(test_df))


    # ========================================================
    # 5. SELECT STRUCTURED FEATURES
    # ========================================================

    excluded_columns = set(DROP_COLS + TEXT_COLS + [TARGET_COL])

    feature_columns = [
        col for col in df.columns
        if col not in excluded_columns
    ]

    X_train = train_df[feature_columns].copy()
    X_validation = validation_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()

    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_columns = [
        col for col in feature_columns
        if col not in categorical_columns
    ]

    for col in numeric_columns:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_validation[col] = pd.to_numeric(
            X_validation[col], errors="coerce"
        )
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    print("\nStructured features:", len(feature_columns))
    print("Categorical columns:", categorical_columns)
    print("Numeric columns:", numeric_columns)


    # ========================================================
    # 6. NLP: TF-IDF + TRUNCATED SVD
    # Fit on training data only.
    # ========================================================

    train_corpus = make_text_corpus(train_df)
    validation_corpus = make_text_corpus(validation_df)
    test_corpus = make_text_corpus(test_df)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
        min_df=2,
    )

    try:
        train_tfidf = vectorizer.fit_transform(train_corpus)
        validation_tfidf = vectorizer.transform(validation_corpus)
        test_tfidf = vectorizer.transform(test_corpus)
    except ValueError as exc:
        raise ValueError(
            "TF-IDF could not build a vocabulary. "
            "Check whether the text columns contain useful text."
        ) from exc

    n_components = min(
        N_SVD_COMPONENTS,
        train_tfidf.shape[1] - 1,
        train_tfidf.shape[0] - 1,
    )

    if n_components < 1:
        raise ValueError("Not enough text data for TruncatedSVD.")

    svd = TruncatedSVD(
        n_components=n_components,
        random_state=RANDOM_STATE,
    )

    train_text_features = svd.fit_transform(train_tfidf)
    validation_text_features = svd.transform(validation_tfidf)
    test_text_features = svd.transform(test_tfidf)

    svd_columns = [
        f"NLP_SVD_{i + 1}" for i in range(n_components)
    ]

    print("\nTF-IDF vocabulary size:", len(vectorizer.vocabulary_))
    print("SVD NLP features:", n_components)
    print(
        "Explained variance ratio:",
        round(float(svd.explained_variance_ratio_.sum()), 4),
    )


    # ========================================================
    # 7. PREPROCESS STRUCTURED FEATURES
    # ========================================================

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Unknown",
                ),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []

    if numeric_columns:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    train_structured = preprocessor.fit_transform(X_train)
    validation_structured = preprocessor.transform(X_validation)
    test_structured = preprocessor.transform(X_test)

    # Combine structured features with the NLP SVD features.
    train_combined = hstack(
        [csr_matrix(train_structured), csr_matrix(train_text_features)],
        format="csr",
    )

    validation_combined = hstack(
        [
            csr_matrix(validation_structured),
            csr_matrix(validation_text_features),
        ],
        format="csr",
    )

    test_combined = hstack(
        [csr_matrix(test_structured), csr_matrix(test_text_features)],
        format="csr",
    )

    print("\nTraining matrix:", train_combined.shape)
    print("Validation matrix:", validation_combined.shape)
    print("Testing matrix:", test_combined.shape)


    # ========================================================
    # 8. RANDOM FOREST HYPERPARAMETER TUNING
    # ========================================================

    print("\n========== RANDOM FOREST TUNING ==========")

    base_model = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    param_distributions = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 15, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [1.0, "sqrt", 0.7],
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=N_ITER,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=2,
        refit=True,
        error_score="raise",
    )

    search.fit(train_combined, y_train)

    print("\nBest hyperparameters:")
    print(search.best_params_)
    print("Best CV RMSE:", round(-search.best_score_, 4))

    tuning_results = pd.DataFrame(search.cv_results_)
    tuning_results = tuning_results.sort_values("rank_test_score")
    tuning_results.to_csv(TUNING_PATH, index=False)


    # ========================================================
    # 9. TRAIN FINAL MODEL
    # ========================================================

    print("\n========== TRAINING FINAL RANDOM FOREST ==========")

    model = RandomForestRegressor(
        **search.best_params_,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(train_combined, y_train)

    print("Training completed.")


    # ========================================================
    # 10. VALIDATION AND TEST EVALUATION
    # ========================================================

    validation_predictions = model.predict(validation_combined)
    test_predictions = model.predict(test_combined)

    validation_metrics = calculate_metrics(
        y_validation, validation_predictions
    )

    test_metrics = calculate_metrics(y_test, test_predictions)

    baseline_predictions = np.full(
        len(y_test), y_train.mean()
    )

    baseline_metrics = calculate_metrics(
        y_test, baseline_predictions
    )

    print("\n========== VALIDATION RESULTS ==========")
    print(f"MAE:  {validation_metrics['mae']:.4f}")
    print(f"RMSE: {validation_metrics['rmse']:.4f}")
    print(f"R²:   {validation_metrics['r2']:.4f}")

    print("\n========== TEST RESULTS ==========")
    print(f"MAE:  {test_metrics['mae']:.4f} percentage points")
    print(f"RMSE: {test_metrics['rmse']:.4f} percentage points")
    print(f"R²:   {test_metrics['r2']:.4f}")

    print("\n========== BASELINE COMPARISON ==========")
    print(f"Baseline MAE:  {baseline_metrics['mae']:.4f}")
    print(f"Baseline RMSE: {baseline_metrics['rmse']:.4f}")
    print(f"Baseline R²:   {baseline_metrics['r2']:.4f}")


    # ========================================================
    # 11. SAVE MODEL AND PREPROCESSING OBJECTS
    # ========================================================

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(svd, SVD_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    metadata = {
        "model_type": "RandomForestRegressor",
        "target_column": TARGET_COL,
        "text_columns": TEXT_COLS,
        "dropped_columns": DROP_COLS,
        "structured_feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "svd_columns": svd_columns,
        "svd_components": n_components,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "validation_size": VALIDATION_SIZE,
        "best_params": search.best_params_,
        "warning": (
            "Performance on synthetic data does not establish "
            "real-world predictive accuracy."
        ),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(
            {
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "training_mean_baseline": baseline_metrics,
                "best_cv_rmse": float(-search.best_score_),
                "best_params": search.best_params_,
            },
            file,
            indent=4,
        )


    # ========================================================
    # 12. SAVE TEST PREDICTIONS
    # ========================================================

    output_df = test_df.copy()

    output_df["Actual_Increment_Percentage"] = y_test.to_numpy()
    output_df["Predicted_Increment_Percentage"] = test_predictions
    output_df["Absolute_Error"] = np.abs(
        y_test.to_numpy() - test_predictions
    )

    output_df.to_csv(PREDICTIONS_PATH, index=False)


    # ========================================================
    # 13. COMPLETION SUMMARY
    # ========================================================

    print("\n========== FILES SAVED ==========")
    print("Model:", MODEL_PATH)
    print("Vectorizer:", VECTORIZER_PATH)
    print("SVD:", SVD_PATH)
    print("Preprocessor:", PREPROCESSOR_PATH)
    print("Metadata:", METADATA_PATH)
    print("Metrics:", METRICS_PATH)
    print("Predictions:", PREDICTIONS_PATH)
    print("Tuning results:", TUNING_PATH)

    print("\nRandom Forest training completed successfully.")


if __name__ == "__main__":
    main()
