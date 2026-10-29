
# ============================================================
# SALARY INCREMENT PREDICTION
# CatBoost Regressor + Hyperparameter Tuning + NLP
# Dataset: data/salary_increment.csv
# ============================================================

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "salary_increment.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_PATH = MODEL_DIR / "salary_increment_catboost_nlp.cbm"
VECTORIZER_PATH = MODEL_DIR / "salary_increment_tfidf_vectorizer.joblib"
SVD_PATH = MODEL_DIR / "salary_increment_svd.joblib"
METADATA_PATH = MODEL_DIR / "salary_increment_metadata.json"

METRICS_PATH = OUTPUT_DIR / "salary_increment_metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "salary_increment_test_predictions.csv"
TUNING_PATH = OUTPUT_DIR / "salary_increment_hyperparameter_results.csv"

TARGET_COL = "Increment_Percentage"

TEXT_COLS = [
    "Employee_Self_Review",
    "Manager_Feedback_Pre_Review",
    "Achievements_Text_Pre_Review",
]

# Remove IDs, dates and post-decision/target-derived columns.
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


def calculate_metrics(y_true, predictions):
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
    }


def create_text_features(frame, vectorizer, svd):
    corpus = [
        " ".join(clean_text(row[col]) for col in TEXT_COLS)
        for _, row in frame.iterrows()
    ]

    tfidf_matrix = vectorizer.transform(corpus)
    return svd.transform(tfidf_matrix)


# ============================================================
# 3. LOAD AND VALIDATE DATASET
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

    print("\n========== DATASET INFORMATION ==========")
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    required_columns = [TARGET_COL] + TEXT_COLS
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df[TARGET_COL] = pd.to_numeric(
        df[TARGET_COL], errors="coerce"
    )

    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    if len(df) < 30:
        raise ValueError(
            "At least 30 valid rows are required for this workflow."
        )

    for col in TEXT_COLS:
        df[col] = df[col].fillna("").astype(str)

    y = df[TARGET_COL].astype(float)

    print("\nTarget variable summary:")
    print(y.describe().round(3))


    # ========================================================
    # 4. SPLIT DATA BEFORE PREPROCESSING
    # ========================================================

    all_indices = np.arange(len(df))

    train_val_indices, test_indices = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    relative_validation_size = (
        VALIDATION_SIZE / (1.0 - TEST_SIZE)
    )

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

    y_train = y.iloc[train_indices].reset_index(drop=True)
    y_validation = y.iloc[validation_indices].reset_index(drop=True)
    y_test = y.iloc[test_indices].reset_index(drop=True)

    print("\n========== DATA SPLITS ==========")
    print("Training rows:", len(train_df))
    print("Validation rows:", len(validation_df))
    print("Testing rows:", len(test_df))


    # ========================================================
    # 5. NLP: TF-IDF + TRUNCATED SVD
    # ========================================================

    train_corpus = [
        " ".join(clean_text(row[col]) for col in TEXT_COLS)
        for _, row in train_df.iterrows()
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
        min_df=2,
    )

    try:
        train_tfidf = vectorizer.fit_transform(train_corpus)
    except ValueError as exc:
        raise ValueError(
            "Could not create a TF-IDF vocabulary. "
            "Check whether the text columns contain useful text."
        ) from exc

    print("\nTF-IDF vocabulary size:", len(vectorizer.vocabulary_))

    n_components = min(
        N_SVD_COMPONENTS,
        train_tfidf.shape[1] - 1,
        train_tfidf.shape[0] - 1,
    )

    if n_components < 1:
        raise ValueError(
            "Insufficient text data for TruncatedSVD."
        )

    svd = TruncatedSVD(
        n_components=n_components,
        random_state=RANDOM_STATE,
    )

    train_text_features = svd.fit_transform(train_tfidf)

    validation_text_features = create_text_features(
        validation_df, vectorizer, svd
    )
    test_text_features = create_text_features(
        test_df, vectorizer, svd
    )

    svd_columns = [
        f"NLP_SVD_{i + 1}" for i in range(n_components)
    ]

    print("SVD NLP features:", n_components)
    print(
        "Explained variance ratio:",
        round(float(svd.explained_variance_ratio_.sum()), 4),
    )


    # ========================================================
    # 6. PREPARE STRUCTURED FEATURES
    # ========================================================

    excluded_columns = set(
        DROP_COLS + TEXT_COLS + [TARGET_COL]
    )

    feature_columns = [
        col for col in df.columns
        if col not in excluded_columns
    ]

    if not feature_columns:
        raise ValueError(
            "No structured features remain after exclusions."
        )

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

    for col in categorical_columns:
        X_train[col] = X_train[col].fillna("Unknown").astype(str)
        X_validation[col] = (
            X_validation[col].fillna("Unknown").astype(str)
        )
        X_test[col] = X_test[col].fillna("Unknown").astype(str)

    numeric_medians = {}

    for col in numeric_columns:
        X_train[col] = pd.to_numeric(
            X_train[col], errors="coerce"
        )
        X_validation[col] = pd.to_numeric(
            X_validation[col], errors="coerce"
        )
        X_test[col] = pd.to_numeric(
            X_test[col], errors="coerce"
        )

        median = X_train[col].median()

        if pd.isna(median):
            median = 0.0

        numeric_medians[col] = float(median)

        X_train[col] = X_train[col].fillna(median)
        X_validation[col] = X_validation[col].fillna(median)
        X_test[col] = X_test[col].fillna(median)

    # Add the NLP features to the structured data.
    train_nlp_df = pd.DataFrame(
        train_text_features, columns=svd_columns
    )
    validation_nlp_df = pd.DataFrame(
        validation_text_features, columns=svd_columns
    )
    test_nlp_df = pd.DataFrame(
        test_text_features, columns=svd_columns
    )

    X_train = pd.concat(
        [X_train.reset_index(drop=True), train_nlp_df], axis=1
    )
    X_validation = pd.concat(
        [X_validation.reset_index(drop=True), validation_nlp_df],
        axis=1,
    )
    X_test = pd.concat(
        [X_test.reset_index(drop=True), test_nlp_df], axis=1
    )

    final_feature_columns = X_train.columns.tolist()

    categorical_feature_indices = [
        final_feature_columns.index(col)
        for col in categorical_columns
    ]

    print("\nStructured features:", len(feature_columns))
    print("Categorical columns:", categorical_columns)
    print("Total model features:", len(final_feature_columns))


    # ========================================================
    # 7. CATBOOST HYPERPARAMETER TUNING
    # ========================================================

    print("\n========== HYPERPARAMETER TUNING ==========")

    base_model = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
    )

    param_distributions = {
        "iterations": [300, 500, 700, 900],
        "depth": [4, 5, 6, 7, 8],
        "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.10],
        "l2_leaf_reg": [3, 5, 7, 9],
        "random_strength": [0.5, 1, 2],
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

    search.fit(
        X_train,
        y_train,
        cat_features=categorical_feature_indices,
    )

    print("\nBest hyperparameters:")
    print(search.best_params_)
    print("Best cross-validation RMSE:", round(-search.best_score_, 4))

    tuning_results = pd.DataFrame(search.cv_results_)
    tuning_results = tuning_results.sort_values("rank_test_score")
    tuning_results.to_csv(TUNING_PATH, index=False)


    # ========================================================
    # 8. TRAIN FINAL CATBOOST REGRESSION MODEL
    # ========================================================

    print("\n========== FINAL MODEL TRAINING ==========")

    model = CatBoostRegressor(
        **search.best_params_,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=RANDOM_STATE,
        verbose=100,
        allow_writing_files=False,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_feature_indices,
        eval_set=(X_validation, y_validation),
        early_stopping_rounds=100,
        verbose=100,
    )


    # ========================================================
    # 9. EVALUATE ON UNSEEN TEST DATA
    # ========================================================

    print("\n========== TEST RESULTS ==========")

    test_predictions = model.predict(X_test)

    metrics = calculate_metrics(y_test, test_predictions)

    print(f"MAE:  {metrics['mae']:.4f} percentage points")
    print(f"RMSE: {metrics['rmse']:.4f} percentage points")
    print(f"R²:   {metrics['r2']:.4f}")

    # Baseline: predict the mean increment from training data.
    baseline_predictions = np.full(
        len(y_test), y_train.mean()
    )

    baseline_metrics = calculate_metrics(
        y_test, baseline_predictions
    )

    print("\n========== BASELINE COMPARISON ==========")
    print(f"Baseline MAE:  {baseline_metrics['mae']:.4f}")
    print(f"Baseline RMSE: {baseline_metrics['rmse']:.4f}")
    print(f"Baseline R²:   {baseline_metrics['r2']:.4f}")


    # ========================================================
    # 10. SAVE MODEL AND PREPROCESSING FILES
    # ========================================================

    model.save_model(str(MODEL_PATH))
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(svd, SVD_PATH)

    metadata = {
        "target_column": TARGET_COL,
        "text_columns": TEXT_COLS,
        "dropped_columns": DROP_COLS,
        "structured_feature_columns": feature_columns,
        "feature_columns": final_feature_columns,
        "categorical_columns": categorical_columns,
        "categorical_feature_indices": categorical_feature_indices,
        "numeric_columns": numeric_columns,
        "numeric_medians": numeric_medians,
        "svd_columns": svd_columns,
        "svd_components": n_components,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "validation_size": VALIDATION_SIZE,
        "best_params": search.best_params_,
        "model_type": "CatBoostRegressor with TF-IDF and TruncatedSVD",
        "warning": (
            "If this dataset is synthetic, test metrics do not establish "
            "real-world salary increment prediction performance."
        ),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(
            {
                "test": metrics,
                "training_mean_baseline": baseline_metrics,
                "best_cv_rmse": float(-search.best_score_),
                "best_params": search.best_params_,
            },
            file,
            indent=4,
        )


    # ========================================================
    # 11. SAVE TEST PREDICTIONS
    # ========================================================

    output_df = test_df.copy()

    output_df["Actual_Increment_Percentage"] = y_test.values
    output_df["Predicted_Increment_Percentage"] = test_predictions
    output_df["Absolute_Error"] = np.abs(
        y_test.values - test_predictions
    )

    output_df.to_csv(PREDICTIONS_PATH, index=False)


    # ========================================================
    # 12. COMPLETION SUMMARY
    # ========================================================

    print("\n========== FILES SAVED ==========")
    print("Model:", MODEL_PATH)
    print("TF-IDF vectorizer:", VECTORIZER_PATH)
    print("SVD transformer:", SVD_PATH)
    print("Metadata:", METADATA_PATH)
    print("Metrics:", METRICS_PATH)
    print("Test predictions:", PREDICTIONS_PATH)
    print("Hyperparameter results:", TUNING_PATH)
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()


