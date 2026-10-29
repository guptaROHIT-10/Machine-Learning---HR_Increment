
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from scipy.sparse import csr_matrix, hstack


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "salary_increment.csv"
MODEL_PATH = BASE_DIR / "models" / "salary_increment_linear_regression.joblib"
VECTORIZER_PATH = BASE_DIR / "models" / "salary_increment_lr_tfidf.joblib"
SVD_PATH = BASE_DIR / "models" / "salary_increment_lr_svd.joblib"
PREPROCESSOR_PATH = BASE_DIR / "models" / "salary_increment_lr_preprocessor.joblib"
METADATA_PATH = BASE_DIR / "models" / "salary_increment_lr_metadata.json"

TEXT_COLS = [
    "Employee_Self_Review",
    "Manager_Feedback_Pre_Review",
    "Achievements_Text_Pre_Review",
]


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Salary Increment Predictor",
    page_icon="📈",
    layout="wide",
)

st.title("Salary Increment Prediction")
st.caption("Linear Regression + NLP (TF-IDF and Truncated SVD)")

st.write(
    "Enter the employee's pre-review information to estimate "
    "their salary increment percentage."
)


# ============================================================
# 3. LOAD MODEL AND PREPROCESSING ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():
    required_paths = [
        DATA_PATH,
        MODEL_PATH,
        VECTORIZER_PATH,
        SVD_PATH,
        PREPROCESSOR_PATH,
        METADATA_PATH,
    ]

    missing_paths = [
        str(path) for path in required_paths if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "The following required files are missing:\n"
            + "\n".join(missing_paths)
            + "\nRun increment_linear_regression.py first."
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    svd = joblib.load(SVD_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    data = pd.read_csv(DATA_PATH)

    return model, vectorizer, svd, preprocessor, metadata, data


try:
    (
        model,
        vectorizer,
        svd,
        preprocessor,
        metadata,
        reference_data,
    ) = load_artifacts()

except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).lower().split())


def make_text_corpus(employee):
    """
    Keep the same order and text-combination logic as training.
    """
    return " ".join(
        clean_text(employee[col]) for col in TEXT_COLS
    )


def get_category_options(column):
    """
    Use values observed in the training dataset.
    Include Unknown for unseen or unspecified values.
    """
    values = (
        reference_data[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    values = sorted(values)

    if "Unknown" not in values:
        values.insert(0, "Unknown")

    return values if values else ["Unknown"]


def get_numeric_default(column):
    values = pd.to_numeric(
        reference_data[column], errors="coerce"
    ).dropna()

    if values.empty:
        return 0.0

    return float(values.median())


def get_numeric_step(column):
    name = column.lower()

    if "salary" in name or "lpa" in name:
        return 0.1

    if "percentage" in name or "score" in name or "rating" in name:
        return 0.1

    if "years" in name:
        return 1.0

    return 1.0


def format_percentage(value):
    return f"{value:.2f}%"


# ============================================================
# 5. EMPLOYEE INPUT FORM
# ============================================================

st.subheader("1. Employee Information")

structured_columns = metadata["structured_feature_columns"]
categorical_columns = metadata["categorical_columns"]
numeric_columns = metadata["numeric_columns"]

employee_input = {}

with st.form("salary_increment_form"):

    left, right = st.columns(2)

    # Categorical fields
    with left:
        st.markdown("**Categorical information**")

        for column in categorical_columns:
            options = get_category_options(column)

            employee_input[column] = st.selectbox(
                column.replace("_", " "),
                options=options,
                key=f"cat_{column}",
            )

    # Numeric fields
    with right:
        st.markdown("**Numerical information**")

        for column in numeric_columns:
            default_value = get_numeric_default(column)

            employee_input[column] = st.number_input(
                column.replace("_", " "),
                value=default_value,
                step=get_numeric_step(column),
                format="%.2f",
                key=f"num_{column}",
            )

    st.divider()

    st.subheader("2. Employee Review Text")

    employee_input[TEXT_COLS[0]] = st.text_area(
        "Employee Self Review",
        placeholder=(
            "Describe achievements, performance, contributions "
            "and professional development..."
        ),
        height=120,
    )

    employee_input[TEXT_COLS[1]] = st.text_area(
        "Manager Feedback Before Review",
        placeholder=(
            "Enter the manager's pre-review feedback..."
        ),
        height=120,
    )

    employee_input[TEXT_COLS[2]] = st.text_area(
        "Achievements Summary",
        placeholder=(
            "Describe completed projects, measurable impact "
            "and important accomplishments..."
        ),
        height=120,
    )

    submitted = st.form_submit_button(
        "Predict Salary Increment",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# 6. PREDICTION
# ============================================================

if submitted:

    try:
        # Build structured input in the exact training column order.
        structured_input = pd.DataFrame(
            [[employee_input[col] for col in structured_columns]],
            columns=structured_columns,
        )

        # Transform the structured features.
        structured_features = preprocessor.transform(
            structured_input
        )

        # Build NLP input using the same text fields and order.
        combined_text = make_text_corpus(employee_input)

        tfidf_features = vectorizer.transform([combined_text])
        nlp_features = svd.transform(tfidf_features)

        # Combine structured and NLP features in the same order
        # used by the training script.
        prediction_features = hstack(
            [
                csr_matrix(structured_features),
                csr_matrix(nlp_features),
            ],
            format="csr",
        )

        predicted_increment = float(
            model.predict(prediction_features)[0]
        )

        st.divider()
        st.subheader("3. Prediction Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                label="Predicted Increment",
                value=format_percentage(predicted_increment),
            )

        with result_col2:
            if "Current_Annual_Salary_LPA" in employee_input:
                current_salary = float(
                    employee_input["Current_Annual_Salary_LPA"]
                )

                estimated_amount = (
                    current_salary * predicted_increment / 100
                )

                st.metric(
                    label="Estimated Increment Amount",
                    value=f"₹{estimated_amount:.2f} LPA",
                )

                st.caption(
                    "Calculated from the predicted percentage "
                    "and entered current salary."
                )
            else:
                st.info(
                    "Current_Annual_Salary_LPA is not among "
                    "the model's input features."
                )

        if predicted_increment < 0:
            st.warning(
                "The model predicted a negative increment. "
                "This can occur with Linear Regression; review "
                "the result rather than treating it as a final "
                "salary decision."
            )

        st.info(
            "This is a model estimate, not a guaranteed increment "
            "or an employment decision. Model quality depends on "
            "the training data and its relationship to real outcomes."
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


# ============================================================
# 7. MODEL INFORMATION
# ============================================================

with st.expander("Model Information"):
    st.write("**Model:** Linear Regression")
    st.write("**Target:** Increment_Percentage")
    st.write("**NLP:** TF-IDF + Truncated SVD")
    st.write("**Text fields:**")

    for col in TEXT_COLS:
        st.write(f"- {col}")

    st.write(
        "**Structured features:** "
        + ", ".join(structured_columns)
    )

    st.caption(
        "The model uses the saved training-time preprocessing "
        "objects. Retrain the model whenever its feature schema "
        "or preprocessing pipeline changes."
    )

