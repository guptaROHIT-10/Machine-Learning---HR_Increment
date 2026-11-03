
# increment_model_comparison_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Salary Increment Model Comparison",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Salary Increment Prediction")
st.subheader("Machine Learning Model Comparison Dashboard")

st.write(
    "Compare the performance of CatBoost, Linear Regression, "
    "and Random Forest regression models."
)

st.divider()

# =====================================================
# MODEL METRICS
# =====================================================

metrics = pd.DataFrame({
    "Model": [
        "CatBoost",
        "Linear Regression",
        "Random Forest"
    ],
    "MAE": [
        1.4848,
        1.3932,
        1.5664
    ],
    "RMSE": [
        1.8400,
        1.7327,
        1.9316
    ],
    "R2 Score": [
        0.3097,
        0.3879,
        0.2393
    ]
})

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Dashboard Settings")

selected_model = st.sidebar.selectbox(
    "Select Model",
    metrics["Model"].tolist()
)

# =====================================================
# BEST MODEL
# =====================================================

best_mae = metrics.loc[
    metrics["MAE"].idxmin()
]

best_rmse = metrics.loc[
    metrics["RMSE"].idxmin()
]

best_r2 = metrics.loc[
    metrics["R2 Score"].idxmax()
]

# =====================================================
# KPI CARDS
# =====================================================

st.subheader("1. Overall Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Best MAE",
    f"{best_mae['MAE']:.4f}",
    best_mae["Model"]
)

col2.metric(
    "Best RMSE",
    f"{best_rmse['RMSE']:.4f}",
    best_rmse["Model"]
)

col3.metric(
    "Best R² Score",
    f"{best_r2['R2 Score']:.4f}",
    best_r2["Model"]
)

st.success(
    "Linear Regression is the best-performing model "
    "among the three based on MAE, RMSE, and R²."
)

# =====================================================
# COMPARISON TABLE
# =====================================================

st.subheader("2. Complete Model Comparison")

st.dataframe(
    metrics.style.format({
        "MAE": "{:.4f}",
        "RMSE": "{:.4f}",
        "R2 Score": "{:.4f}"
    }).highlight_min(
        subset=["MAE", "RMSE"],
        color="#090909"
    ).highlight_max(
        subset=["R2 Score"],
        color="#0c0c0c"
    ),
    use_container_width=True,
    hide_index=True
)

# st.download_button(
#     "Download Comparison CSV",
#     data=metrics.to_csv(index=False),
#     file_name="increment_model_comparison.csv",
#     mime="text/csv"
# )

# =====================================================
# MAE COMPARISON
# =====================================================

st.subheader("3. Mean Absolute Error (MAE)")

fig, ax = plt.subplots(figsize=(9, 4))

bars = ax.bar(
    metrics["Model"],
    metrics["MAE"]
)

ax.set_title("MAE Comparison")
ax.set_ylabel("Mean Absolute Error")
ax.grid(axis="y", alpha=0.3)

for bar in bars:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{bar.get_height():.4f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.caption(
    "Lower MAE indicates smaller average absolute prediction errors."
)

# =====================================================
# RMSE COMPARISON
# =====================================================

st.subheader("4. Root Mean Squared Error (RMSE)")

fig, ax = plt.subplots(figsize=(9, 4))

bars = ax.bar(
    metrics["Model"],
    metrics["RMSE"]
)

ax.set_title("RMSE Comparison")
ax.set_ylabel("Root Mean Squared Error")
ax.grid(axis="y", alpha=0.3)

for bar in bars:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{bar.get_height():.4f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.caption(
    "Lower RMSE indicates better prediction performance, "
    "with larger errors receiving more weight."
)

# =====================================================
# R2 SCORE COMPARISON
# =====================================================

st.subheader("5. R² Score Comparison")

fig, ax = plt.subplots(figsize=(9, 4))

bars = ax.bar(
    metrics["Model"],
    metrics["R2 Score"]
)

ax.set_title("R² Score Comparison")
ax.set_ylabel("R² Score")
ax.set_ylim(0, 1)
ax.grid(axis="y", alpha=0.3)

for bar in bars:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{bar.get_height():.4f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.caption(
    "Higher R² indicates that the model explains a larger "
    "proportion of the variation in salary increment percentages."
)

# =====================================================
# COMBINED COMPARISON
# =====================================================

st.subheader("6. Combined Model Comparison")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

comparison_metrics = [
    ("MAE", "Lower is better"),
    ("RMSE", "Lower is better"),
    ("R2 Score", "Higher is better")
]

for ax, (metric, description) in zip(
    axes,
    comparison_metrics
):
    bars = ax.bar(
        metrics["Model"],
        metrics[metric]
    )

    ax.set_title(f"{metric}\n({description})")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# =====================================================
# SELECTED MODEL DETAILS
# =====================================================

st.subheader("7. Selected Model Details")

row = metrics[
    metrics["Model"] == selected_model
].iloc[0]

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{row['MAE']:.4f}")
col2.metric("RMSE", f"{row['RMSE']:.4f}")
col3.metric("R² Score", f"{row['R2 Score']:.4f}")

# =====================================================
# FINAL CONCLUSION
# =====================================================

st.subheader("8. Final Conclusion")

st.markdown("""
### Best Performing Model: Linear Regression

Based on the reported test results:

- **MAE:** 1.3932 — lowest prediction error.
- **RMSE:** 1.7327 — lowest root mean squared error.
- **R²:** 0.3879 — highest explanatory power.

Linear Regression outperformed CatBoost and Random Forest
on all three reported evaluation metrics.

This suggests that Linear Regression is the strongest
model among the three for the current dataset.
""")

st.warning(
    "An R² of 0.3879 indicates that the model explains "
    "approximately 38.79% of the variation in the target. "
    "There is still substantial unexplained variation."
)

st.caption(
    "HR Analytics | Salary Increment Model Evaluation"
)
