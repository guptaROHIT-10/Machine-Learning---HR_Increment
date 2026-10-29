
# ============================================================
# MODEL COMPARISON CHART
# ============================================================
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd


st.divider()
st.subheader("Model Performance Comparison")

comparison = pd.DataFrame({
    "Model": [
        "CatBoost",
        "Linear Regression",
        "Random Forest",
    ],
    "MAE": [1.4848, 1.3932, 1.5664],
    "RMSE": [1.8400, 1.7327, 1.9316],
    "R2": [0.3097, 0.3879, 0.2393],
})

st.dataframe(comparison, use_container_width=True)

# MAE and RMSE chart
fig, ax = plt.subplots(figsize=(9, 5))

comparison.plot(
    x="Model",
    y=["MAE", "RMSE"],
    kind="bar",
    ax=ax,
)

ax.set_title("MAE and RMSE Comparison")
ax.set_ylabel("Error (percentage points)")
ax.set_xlabel("Model")
ax.tick_params(axis="x", rotation=0)
ax.legend()
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

# R² chart
fig, ax = plt.subplots(figsize=(9, 4))

ax.bar(comparison["Model"], comparison["R2"])
ax.set_title("R² Comparison")
ax.set_ylabel("R² Score")
ax.set_xlabel("Model")
ax.tick_params(axis="x", rotation=0)
ax.axhline(y=0, linewidth=0.8)
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)