import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

st.title("Model Evaluation")

# Validasi
if "model" not in st.session_state:
    st.warning("Silakan lakukan training terlebih dahulu.")
    st.stop()
model = st.session_state["model"]
model_name = st.session_state["model_name"]

X_test = st.session_state["X_test_scaled"]
y_test = st.session_state["y_test"]

st.subheader("Model Information")
st.write(f"Model yang digunakan: **{model_name}**")

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Evaluation Metrics")
metrics_df = pd.DataFrame({
    "Metric": ["MAE", "MSE", "R² Score"],
    "Value": [mae,mse,r2]
})


st.dataframe(metrics_df, use_container_width=True)



st.subheader("Insight")

if r2 > 0.8:
    st.success("Model memiliki performa yang sangat baik dalam memprediksi kalori.")
elif r2 > 0.6:
    st.info("Model cukup baik, namun masih bisa ditingkatkan.")
else:
    st.warning("Model kurang optimal, mungkin perlu tuning atau fitur tambahan.")

if model_name == "Random Forest":
    st.subheader("Feature Importance")

    importances = model.feature_importances_
    feature_names = st.session_state.get("X_train").columns

    df_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.dataframe(df_importance)

    st.bar_chart(df_importance.set_index("Feature"))

if st.button("Lanjut ke Prediction"):
    st.switch_page("pages/6_Demo.py")