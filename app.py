import streamlit as st

st.set_page_config(page_title="Calories Burn Prediction", layout="wide")

st.title("Calories Burn Prediction")
st.write("""
Aplikasi ini digunakan untuk memprediksi jumlah kalori yang terbakar 
berdasarkan data aktivitas fisik menggunakan Machine Learning.
""")

st.subheader("Pipeline")
st.write("""
1. Exploratory Data Analysis (EDA)
2. Data Preprocessing
3. Model Training
4. Model Evaluation
5. Demo
""")