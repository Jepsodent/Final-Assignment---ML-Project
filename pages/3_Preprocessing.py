import streamlit as st
import pandas as pd
from utils.preprocessing import preprocess_data, clean_data
from sklearn.model_selection import train_test_split

st.title("⚙️ Data Preprocessing")

df = pd.read_csv("data/calories.csv")
# Pilih scaler
test_size = st.slider("Pilih test size", 0.1, 0.4, 0.2)
scaler_type = st.selectbox(
    "Pilih metode scaling",
    ["standard", "minmax"]
)

if st.button("Run Preprocessing"):
    
    
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(
        df, scaler_type, test_size
    )
    st.success("Preprocessing selesai!")
    st.session_state["preprocessing_done"] = True

    st.session_state["X_train_scaled"] = X_train_scaled
    st.session_state["X_train"] = X_train
    st.session_state["X_test_scaled"] = X_test_scaled
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test

    
if st.session_state.get("preprocessing_done", False):
    tab1, tab2 = st.tabs(["Before Scaling", "After Scaling"])
    X_train = st.session_state["X_train"]
    with tab1:
        st.subheader("Before Scaling")
        st.dataframe(X_train.head())
    with tab2:
        st.subheader("After Scaling")
        
        X_scaled_df = pd.DataFrame(
            st.session_state["X_train_scaled"],
            columns=X_train.columns
        )

        st.dataframe(X_scaled_df.head())
    if st.button("Train Model"):
        st.switch_page("pages/4_Training.py")