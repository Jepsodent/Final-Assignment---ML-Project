from utils.data_access import load_calories_dataframe
from utils.sidebar import render_sidebar
import streamlit as st
import pandas as pd
from utils.preprocessing import preprocess_data, clean_data
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Data Preprocessing", layout="wide")
st.title("⚙️ Data Preprocessing")
st.write("Persiapan data sebelum pelatihan model: Pembersihan data, Data Splitting (Train/Test), dan Scaling.")
render_sidebar("Preprocessing")

df, source = load_calories_dataframe()
if df is None:
    st.warning("Dataset belum tersedia. Silakan unggah CSV di halaman Dataset terlebih dahulu.")
    st.stop()
 
col1, col2 = st.columns(2)
with col1:
    test_size = st.slider(
        "Pilih Ukuran Data Uji (Test Size)", 
        min_value=0.1, 
        max_value=0.4, 
        value=0.2, 
        step=0.05,
        help="Proporsi data yang digunakan untuk menguji performa model (contoh: 0.20 = 20% data uji, 80% data latih)"
    )
with col2:
    scaler_type = st.selectbox(
        "Pilih Metode Scaling",
        options=["standard", "minmax"],
        format_func=lambda x: "Standard Scaler (Rata-rata 0, Standar Deviasi 1)" if x == "standard" else "Min-Max Scaler (Rentang 0 sampai 1)",
        help="Penting agar fitur dengan rentang besar tidak mendominasi model."
    )


    
    
X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(
    df, scaler_type, test_size
)

st.session_state["X_train"] = X_train
st.session_state["X_test"] = X_test
st.session_state["X_train_scaled"] = X_train_scaled
st.session_state["X_test_scaled"] = X_test_scaled
st.session_state["y_train"] = y_train
st.session_state["y_test"] = y_test
st.session_state["scaler"] = scaler
st.session_state["preprocessing_done"] = True

st.success(f"Preprocessing selesai! ({100 - int(test_size*100)}% Train / {int(test_size*100)}% Test)")

tab1, tab2 = st.tabs(["Before Scaling", "After Scaling"])
X_train = st.session_state["X_train"]
with tab1:
    st.markdown("**5 Data Teratas Sebelum Scaling (Nilai Asli):**")
    st.subheader("Before Scaling")
    st.dataframe(X_train.head())
with tab2:
    st.subheader("After Scaling")
    st.markdown(f"**5 Data Teratas Setelah Scaling ({scaler_type.capitalize()} Scaler):**")
    X_scaled_df = pd.DataFrame(
        st.session_state["X_train_scaled"],
        columns=X_train.columns
    )
    st.dataframe(X_scaled_df.head(),use_container_width=True)
if st.button("Lanjut Training Model"):
    st.switch_page("pages/4_Training-Evaluation.py")