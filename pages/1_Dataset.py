import streamlit as st
import pandas as pd
from utils.sidebar import render_sidebar
from utils.data_access import load_calories_dataframe, store_calories_dataframe 

render_sidebar("Dataset")

st.title("📂 Dataset Overview")

# df = pd.read_csv("data/calories.csv")

df, source = load_calories_dataframe()
if df is None: 
    st.warning("Dataset lokal tidak ditemukan. Silakan unggah file CSV untuk melanjutkan.")
    uploaded_file = st.file_uploader("Unggah file calories.csv", type=["csv"])
    if uploaded_file is None:
        st.stop() 
    df = store_calories_dataframe(pd.read_csv(uploaded_file), uploaded_file.name)
    source = uploaded_file.name

st.markdown("### 📊 Ringkasan Dataset")
metric_col1, metric_col2, metric_col3 = st.columns(3)
# Cari total nilai kosong (missing values)
total_missing = int(df.isna().sum().sum())
metric_col1.metric("Jumlah Baris (Rows)", f"{df.shape[0]:,}")
metric_col2.metric("Jumlah Fitur (Columns)", f"{df.shape[1]}")
metric_col3.metric("Nilai Kosong (Missing Values)", f"{total_missing:,}")
st.markdown("---")


st.subheader("Dataset Preview")
num_rows = st.slider("Pilih jumlah baris yang ditampilkan: ",5,50,100)
show_full = st.checkbox("Tampilkan seluruh dataset")
if show_full:
    st.dataframe(df)
else:
    st.dataframe(df.head(num_rows))



st.subheader("Features")
col1,col2 = st.columns(2)
with col1: 
    st.write(df.columns.tolist())

with col2: 
    st.write("**Description**")
    st.write("""
    - **Gender**: Jenis kelamin pengguna  
    - **Age**: Umur pengguna  
    - **Height**: Tinggi badan  
    - **Weight**: Berat badan  
    - **Duration**: Durasi olahraga (menit)  
    - **Heart_rate**: Detak jantung saat olahraga  
    - **Body_temp**: Suhu tubuh  
    - **Calories**: Kalori yang terbakar (target)
""")
st.info("Catatan: Kolom identitas pengguna (User_ID) akan dihapus secara otomatis sebelum proses training agar tidak mempengaruhi keakuratan model.")