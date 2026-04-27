import streamlit as st
import pandas as pd

st.title("📂 Dataset Overview")

df = pd.read_csv("data/calories.csv")

st.subheader("Dataset Preview")
num_rows = st.slider("Pilih jumlah baris yang ditampilkan: ",5,50,100)
show_full = st.checkbox("Tampilkan seluruh dataset")
if show_full:
    st.dataframe(df)
else:
    st.dataframe(df.head(num_rows))
st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")



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
st.info("User_Id tidak digunakan dalam model karena tidak memiliki pengaruh terhadap prediksi.")