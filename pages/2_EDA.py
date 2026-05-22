from utils.data_access import load_calories_dataframe
from utils.sidebar import render_sidebar
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Exploratory Data Analysis (EDA)")

render_sidebar("EDA")

df, source = load_calories_dataframe()
if df is None:
    st.warning("Dataset belum tersedia. Silakan unggah CSV di halaman Dataset terlebih dahulu.")
    st.stop()

st.caption(f"Sumber data aktif: `{source}`")

st.subheader("Missing Values: ")
missing = df.isnull().sum().reset_index()
missing.columns = ["Name", "Null Value"]
st.dataframe(missing)


st.subheader("Dataset Descriptive Statistic")
st.dataframe(df.describe().round(2))
st.markdown("---")

st.subheader("Visualization")
eda_option = st.selectbox(
    "Piih Jenis Visualisasi", 
    [
        "Distribution Calories", 
        "Duration vs Calories",
        "Heart Rate vs Calories",
        "Age vs Calories",
        "Gender vs Calories",
        "Height vs Calories",
        "Weight vs Calories",
    ]
)
fig, ax = plt.subplots(figsize=(15,5))
if eda_option == "Distribution Calories":
    plt.title("Distribusi Calories")
    sns.histplot(df["Calories"], bins=30, kde=True, ax=ax, color="skyblue")
    insight = "Kerapatan grafik menumpuk di sisi kiri, mengindikasikan mayoritas aktivitas membakar kalori dalam jumlah rendah hingga sedang (di bawah 100 kcal)."


elif eda_option == "Duration vs Calories":
    plt.title("Perbandingan Duration vs Calories")
    sns.scatterplot(x=df['Duration'], y=df['Calories'], ax=ax)
    insight = "Korelasi positif yang sangat kuat dan berbentuk linier. Semakin lama durasi latihan, semakin besar kalori yang terbakar secara konstan."

elif eda_option == "Heart Rate vs Calories":
    plt.title("Perbandingan Heart Rate vs Calories")
    sns.scatterplot(x=df["Heart_Rate"], y=df["Calories"], ax=ax)
    insight = "Korelasi sangat kuat. Detak jantung yang tinggi menunjukkan intensitas latihan yang berat, sehingga kalori terbakar meningkat secara eksponensial."

elif eda_option == "Age vs Calories":
    plt.title("Perbandingan Age vs Calories")
    sns.scatterplot(x=df["Age"], y=df["Calories"], ax=ax)
    insight = "Pola menyebar rata. Usia tidak memiliki pengaruh linear langsung terhadap jumlah kalori yang terbakar secara individual."

elif eda_option ==  "Gender vs Calories":
    plt.title("Perbandingan Gender vs Calories")
    sns.boxplot(x=df["Gender"], y=df["Calories"], ax=ax)
    insight = "Distribusi kalori yang terbakar antara Laki-laki (Male) dan Perempuan (Female) cenderung mirip, namun rentang maksimum kalori laki-laki sedikit lebih tinggi."

elif eda_option ==  "Height vs Calories":
    plt.title("Perbandingan Height vs Calories")
    sns.scatterplot(x=df["Height"], y=df["Calories"],  ax=ax)
    insight = "Pola distribusi tersebar, menunjukkan tinggi badan tidak secara langsung menentukan kalori yang terbakar tanpa dipengaruhi durasi latihan."

elif eda_option ==  "Weight vs Calories":
    plt.title("Perbandingan Weight vs Calories")
    sns.scatterplot(x=df["Weight"], y=df["Calories"],  ax=ax)
    insight = "Pola mirip dengan Tinggi Badan. Berat badan sendiri tidak memiliki korelasi linear yang kuat secara langsung terhadap pembakaran kalori."

st.pyplot(fig, use_container_width=True)
st.info(f"**💡 Insight Utama:** {insight}")
st.markdown("---")

if st.checkbox("Tampilkan Correlation Heatmap"):
    fig, ax = plt.subplots(figsize=(15,5))
    corr_df = df.copy()
    if "Gender" in corr_df.columns and not pd.api.types.is_numeric_dtype(corr_df["Gender"]): 
        corr_df["Gender"] = corr_df["Gender"].astype(str).str.strip().str.lower().map({"male" : 0 , "female" : 1})

    sns.heatmap(corr_df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig, use_container_width=True) 
    st.markdown("""
    **Cara membaca nilai korelasi:**
    * Nilai mendekati **1.0**: Korelasi positif kuat (jika A naik, B ikut naik). Contoh: *Duration* & *Calories* (0.96).
    * Nilai mendekati **0.0**: Tidak ada hubungan linear. Contoh: *Age* & *Height* (0.01).
    * Nilai mendekati **-1.0**: Korelasi negatif kuat (jika A naik, B turun).
    """)