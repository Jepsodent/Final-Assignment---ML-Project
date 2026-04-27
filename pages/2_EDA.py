import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Exploratory Data Analysis (EDA)")


df = pd.read_csv("data/calories.csv")
st.subheader("Missing Values: ")
missing = df.isnull().sum().reset_index()
missing.columns = ["Name", "Null Value"]
st.dataframe(missing)




st.subheader("Dataset Descriptive Statistic")
st.dataframe(df.describe())

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
elif eda_option == "Duration vs Calories":
    plt.title("Perbandingan Duration vs Calories")
    sns.scatterplot(x=df['Duration'], y=df['Calories'], ax=ax)
elif eda_option == "Heart Rate vs Calories":
    plt.title("Perbandingan Heart Rate vs Calories")
    sns.scatterplot(x=df["Heart_Rate"], y=df["Calories"], ax=ax)
elif eda_option == "Age vs Calories":
    plt.title("Perbandingan Age vs Calories")
    sns.scatterplot(x=df["Age"], y=df["Calories"], ax=ax)
elif eda_option ==  "Gender vs Calories":
    plt.title("Perbandingan Gender vs Calories")
    sns.boxplot(x=df["Gender"], y=df["Calories"], ax=ax)
elif eda_option ==  "Height vs Calories":
    plt.title("Perbandingan Height vs Calories")
    sns.scatterplot(x=df["Height"], y=df["Calories"],  ax=ax)
elif eda_option ==  "Weight vs Calories":
    plt.title("Perbandingan Weight vs Calories")
    sns.scatterplot(x=df["Weight"], y=df["Calories"],  ax=ax)

st.pyplot(fig, use_container_width=False)


if st.checkbox("Tampilkan Correlation Heatmap"):
    fig, ax = plt.subplots(figsize=(15,5))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)