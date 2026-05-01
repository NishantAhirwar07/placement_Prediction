import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Page Config
st.set_page_config(
    page_title="Placement Prediction App",
    page_icon="🎓",
    layout="centered"
)

# Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("placement.csv")

df = load_data()

# Train Model
X = df[["cgpa"]]
y = df["package"]

model = LinearRegression()
model.fit(X, y)

# Custom Styling
st.markdown("""
<style>
.big-title {
    font-size:40px;
    font-weight:700;
    margin-bottom:5px;
}
.small-text {
    color: #cfcfcf;
    font-size:15px;
    margin-bottom:25px;
}
.stButton>button {
    width: 180px;
    border-radius:8px;
    height:45px;
    font-size:16px;
}
.result-box {
    background-color:#0d5c2f;
    padding:18px;
    border-radius:10px;
    color:#8dffb1;
    font-size:18px;
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="big-title">🎓 Placement Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="small-text">Predict expected placement package based on CGPA using Linear Regression.</div>', unsafe_allow_html=True)

# Dataset Preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Input
cgpa = st.number_input(
    "Enter CGPA",
    min_value=0.0,
    max_value=10.0,
    value=7.2,
    step=0.1
)

# Prediction
if st.button("Predict Package"):
    prediction = model.predict(np.array([[cgpa]]))[0]

    st.markdown(
        f'<div class="result-box">Estimated Package: {prediction:.2f} LPA</div>',
        unsafe_allow_html=True
    )
