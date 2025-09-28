import streamlit as st
import pandas as pd

st.title("Streamlit text input")

name = st.text_input("Enter your name:")

if name:
    st.write(f"Hello {name}!")
else:
    st.write("Please enter your name.")

age = st.slider("Enter your age:", min_value=0, max_value=100, value=25) ## Default value is 25

st.write(f"You are {age} years old.")

gender = st.selectbox("Select your gender:", ["Male", "Female", "Other"])

st.write(f"You are {gender}.")

df = pd.DataFrame({
    'name': [name],
    'age': [age],
    'gender': [gender]
})

st.write(df)

uploaded_file = st.file_uploader("Upload a file:")## Used to upload a file from local machine

if uploaded_file:
    st.write(f"File uploaded: {uploaded_file.name}")
else:
    st.write("No file uploaded.")

