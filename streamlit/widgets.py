import streamlit as st

st.title("Streamlit text input")

name = st.text_input("Enter your name:")

if name:
    st.write(f"Hello {name}!")
else:
    st.write("Please enter your name.")

age = st.slider("Enter your age:", min_value=0, max_value=100, value=25)

st.write(f"You are {age} years old.")

gender = st.selectbox("Select your gender:", ["Male", "Female", "Other"])

st.write(f"You are {gender}.")

