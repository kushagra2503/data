import streamlit as st

st.title("Streamlit text input")

name = st.text_input("Enter your name:")

if name:
    st.write(f"Hello {name}!")
else:
    st.write("Please enter your name.")