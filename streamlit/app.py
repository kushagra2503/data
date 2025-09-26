import streamlit as st
import pandas as pd
import numpy as np

# Web app using streamlit
st.title("Hello World") ## Display a title

st.write("This is a test") ## Display a text

df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
})

st.write("DataFrame: ", df)

chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c'])

st.line_chart( chart_data)
