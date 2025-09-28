import streamlit as st
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def load_data():
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    return df, wine.target_names

df, target_names = load_data()

model = RandomForestClassifier() ## Random Forest Classifier is a machine learning algorithm that is used to classify data into categories.

# Train the model using only the 2 features we're collecting: alcohol and proline
model.fit(df[['alcohol', 'proline']], df['target'])

st.sidebar.title("Input features")

alcohol = st.sidebar.slider("Alcohol", float(df['alcohol'].min()), float(df['alcohol'].max()), float(df['alcohol'].mean()))

proline = st.sidebar.slider("Proline", float(df['proline'].min()), float(df['proline'].max()), float(df['proline'].mean()))

input_data = [[alcohol, proline]]

prediction = model.predict(input_data)
predicted_class = prediction[0]

# Get prediction probabilities for more detailed output
prediction_proba = model.predict_proba(input_data)
confidence_scores = prediction_proba[0]

# Display results in main area
st.title("Wine Classification Prediction")
st.write("### Input Values:")
st.write(f"**Alcohol:** {alcohol:.2f}")
st.write(f"**Proline:** {proline:.2f}")

st.write("### Prediction Results:")
st.write(f"**Predicted Wine Type:** {target_names[predicted_class]}")

# Show confidence scores
st.write("**Confidence Scores:**")
for i, (class_name, confidence) in enumerate(zip(target_names, confidence_scores)):
    st.write(f"- {class_name}: {confidence:.3f} ({confidence*100:.1f}%)")

# Also show in sidebar for quick reference
st.sidebar.write(f"**Prediction:** {target_names[predicted_class]}")
st.sidebar.write(f"**Confidence:** {confidence_scores[predicted_class]:.1%}")



