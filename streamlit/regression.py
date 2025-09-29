# =====================================================
# SIMPLE LINEAR REGRESSION DEMO WITH STREAMLIT
# =====================================================
# This code demonstrates how to create an interactive web application
# that uses machine learning to predict test scores based on study hours

# Import necessary libraries:
import streamlit as st          # Streamlit: Framework for creating interactive web apps with Python
import numpy as np             # NumPy: Library for numerical operations and array handling
from sklearn.linear_model import LinearRegression  # Scikit-learn: Machine learning library

# =====================================================
# STEP 1: TRAIN THE MACHINE LEARNING MODEL
# =====================================================
# We train the model once when the app starts, using a small dataset
# This is called "supervised learning" because we have labeled training data

# Input features (X): Study hours - this is what we'll use to make predictions
# Each row represents one student's study time
X = np.array([[1], [2], [3], [4], [5]])   # Study hours for 5 students

# Target variable (y): Test scores - this is what we're trying to predict
# Each score corresponds to the study hours above
y = np.array([40, 50, 60, 70, 80])        # Test scores for each student

# Create and train the linear regression model
# Linear regression finds the best straight line that fits our data points
model = LinearRegression().fit(X, y)
# After training, the model has learned the relationship: score = slope * hours + intercept

# =====================================================
# STEP 2: CREATE THE INTERACTIVE WEB INTERFACE
# =====================================================

# Display the main title of our web application
st.title("Simple Supervised Regression Demo")

# Display a description of what this demo does
# The triple quotes allow us to write multi-line text with formatting
st.write("""
This demo shows a **Linear Regression** model trained on a tiny dataset
of *study hours* vs *test scores*.

**How it works:**
1. We trained a model on 5 data points (study hours → test scores)
2. Use the slider to input different study hours
3. The model predicts what test score you'd get
4. The model parameters show the mathematical relationship it learned
""")

# =====================================================
# STEP 3: GET USER INPUT WITH AN INTERACTIVE SLIDER
# =====================================================

# Create an interactive slider widget for user input
# Users can slide to select how many hours they studied
hours = st.slider(
    "Hours studied",           # Label shown to the user
    min_value=0.0,            # Minimum allowed value (0 hours)
    max_value=10.0,           # Maximum allowed value (10 hours)
    step=0.5                  # How much the value changes with each slide (0.5 hours)
)

# =====================================================
# STEP 4: MAKE A PREDICTION USING THE TRAINED MODEL
# =====================================================

# Use our trained model to predict the test score
# We pass the hours value in the format the model expects: [[hours]]
# The model returns an array, so we take the first (and only) value with [0]
predicted_score = model.predict([[hours]])[0]

# Display the prediction result to the user
# f-string formatting allows us to insert the variables into the text
st.write(f"**Predicted score for {hours} hours of study:** {predicted_score:.2f}")

# =====================================================
# STEP 5: SHOW THE MODEL'S LEARNED PARAMETERS
# =====================================================

# Display the mathematical parameters the model learned
# These show the equation: predicted_score = coef * hours + intercept
st.caption(f"Slope (coef): {model.coef_[0]:.2f}, Intercept: {model.intercept_:.2f}")

# EXPLANATION OF PARAMETERS:
# - Slope (coefficient): How much the score increases for each additional hour of study
# - Intercept: The predicted score when study hours = 0
# - Example: If slope = 10 and intercept = 30, then score = 10 * hours + 30
