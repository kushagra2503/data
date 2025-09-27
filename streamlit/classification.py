# Import necessary libraries for the machine learning web app
import streamlit as st  # Streamlit: Framework for creating web applications with Python
import pandas as pd    # Pandas: Library for data manipulation and analysis
from sklearn.datasets import load_iris  # Scikit-learn dataset: Contains the famous Iris flower dataset
from sklearn.ensemble import RandomForestClassifier  # Machine learning algorithm: Random Forest for classification

# Decorator to cache the function results - this makes the app faster by avoiding reloading data
@st.cache_data
def load_data():
    """
    Function to load the Iris dataset and prepare it for machine learning.

    Returns:
        df: DataFrame containing flower measurements and species labels
        target_names: Array of species names (setosa, versicolor, virginica)
    """
    # Load the Iris dataset - this contains measurements of 150 iris flowers
    iris = load_iris()

    # Create a pandas DataFrame from the dataset with proper column names
    # iris.data contains the measurements (sepal length/width, petal length/width)
    # iris.feature_names contains the column names for these measurements
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    # Add the target variable (species) as a new column to our DataFrame
    # iris.target contains numeric labels (0, 1, 2) representing each species
    df['species'] = iris.target  # 0=setosa, 1=versicolor, 2=virginica

    # Return both the DataFrame and the species names
    return df, iris.target_names

# Call the load_data function to get our dataset and species names
# df contains all the flower measurements and species labels
# target_names contains the actual species names as strings
df, target_names = load_data()

# Create a Random Forest classifier - this is a machine learning algorithm
# Random Forest works by creating many decision trees and combining their results
model = RandomForestClassifier()

# Train the model using our data
# df.iloc[:, :-1] selects all columns except the last one (species) - these are our input features
# df['species'] is our target variable - what we want to predict
model.fit(df.iloc[:, :-1], df['species'])

# Create the user interface in the sidebar for inputting flower measurements
# The sidebar is a column on the left side of the Streamlit app
st.sidebar.title("Input features")

# Create interactive sliders for each flower measurement
# Each slider has: label, minimum value, maximum value, and default (starting) value
# We use the actual min/max/mean values from our dataset to set realistic ranges
sepal_length = st.sidebar.slider(
    "Sepal length",  # Label shown to user
    float(df['sepal length (cm)'].min()),  # Minimum value (smallest sepal length in dataset)
    float(df['sepal length (cm)'].max()),  # Maximum value (largest sepal length in dataset)
    float(df['sepal length (cm)'].mean())  # Default value (average sepal length)
)

sepal_width = st.sidebar.slider(
    "Sepal width",  # Label shown to user
    float(df['sepal width (cm)'].min()),   # Minimum value
    float(df['sepal width (cm)'].max()),   # Maximum value
    float(df['sepal width (cm)'].mean())   # Default value (average)
)

petal_length = st.sidebar.slider(
    "Petal length",  # Label shown to user
    float(df['petal length (cm)'].min()),  # Minimum value
    float(df['petal length (cm)'].max()),  # Maximum value
    float(df['petal length (cm)'].mean())  # Default value (average)
)

petal_width = st.sidebar.slider(
    "Petal width",  # Label shown to user
    float(df['petal width (cm)'].min()),   # Minimum value
    float(df['petal width (cm)'].max()),   # Maximum value
    float(df['petal width (cm)'].mean())   # Default value (average)
)

# Combine all the user's input values into a single list (2D array format expected by the model)
# The model was trained on data in this format: [[feature1, feature2, feature3, feature4]]
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

# Use our trained model to make a prediction based on the user's input
# The model will return a number (0, 1, or 2) representing the predicted species
prediction = model.predict(input_data)

# Convert the numeric prediction to the actual species name
# prediction[0] gets the first (and only) prediction from the array
# target_names contains: ['setosa', 'versicolor', 'virginica']
predicted_species = target_names[prediction][0]

# Display the prediction result to the user in the main area of the app
st.write("Prediction: ", predicted_species)
