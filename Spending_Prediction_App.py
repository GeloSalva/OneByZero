import streamlit as st
import pandas as pd
import pickle

# Set the title of the app
st.title("Cardholder Spending Prediction")

# Load the trained model from the pickle file
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)
    
# Load the CSV file
data = pd.read_csv('test1.csv')

# Load the CSV file with the "Cardholder" column
cardholders_data = pd.read_csv('cardholders.csv')

# Extract the "Cardholder" column from the DataFrame
cardholders = cardholders_data['Cardholder']

# Create a sidebar for user inputs
st.sidebar.title("User Inputs")

# Create a dropdown to select the cardholder
selected_cardholder = st.sidebar.selectbox('Select a cardholder', cardholders)

# Get the index of the selected cardholder
selected_index = cardholders_data[cardholders_data['Cardholder'] == selected_cardholder].index[0]

# Create a toggle button to select the number of days to predict
k_days = st.sidebar.slider('Select the number of days to predict', min_value=1, max_value=30)

# Add a "Predict" button
predict_button = st.sidebar.button('Predict')

# Check if the "Predict" button is clicked
if predict_button:
    # Get the features for the selected index
    features = data.loc[selected_index]

    # Make the prediction
    prediction = model.predict([features])[0]

    # Determine the prediction label
    if prediction == 0:
        prediction_label = "not expected to spend🚨"
    else:
        prediction_label = "expected to spend✅"

    # Display the prediction
    st.write(f"The cardholder {selected_cardholder} is {prediction_label} in the next {k_days} days.")
