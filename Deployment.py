import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load our AI fifa trained model
with open("Fifa_prediction.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit application
st.title("AI FIFA Rating Model")

st.sidebar.header("Input Features")

# Create input boxes for "Value (EUR)," "Wage (EUR)," and "Release Clause (EUR)"
value_eur = st.sidebar.number_input(
    "Value (EUR)", min_value=0, max_value=200000000, value=10000000
)
wage_eur = st.sidebar.number_input(
    "Wage (EUR)", min_value=0, max_value=1000000, value=50000
)
release_clause_eur = st.sidebar.number_input(
    "Release Clause (EUR)", min_value=0, max_value=200000000, value=10000000
)

# Create sliders for the remaining features
potential = st.sidebar.slider("Potential", min_value=0, max_value=100, value=50)
passing = st.sidebar.slider("Passing", min_value=0, max_value=100, value=50)
dribbling = st.sidebar.slider("Dribbling", min_value=0, max_value=100, value=50)
attacking_short_passing = st.sidebar.slider(
    "Attacking Short Passing", min_value=0, max_value=100, value=50
)
movement_reactions = st.sidebar.slider(
    "Movement Reactions", min_value=0, max_value=100, value=50
)
power_shot_power = st.sidebar.slider(
    "Power Shot Power", min_value=0, max_value=100, value=50
)
mentality_vision = st.sidebar.slider(
    "Mentality Vision", min_value=0, max_value=100, value=50
)
mentality_composure = st.sidebar.slider(
    "Mentality Composure", min_value=0, max_value=100, value=50
)

# Predict the player's overall rating
# Combine input features into an array
input_data = np.array(
    [
        [
            potential,
            value_eur,
            wage_eur,
            release_clause_eur,
            passing,
            dribbling,
            attacking_short_passing,
            movement_reactions,
            power_shot_power,
            mentality_vision,
            mentality_composure,
        ]
    ]
).astype(float)

if st.sidebar.button("Predict Overall"):
    # Apply the same scaler used during training to the scaled features
    #scaled_features = scaler.transform(input_data)

    # Predict the overall rating
    predicted_rating = model.predict(input_data)[0]
    rounded_rating = round(predicted_rating)
 

    st.sidebar.subheader("Predicted Overall Rating:")
    st.sidebar.write(rounded_rating)

# Streamlit run command
if __name__ == "__main__":
    st.write(
        "Use the sidebar to input player attributes and see the predicted overall rating."
    )
