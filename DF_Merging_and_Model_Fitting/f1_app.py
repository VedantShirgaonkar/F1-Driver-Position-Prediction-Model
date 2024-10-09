import streamlit as st
import pandas as pd
import joblib

# Load the trained model pipeline
pipeline = joblib.load('/Users/rahul/Desktop/F1 Project/DF_Merging_and_Model_Fitting/f1_pipeline.pkl')

# Streamlit App Interface
st.title("F1 Driver Position Prediction")

# Collect user input for new predictions
st.header("Input Parameters")

def user_input_features():
    grid_position = st.number_input("Grid Position", min_value=1, max_value=20)
    driver_age_atrace = st.number_input("Driver Age at Race", min_value=18, max_value=50)
    average_lap_time = st.number_input("Average Lap Time (seconds)")
    total_pit_stops = st.number_input("Total Pit Stops", min_value=0)
    
    # Handling categorical inputs (you can expand this with other one-hot encoded features as needed)
    driver_home = st.selectbox("Driver Home", ['Germany', 'UK', 'Australia', 'Brazil', 'France', 'Finland',
       'Spain', 'Netherlands', 'Denmark', 'Mexico', 'Sweden', 'Indonesia',
       'Russia', 'Belgium', 'Italy', 'Canada', 'New Zealand', 'Monaco',
       'Thailand', 'Poland', 'Japan', 'China', 'USA'])
    constructor_name = st.selectbox("Constructor", ['Mercedes', 'Ferrari', 'Red Bull', 'Williams', 'Haas F1 Team',
       'Force India', 'Toro Rosso', 'Renault', 'McLaren', 'Sauber',
       'Manor Marussia', 'Alfa Romeo', 'Racing Point', 'AlphaTauri',
       'Aston Martin', 'Alpine F1 Team'])
    circuit_location = st.selectbox("Circuit Location", ['Melbourne', 'Sakhir', 'Shanghai', 'Sochi', 'Montmeló',
       'Monte-Carlo', 'Montreal', 'Baku', 'Spielberg', 'Silverstone',
       'Budapest', 'Spa', 'Monza', 'Marina Bay', 'Kuala Lumpur', 'Suzuka',
       'Austin', 'Mexico City', 'São Paulo', 'Abu Dhabi', 'Le Castellet',
       'Hockenheim', 'Mugello', 'Nürburg', 'Portimão', 'Imola',
       'Istanbul', 'Al Daayen', 'Zandvoort', 'Jeddah', 'Miami'])
    weather_conditions = st.selectbox("Weather Conditions", ['Overcast', 'Dry', 'Sunny', 'Wet'])

    data = {'grid_position': grid_position,
            'driver_age_atrace': driver_age_atrace,
            'average_lap_time': average_lap_time,
            'total_pit_stops': total_pit_stops,
            'driver_home_' + driver_home: 1,
            'constructor_name_' + constructor_name: 1,
            'circuit_location_' + circuit_location: 1,
            'weather_conditions_' + weather_conditions: 1}

    return pd.DataFrame([data])

# Get user input
input_df = user_input_features()

# Ensure that all columns from training are included, filling missing ones with 0
input_df = input_df.reindex(columns=pipeline.named_steps['classifier'].feature_importances_, fill_value=0)

# Predict using the pipeline
prediction = pipeline.predict(input_df)

# Display prediction
st.subheader('Predicted Final Position')
st.write(prediction)