import streamlit as st
import joblib
import pandas as pd

# Load model
pipeline = joblib.load('f1_pipeline.pkl')

# App Title
st.title('F1 Driver Final Position Predictor')

# Input fields for features
st.header('Enter Driver and Race Details')
grid_position = st.number_input('Grid Position', 1, 20, 10)
driver_age_atrace = st.number_input('Driver Age at Race', 18, 50, 28)
average_lap_time = st.number_input('Average Lap Time (seconds)', value=90.0)
total_pit_stops = st.number_input('Total Pit Stops', 0, 10, 2)

# Categorical features
driver_nationality = st.selectbox('Driver Nationality', ['German', 'British', 'Australian', 'Brazilian', 'French',
       'Finnish', 'Spanish', 'Dutch', 'Danish', 'Mexican', 'Swedish',
       'Indonesian', 'Russian', 'Belgian', 'Italian', 'Canadian',
       'New Zealander', 'Monegasque', 'Thai', 'Polish', 'Japanese',
       'Chinese', 'American'])
constructor_name = st.selectbox('Constructor Name', ['Mercedes', 'Ferrari', 'Red Bull', 'Williams', 'Haas F1 Team', 'Force India', 'Toro Rosso', 'Renault', 'McLaren', 'Sauber', 'Manor Marussia', 'Alfa Romeo', 'Racing Point', 'AlphaTauri', 'Aston Martin', 'Alpine F1 Team'])
circuit_location = st.selectbox('Circuit Location', ['Melbourne', 'Sakhir', 'Shanghai', 'Sochi', 'Montmeló', 'Monte-Carlo', 'Montreal', 'Baku', 'Spielberg', 'Silverstone', 'Budapest', 'Spa', 'Monza', 'Marina Bay', 'Kuala Lumpur', 'Suzuka', 'Austin', 'Mexico City', 'São Paulo', 'Abu Dhabi', 'Le Castellet', 'Hockenheim', 'Mugello', 'Nürburg', 'Portimão', 'Imola', 'Istanbul', 'Al Daayen', 'Zandvoort', 'Jeddah', 'Miami'])
weather_conditions = st.selectbox('Weather Conditions', ['Overcast', 'Dry', 'Sunny', 'Wet'])

# Prepare input data
input_data = pd.DataFrame({
    'grid_position': [grid_position],
    'driver_age_atrace': [driver_age_atrace],
    'average_lap_time': [average_lap_time],
    'total_pit_stops': [total_pit_stops],
    'driver_nationality': [driver_nationality],
    'constructor_name': [constructor_name],
    'circuit_location': [circuit_location],
    'weather_conditions': [weather_conditions]
})

if st.button('Predict Final Position'):
    predicted_class = pipeline.predict(input_data)
    prediction_proba = pipeline.predict_proba(input_data)
    confidence = prediction_proba[0, predicted_class]

    # Show prediction and confidence
    st.write(f'The predicted final position is: {predicted_class[0]} with a confidence of {confidence[0]:.2f}')