import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Load model and PCA transformer
model = joblib.load('best_model.pkl')
pca = joblib.load('pca_transformer.pkl')
scaler = joblib.load('scaler.pkl') 

# Define the feature names used in training the model
feature_names = ['grid_position', 'driver_age_atrace', 'average_lap_time',
       'total_pit_stops', 'driver_nationality_Australian',
       'driver_nationality_Belgian', 'driver_nationality_Brazilian',
       'driver_nationality_British', 'driver_nationality_Canadian',
       'driver_nationality_Chinese', 'driver_nationality_Danish',
       'driver_nationality_Dutch', 'driver_nationality_Finnish',
       'driver_nationality_French', 'driver_nationality_German',
       'driver_nationality_Indonesian', 'driver_nationality_Italian',
       'driver_nationality_Japanese', 'driver_nationality_Mexican',
       'driver_nationality_Monegasque', 'driver_nationality_New Zealander',
       'driver_nationality_Polish', 'driver_nationality_Russian',
       'driver_nationality_Spanish', 'driver_nationality_Swedish',
       'driver_nationality_Thai', 'constructor_name_AlphaTauri',
       'constructor_name_Alpine F1 Team', 'constructor_name_Aston Martin',
       'constructor_name_Ferrari', 'constructor_name_Force India',
       'constructor_name_Haas F1 Team', 'constructor_name_Manor Marussia',
       'constructor_name_McLaren', 'constructor_name_Mercedes',
       'constructor_name_Racing Point', 'constructor_name_Red Bull',
       'constructor_name_Renault', 'constructor_name_Sauber',
       'constructor_name_Toro Rosso', 'constructor_name_Williams',
       'circuit_location_Al Daayen', 'circuit_location_Austin',
       'circuit_location_Baku', 'circuit_location_Budapest',
       'circuit_location_Hockenheim', 'circuit_location_Imola',
       'circuit_location_Istanbul', 'circuit_location_Jeddah',
       'circuit_location_Kuala Lumpur', 'circuit_location_Le Castellet',
       'circuit_location_Marina Bay', 'circuit_location_Melbourne',
       'circuit_location_Mexico City', 'circuit_location_Miami',
       'circuit_location_Monte-Carlo', 'circuit_location_Montmeló',
       'circuit_location_Montreal', 'circuit_location_Monza',
       'circuit_location_Mugello', 'circuit_location_Nürburg',
       'circuit_location_Portimão', 'circuit_location_Sakhir',
       'circuit_location_Shanghai', 'circuit_location_Silverstone',
       'circuit_location_Sochi', 'circuit_location_Spa',
       'circuit_location_Spielberg', 'circuit_location_Suzuka',
       'circuit_location_São Paulo', 'circuit_location_Zandvoort',
       'weather_conditions_Overcast', 'weather_conditions_Sunny',
       'weather_conditions_Wet']

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

# One-hot encode categorical variables
input_data_encoded = pd.get_dummies(input_data, columns=['driver_nationality', 'constructor_name', 'circuit_location', 'weather_conditions'])

# Ensure the input data has the same columns as during training
input_data_encoded = input_data_encoded.reindex(columns=feature_names, fill_value=0)

# Standardize the input data (ensure this matches the training procedure)
input_data_scaled = scaler.transform(input_data_encoded)

# Apply PCA transformation
input_data_pca = pca.transform(input_data_scaled)

if st.button('Predict Final Position'):
    prediction_proba = model.predict_proba(input_data_pca)
    predicted_class = np.argmax(prediction_proba)
    confidence = prediction_proba[0, predicted_class]

    # Show prediction and confidence
    st.write(f'The predicted final position is: {predicted_class} with a confidence of {confidence}')




