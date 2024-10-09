F1 Driver Final Position Prediction Model

This project is designed to predict the final position of an F1 driver in a race using machine learning. The model takes into account several key factors, including grid position, driver attributes, constructor data, and race conditions, to make its predictions.

Dataset Overview

The dataset contains information from past F1 races, with features like:

	•	Grid Position
	•	Driver Nationality and Age
	•	Constructor Name
	•	Circuit Location and Weather Conditions
	•	Lap Times and Pit Stops

Key Features of the Model

	•	Random Forest Classifier: Chosen for its strong performance on structured data.
	•	PCA: Used to reduce the feature space for optimized performance.
	•	One-Hot Encoding: Applied to categorical variables like driver nationality and constructor name.

Deployment

The model is deployed using Streamlit, allowing users to input race details and predict the driver’s final position.

Installation

	1.	Clone the repository.
	2.	Install required dependencies: pip install -r requirements.txt.
	3.	Run the app locally with streamlit run app.py.

Future Improvements

	•	Explore more advanced algorithms.
	•	Tuning hyperparameters for enhanced accuracy.
 
