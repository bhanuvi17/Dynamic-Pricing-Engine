"""
Model prediction module
Handles model loading and price prediction
"""
import pickle
import pandas as pd
import numpy as np


class PricingPredictor:
    def __init__(self, model_path):
        """Initialize predictor with trained model"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, X):
        """Predict ride price"""
        return self.model.predict(X)

    def prepare_features(self, data_dict):
        """
        Prepare features from input dictionary for prediction.
        Ensures correct feature order matching training.
        """
        feature_cols = [
            'Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides',
            'Expected_Ride_Duration', 'Vehicle_Type',
            'Location_Category_Suburban', 'Location_Category_Urban',
            'Customer_Loyalty_Status_Regular', 'Customer_Loyalty_Status_Silver',
            'Time_of_Booking_Evening', 'Time_of_Booking_Morning',
            'Time_of_Booking_Night'
        ]

        features = {}

        # Numerical features
        features['Number_of_Riders'] = data_dict['Number_of_Riders']
        features['Number_of_Drivers'] = data_dict['Number_of_Drivers']
        features['Number_of_Past_Rides'] = data_dict['Number_of_Past_Rides']
        features['Expected_Ride_Duration'] = data_dict['Expected_Ride_Duration']

        # Vehicle Type
        features['Vehicle_Type'] = 1 if data_dict['Vehicle_Type'] == 'Premium' else 0

        # Location (one-hot)
        features['Location_Category_Suburban'] = 0
        features['Location_Category_Urban'] = 0
        if data_dict['Location_Category'] == 'Urban':
            features['Location_Category_Urban'] = 1
        elif data_dict['Location_Category'] == 'Suburban':
            features['Location_Category_Suburban'] = 1

        # Loyalty Status (one-hot)
        features['Customer_Loyalty_Status_Regular'] = 0
        features['Customer_Loyalty_Status_Silver'] = 0
        if data_dict['Customer_Loyalty_Status'] == 'Regular':
            features['Customer_Loyalty_Status_Regular'] = 1
        elif data_dict['Customer_Loyalty_Status'] == 'Silver':
            features['Customer_Loyalty_Status_Silver'] = 1

        # Time of Booking (one-hot)
        features['Time_of_Booking_Evening'] = 0
        features['Time_of_Booking_Morning'] = 0
        features['Time_of_Booking_Night'] = 0
        if data_dict['Time_of_Booking'] == 'Evening':
            features['Time_of_Booking_Evening'] = 1
        elif data_dict['Time_of_Booking'] == 'Morning':
            features['Time_of_Booking_Morning'] = 1
        elif data_dict['Time_of_Booking'] == 'Night':
            features['Time_of_Booking_Night'] = 1

        return pd.DataFrame([features], columns=feature_cols)
