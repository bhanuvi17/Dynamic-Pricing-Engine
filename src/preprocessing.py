"""
Data preprocessing module for dynamic pricing system
Handles encoding and scaling of features
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.num_cols = ['Number_of_Riders', 'Number_of_Drivers',
                         'Number_of_Past_Rides', 'Expected_Ride_Duration']

    def encode_features(self, df):
        """Encode categorical features"""
        df = df.copy()

        # Map Vehicle Type
        df['Vehicle_Type'] = df['Vehicle_Type'].map({'Economy': 0, 'Premium': 1})

        # One-hot encode categorical features
        df = pd.get_dummies(df, columns=['Location_Category', 'Customer_Loyalty_Status',
                                         'Time_of_Booking'], drop_first=True)

        # Convert boolean columns to int
        bool_cols = [
            'Location_Category_Suburban', 'Location_Category_Urban',
            'Customer_Loyalty_Status_Regular', 'Customer_Loyalty_Status_Silver',
            'Time_of_Booking_Evening', 'Time_of_Booking_Morning',
            'Time_of_Booking_Night'
        ]

        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)

        return df

    def fit_scaler(self, df):
        """Fit scaler on training data"""
        self.scaler.fit(df[self.num_cols])
        return self

    def transform(self, df):
        """Scale numerical features"""
        df = df.copy()

        expected_cols = [
            'Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides',
            'Expected_Ride_Duration', 'Vehicle_Type',
            'Location_Category_Suburban', 'Location_Category_Urban',
            'Customer_Loyalty_Status_Regular', 'Customer_Loyalty_Status_Silver',
            'Time_of_Booking_Evening', 'Time_of_Booking_Morning',
            'Time_of_Booking_Night'
        ]

        # Add missing columns with 0
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0

        df = df[expected_cols]
        df[self.num_cols] = self.scaler.transform(df[self.num_cols])
        return df

    def save_scaler(self, path):
        """Save fitted scaler"""
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)

    def load_scaler(self, path):
        """Load fitted scaler"""
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        return self
