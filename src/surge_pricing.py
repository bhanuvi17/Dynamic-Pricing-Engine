"""
Surge pricing algorithm module
Calculates dynamic multipliers based on demand/supply ratio and contextual factors
"""
import numpy as np
import pandas as pd


class SurgePricingEngine:
    def __init__(self):
        self.percentiles = {}

    def calculate_percentiles(self, df):
        """Calculate percentiles for reference thresholds"""
        self.percentiles = {
            'riders_p70':     np.percentile(df['Number_of_Riders'], 70),
            'riders_p30':     np.percentile(df['Number_of_Riders'], 30),
            'drivers_p70':    np.percentile(df['Number_of_Drivers'], 70),
            'drivers_p30':    np.percentile(df['Number_of_Drivers'], 30),
            'past_rides_p70': np.percentile(df['Number_of_Past_Rides'], 70),
            'past_rides_p30': np.percentile(df['Number_of_Past_Rides'], 30),
            'duration_p70':   np.percentile(df['Expected_Ride_Duration'], 70),
            'duration_p30':   np.percentile(df['Expected_Ride_Duration'], 30),
            'ratio_median':   np.median(df['Number_of_Riders'] / df['Number_of_Drivers'].clip(lower=1)),
        }
        return self

    def apply_surge_pricing(self, df):
        """Apply surge pricing to a DataFrame"""
        df = df.copy()

        # 1. Demand/Supply Ratio — core surge signal
        ratio = df['Number_of_Riders'] / df['Number_of_Drivers'].clip(lower=1)
        ratio_median = self.percentiles['ratio_median']
        # Above median ratio = surge, below = discount
        ds_multiplier = np.clip(ratio / ratio_median, 0.85, 1.25)

        # 2. Duration Multiplier — longer rides cost more
        duration_ratio = np.where(
            df['Expected_Ride_Duration'] > self.percentiles['duration_p70'],
            df['Expected_Ride_Duration'] / self.percentiles['duration_p70'],
            np.where(
                df['Expected_Ride_Duration'] < self.percentiles['duration_p30'],
                df['Expected_Ride_Duration'] / self.percentiles['duration_p30'],
                1.0
            )
        )
        duration_multiplier = np.clip(duration_ratio, 0.9, 1.2)

        # 3. Loyalty Multiplier — small discount for loyal customers (max 5%)
        past_ratio = np.where(
            df['Number_of_Past_Rides'] > self.percentiles['past_rides_p70'],
            0.97,   # loyal: 3% discount
            np.where(
                df['Number_of_Past_Rides'] < self.percentiles['past_rides_p30'],
                1.02,  # new user: slight premium
                1.0
            )
        )
        loyalty_multiplier = np.clip(past_ratio, 0.95, 1.05)

        # 4. Vehicle Type
        vehicle_multiplier = np.where(df['Vehicle_Type'] == 1, 1.25, 1.0)

        # 5. Location
        loc_col_urban    = df.get('Location_Category_Urban', pd.Series(0, index=df.index))
        loc_col_suburban = df.get('Location_Category_Suburban', pd.Series(0, index=df.index))
        loc_mult = np.where(loc_col_urban == 1, 1.10,
                   np.where(loc_col_suburban == 1, 1.05, 1.0))

        # 6. Time of Booking — peak hours surge, not discount
        eve  = df.get('Time_of_Booking_Evening', pd.Series(0, index=df.index))
        morn = df.get('Time_of_Booking_Morning', pd.Series(0, index=df.index))
        nigh = df.get('Time_of_Booking_Night',   pd.Series(0, index=df.index))
        time_mult = np.where(eve  == 1, 1.10,   # Evening: peak
                    np.where(morn == 1, 1.08,   # Morning: peak
                    np.where(nigh == 1, 1.05,   # Night: moderate surge (safety premium)
                    1.0)))                       # Afternoon: base

        combined = (
            ds_multiplier *
            duration_multiplier *
            loyalty_multiplier *
            vehicle_multiplier *
            loc_mult *
            time_mult
        )
        combined = np.clip(combined, 0.8, 1.5)

        df['adjusted_ride_cost'] = df['Historical_Cost_of_Ride'] * combined
        df['surge_multiplier'] = combined
        return df

    def calculate_single_surge(self, riders, drivers, past_rides, duration,
                               vehicle_type, location, time_booking, base_price):
        """Calculate surge for a single ride request"""

        # 1. Demand/Supply ratio
        ratio = riders / max(drivers, 1)
        ratio_median = self.percentiles['ratio_median']
        ds_mult = float(np.clip(ratio / ratio_median, 0.85, 1.25))

        # 2. Duration
        if duration > self.percentiles['duration_p70']:
            dur_mult = float(np.clip(duration / self.percentiles['duration_p70'], 0.9, 1.2))
        elif duration < self.percentiles['duration_p30']:
            dur_mult = float(np.clip(duration / self.percentiles['duration_p30'], 0.9, 1.2))
        else:
            dur_mult = 1.0

        # 3. Loyalty
        if past_rides > self.percentiles['past_rides_p70']:
            loy_mult = 0.97
        elif past_rides < self.percentiles['past_rides_p30']:
            loy_mult = 1.02
        else:
            loy_mult = 1.0

        # 4. Vehicle
        vehicle_mult = 1.25 if vehicle_type == 'Premium' else 1.0

        # 5. Location
        if location == 'Urban':
            loc_mult = 1.10
        elif location == 'Suburban':
            loc_mult = 1.05
        else:
            loc_mult = 1.0

        # 6. Time — all times either neutral or surge, never discount
        if time_booking == 'Evening':
            time_mult = 1.10
        elif time_booking == 'Morning':
            time_mult = 1.08
        elif time_booking == 'Night':
            time_mult = 1.05
        else:  # Afternoon
            time_mult = 1.0

        combined = ds_mult * dur_mult * loy_mult * vehicle_mult * loc_mult * time_mult
        combined = float(np.clip(combined, 0.8, 1.5))

        adjusted_price = base_price * combined
        return adjusted_price, combined