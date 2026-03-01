"""
Build System - Orchestrator
Trains XGBoost model, fits scaler, calculates percentiles, and saves artifacts
"""
import pandas as pd
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing import DataPreprocessor
from src.surge_pricing import SurgePricingEngine


def build_system():
    """Complete pipeline: load data, preprocess, train model, save artifacts"""
    print("Starting Dynamic Pricing System Build...")

    # 1. Load Data
    print("\nLoading data...")
    df = pd.read_csv('data/dynamic_pricing.csv')
    print(f"Loaded {len(df)} records with columns: {list(df.columns)}")

    # 2. Initialize components
    preprocessor = DataPreprocessor()
    surge_engine = SurgePricingEngine()

    # 3. Calculate percentiles for surge pricing
    print("\nCalculating percentiles for surge pricing...")
    surge_engine.calculate_percentiles(df)
    print(f"   Riders  p30/p70 : {surge_engine.percentiles['riders_p30']:.1f} / {surge_engine.percentiles['riders_p70']:.1f}")
    print(f"   Drivers p30/p70 : {surge_engine.percentiles['drivers_p30']:.1f} / {surge_engine.percentiles['drivers_p70']:.1f}")

    # 4. Apply surge pricing to create target variable
    print("\nApplying surge pricing algorithm...")
    df = surge_engine.apply_surge_pricing(df)
    print(f"Surge pricing applied.")
    print(f"   Avg multiplier : {df['surge_multiplier'].mean():.3f}")
    print(f"   Price range    : Rs.{df['adjusted_ride_cost'].min():.2f} - Rs.{df['adjusted_ride_cost'].max():.2f}")

    # 5. Encode categorical features
    print("\nEncoding features...")
    df_encoded = preprocessor.encode_features(df)

    # 6. Prepare features and target
    drop_cols = ['adjusted_ride_cost', 'Historical_Cost_of_Ride', 'surge_multiplier']
    if 'Average_Ratings' in df_encoded.columns:
        drop_cols.append('Average_Ratings')

    X = df_encoded.drop(drop_cols, axis=1)
    y = df_encoded['adjusted_ride_cost']
    print(f"Features: {list(X.columns)}")

    # 7. Split data
    print("\nSplitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

    # 8. Fit scaler
    print("\nFitting scaler...")
    preprocessor.fit_scaler(X_train)
    X_train_scaled = preprocessor.transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # 9. Train XGBoost model
    print("\nTraining XGBoost Regressor...")
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train_scaled, y_train,
              eval_set=[(X_test_scaled, y_test)],
              verbose=False)

    # 10. Evaluate
    train_preds = model.predict(X_train_scaled)
    test_preds = model.predict(X_test_scaled)
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)
    test_mae = mean_absolute_error(y_test, test_preds)

    print(f"Training R2 : {train_r2:.4f}")
    print(f"Test R2     : {test_r2:.4f}")
    print(f"Test MAE    : Rs.{test_mae:.2f}")

    # 11. Save artifacts
    print("\nSaving artifacts...")
    os.makedirs('models', exist_ok=True)

    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("XGBoost model saved  -> models/model.pkl")

    preprocessor.save_scaler('models/scaler.pkl')
    print("Scaler saved         -> models/scaler.pkl")

    with open('models/surge_engine.pkl', 'wb') as f:
        pickle.dump(surge_engine, f)
    print("Surge engine saved   -> models/surge_engine.pkl")

    print("\n" + "-"*50)
    print("BUILD COMPLETE - All artifacts ready!")
    print("-"*50)
    print(f"   Model    : XGBoost Regressor")
    print(f"   Features : {len(X.columns)}")
    print(f"   Test R2  : {test_r2:.4f}")
    print(f"   Test MAE : Rs.{test_mae:.2f}")
    print("\n   Run: uvicorn api.app:app --reload")
    print("-"*50)


if __name__ == "__main__":
    build_system()