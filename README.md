# Dynamic Pricing Engine

A machine learning-powered ride pricing system that combines an **XGBoost regression model** with a **custom surge pricing algorithm** to predict dynamic ride fares based on real-time demand, supply, and contextual signals.

---

## How It Works

The system uses a two-stage pricing pipeline:

1. **ML Prediction** — XGBoost predicts a base price from ride features (riders, drivers, vehicle type, duration, location, time, loyalty status)
2. **Surge Multiplier** — A rule-based surge engine calculates a multiplier (0.8× – 1.4×) from demand/supply ratio and contextual factors, then applies it on top of the ML base price

```
Final Price = ML Base Price × Surge Multiplier
```

---

## Project Structure

```
Dynamic_pricing_Engine/
│
├── api/
│   └── app.py                  # FastAPI backend — prediction endpoint
│
├── data/
│   └── dynamic_pricing.csv     # Training dataset
│
├── models/                     # Auto-generated after build
│   ├── model.pkl               # Trained XGBoost model
│   ├── scaler.pkl              # Fitted StandardScaler
│   └── surge_engine.pkl        # Surge engine with fitted percentiles
│
├── src/
│   ├── preprocessing.py        # Feature encoding & scaling
│   ├── prediction.py           # Model loading & feature preparation
│   └── surge_pricing.py        # Surge algorithm (multiplier logic)
│
├── ui/
│   └── index.html              # Frontend — ride fare calculator
│
├── build_system.py             # Training pipeline — run this first
├── requirements.txt
└── README.md
```

---

## Surge Pricing Factors

| Factor | Effect |
|---|---|
| Riders / Drivers ratio | Core signal — above median triggers surge |
| Ride duration | Long rides → up to 1.2× boost |
| Vehicle type | Premium → 1.25× |
| Location | Urban 1.1×, Suburban 1.05×, Rural 1.0× |
| Time of booking | Evening 1.1×, Morning 1.08×, Night 1.05× |
| Past rides (loyalty) | High rides → 3% discount, new user → 2% premium |

Combined multiplier is clipped to **[0.8, 1.4]**.

---

## Setup & Usage

### 1. Clone the repo

```bash
git clone https://github.com/bhanuvi17/Dynamic-Pricing-Engine.git
cd Dynamic_pricing_Engine
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv dynamicenv
# Windows
dynamicenv\Scripts\activate
# macOS/Linux
source dynamicenv/bin/activate

pip install -r requirements.txt
```

### 3. Train the model

```bash
python build_system.py
```

This preprocesses the dataset, trains XGBoost, fits the scaler, calculates surge percentiles, and saves all three artifacts to `models/`.

### 4. Start the API server

```bash
uvicorn api.app:app --reload
```

Server runs at `http://127.0.0.1:8000`

Open `http://127.0.0.1:8000` in your browser to use the UI.

---

## API

### `POST /predict`

**Request body:**

```json
{
  "Number_of_Riders": 80,
  "Number_of_Drivers": 10,
  "Location_Category": "Urban",
  "Customer_Loyalty_Status": "Silver",
  "Number_of_Past_Rides": 20,
  "Time_of_Booking": "Evening",
  "Vehicle_Type": "Premium",
  "Expected_Ride_Duration": 90,
  "Historical_Cost_of_Ride": 350.00
}
```

**Response:**

```json
{
  "success": true,
  "base_price": 350.00,
  "ml_base_price": 362.45,
  "surge_multiplier": 1.386,
  "surge_adjusted_price": 485.10,
  "ml_surge_price": 502.36,
  "percent_change": 43.53
}
```

Interactive API docs available at `http://127.0.0.1:8000/docs`

---

## Dataset Features

| Column | Description |
|---|---|
| `Number_of_Riders` | Active ride requests |
| `Number_of_Drivers` | Available drivers |
| `Location_Category` | Urban / Suburban / Rural |
| `Customer_Loyalty_Status` | Gold / Silver / Regular |
| `Number_of_Past_Rides` | Rider's ride history |
| `Time_of_Booking` | Morning / Afternoon / Evening / Night |
| `Vehicle_Type` | Economy / Premium |
| `Expected_Ride_Duration` | In minutes |
| `Historical_Cost_of_Ride` | Base fare (Rs.) |

---

## Tech Stack

- **Python 3.10+**
- **XGBoost** — regression model
- **scikit-learn** — preprocessing & evaluation
- **FastAPI + Uvicorn** — REST API
- **Pandas / NumPy** — data processing
- **HTML / CSS / Vanilla JS** — frontend UI

---

## Requirements

See `requirements.txt` Key dependencies:

```
xgboost
scikit-learn
fastapi
uvicorn
pandas
numpy
pydantic
```
