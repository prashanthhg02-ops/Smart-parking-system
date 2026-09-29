# Smart-parking-system
# Parkline Smart Parking

A small smart-parking dashboard that predicts lot occupancy for a selected arrival time, weather forecast, and nearby event. It recommends a facility and lets you create or cancel demo reservations.

## Run locally

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## How the prediction works

The app trains a scikit-learn `RandomForestRegressor` on a reproducible synthetic history of hourly occupancy. Hour, weekday, facility, rain, event status, and weekend status are model inputs. The training data and current availability are illustrative; connect real parking sensors and operational history before using predictions for real-world decisions. Reservations are kept in Streamlit session state and are not persisted between sessions.
