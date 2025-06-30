import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model_forecast import train_models, predict_cases, plot_forecast


# Cache model loading for performance
@st.cache_resource
def load_models():
    return train_models("yourfile.csv")

st.set_page_config(page_title="G2D Leprosy Forecast", layout="centered")
st.title("📊 G2D Leprosy Forecasting by Country")

# Load trained models
trained_models, poly_transformers, year_ranges, training_data = load_models()
countries = list(trained_models.keys())

# Sidebar inputs
st.sidebar.header("Prediction Controls")
country = st.sidebar.selectbox("Select Country", countries)
year = st.sidebar.slider("Select Year", 2023, 2035, 2024)

# Forecast
model = trained_models[country]
poly = poly_transformers[country]
min_year, max_year = year_ranges[country]
df_yearly = training_data[country]

# Predict and Plot
from numpy import exp
from sklearn.preprocessing import PolynomialFeatures

X_input = poly.transform(pd.DataFrame([[year]], columns=['Period']))
log_pred = model.predict(X_input)[0]
prediction = int(round(exp(log_pred)))

st.subheader(f"📍 Prediction for **{country}** in **{year}**")
st.success(f"Estimated new G2D Leprosy Cases: **{prediction:,}**")

# Show plot
import numpy as np
X = df_yearly[['Period']]
y = df_yearly['FactValueNumeric']

future_years = pd.DataFrame({'Period': list(range(X['Period'].max() + 1, year + 1))}) if year > X['Period'].max() else pd.DataFrame()
all_years = pd.concat([X, future_years])
X_all_poly = poly.transform(all_years)
log_preds = model.predict(X_all_poly)
preds = np.exp(log_preds)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(X['Period'], y, label="Historical", marker='o')
ax.plot(all_years['Period'], preds, label="Forecast", linestyle='--', marker='x')
ax.axvline(x=year, color='red', linestyle=':', label='Prediction Year')
ax.set_title(f"G2D Leprosy Cases Forecast - {country}")
ax.set_xlabel("Year")
ax.set_ylabel("Cases")
ax.grid(True)
ax.legend()
st.pyplot(fig)
