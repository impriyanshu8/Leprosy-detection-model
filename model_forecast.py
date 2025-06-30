import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Global containers
trained_models = {}
poly_transformers = {}
year_ranges = {}
training_data = {}


def train_models(csv_path):
    global trained_models, poly_transformers, year_ranges, training_data

    df = pd.read_csv(csv_path)
    df = df[['Location', 'Period', 'FactValueNumeric']].dropna()
    countries = df['Location'].unique()

    trained_models = {}
    poly_transformers = {}
    year_ranges = {}
    training_data = {}

    for country in countries:
        df_country = df[df['Location'] == country]
        df_yearly = df_country.groupby('Period')['FactValueNumeric'].mean().reset_index()
        df_yearly = df_yearly[df_yearly['FactValueNumeric'] > 0]

        if df_yearly.empty or len(df_yearly['Period'].unique()) < 3:
            continue

        X = df_yearly[['Period']]
        y = np.log(df_yearly['FactValueNumeric'])

        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        trained_models[country] = model
        poly_transformers[country] = poly
        year_ranges[country] = (X['Period'].min(), X['Period'].max())
        training_data[country] = df_yearly

    return trained_models, poly_transformers, year_ranges, training_data


def predict_cases(country, year):
    if country not in trained_models:
        return None, f"No model available for country: {country}"

    model = trained_models[country]
    poly = poly_transformers[country]
    min_year, _ = year_ranges[country]

    if year < min_year:
        return None, f"Year too early. Data for {country} starts from {min_year}"

    X_input = poly.transform(pd.DataFrame([[year]], columns=['Period']))
    log_pred = model.predict(X_input)[0]
    prediction = int(round(np.exp(log_pred)))
    return prediction, None


def plot_forecast(country, year):
    if country not in trained_models:
        return None

    model = trained_models[country]
    poly = poly_transformers[country]
    df_yearly = training_data[country]

    X = df_yearly[['Period']]
    y = df_yearly['FactValueNumeric']

    future_years = pd.DataFrame({
        'Period': list(range(X['Period'].max() + 1, year + 1))
    }) if year > X['Period'].max() else pd.DataFrame()

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
    ax.set_ylabel("Predicted Cases")
    ax.grid(True)
    ax.legend()
    return fig
