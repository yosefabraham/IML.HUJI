import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    X = pd.read_csv(filename, parse_dates=['Date'])
    X['DayOfYear'] = X['Date'].dt.dayofyear
    # Remove invalid temperature values
    X = X[X['Temp'] > -40]
    return X


if __name__ == '__main__':
    MONTH_NAMES = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct',
                   11: 'Nov', 12: 'Dec'}
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X = load_data('../datasets/City_Temperature.csv')
    # This is discreet and not continious
    X['Year'] = X['Year'].astype(str)
    # Question 2 - Exploring data for specific country
    iX = X.loc[X['Country'] == 'Israel']

    fig = px.scatter(iX, x="DayOfYear", y="Temp", title="Temperatures in Israel", color='Year')
    fig.update_layout({'xaxis_title': 'Day of Year', 'yaxis_title': 'Temperature'})
    fig.show()

    StdTemperaturesByMonth = iX.groupby('Month').agg({"Temp": "std"})
    StdTemperaturesByMonth = StdTemperaturesByMonth.rename(columns={'Temp': 'StdTemp'})

    fig = px.bar(StdTemperaturesByMonth, y="StdTemp", barmode="group", title="Temperatures in Israel")
    fig.update_layout({'xaxis_title': 'Month', 'yaxis_title': 'STD of temperature'})

    fig.show()

    # Question 3 - Exploring differences between countries
    StdTemperaturesByCountryMonth = X.groupby(['Country', 'Month']).Temp.agg(["std", "mean"]).reset_index()
    fig = px.line(StdTemperaturesByCountryMonth, x=['Month'], y='mean', error_y='std', color='Country',
                  title='Temperatures around the world')
    fig.update_layout({'xaxis_title': 'Month', 'yaxis_title': 'Avg. temperature'})
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    iTrainX, iTrainY, iTestX, iTestY = split_train_test(iX["DayOfYear"].to_frame(), iX["Temp"])
    iTrainX = iTrainX['DayOfYear']
    iTestX = iTestX['DayOfYear']
    degs = np.array(range(1, 11))
    lossesByDeg = []
    for k in degs:
        poly = PolynomialFitting(k)
        poly.fit(iTrainX.values, iTrainY.values)
        lossesByDeg.append(round(poly.loss(iTestX.values, iTestY.values), 2))

    print(lossesByDeg)
    fig = px.bar(x=degs, y=lossesByDeg, title='Polynomial degrees losses')
    fig.update_layout({'xaxis_title': 'Polynomial degree', 'yaxis_title': 'Loss'})
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    poly = PolynomialFitting(5)
    poly.fit(iX["DayOfYear"].values, iX["Temp"].values)

    countries = X["Country"].unique()
    other_countries = [c for c in countries if c != "Israel"]
    error_by_country = []
    for country in other_countries:
        countryX = X.loc[X["Country"] == country]
        countryLoss = poly.loss(countryX["DayOfYear"].values, countryX["Temp"].values)
        error_by_country.append(countryLoss)

    fig = px.bar(x=other_countries, y=error_by_country, title='5-Polynomial fitting model Performance')
    fig.update_layout({'xaxis_title': 'Country', 'yaxis_title': 'Loss'})
    fig.show()
