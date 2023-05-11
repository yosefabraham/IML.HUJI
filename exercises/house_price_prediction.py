import os

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    y : array-like of shape (n_samples, )
        Response vector corresponding given samples
    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # New features
    X['livingroom_apartment_ratio'] = X.sqft_living / X.sqft_lot
    X['built_or_renovated'] = X[['yr_built', 'yr_renovated']].max(axis=1)
    X['floors_squared'] = X.floors * X.floors
    X['relative_lot_sqft15'] = X.sqft_lot / X.sqft_lot15
    X['relative_living_sqft15'] = X.sqft_living / X.sqft_living15

    X['zipcode'] = X['zipcode'].fillna(0)
    X['zipcode'] = X['zipcode'].apply(lambda x: int(float(x) / 10))
    #
    X = pd.get_dummies(X, columns=["zipcode"])

    if y is not None:
        # Remove rows with invalid price
        X.drop_duplicates(inplace=True)
        X.dropna(inplace=True)
        y = y.loc[X.index]

        # remove rows with invalid prices
        y = y[y > 0]
        X = X.loc[y.index]
    else:
        X.fillna(X.mean(), inplace=True)
    X = X.drop(columns=['date', 'lat', 'long', 'id'])
    return X, y


def pearson_coeff(x: pd.Series, y: pd.Series):
    return y.cov(x) / (np.sqrt(np.var(x) * np.var(y)))


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    y : array-like of shape (n_samples, )
        Response vector to evaluate against
    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature_name in X.columns:
        x = X[feature_name]
        pearson = pearson_coeff(x, y)
        fig = go.Figure()
        fig.add_traces([go.Scatter(x=x, y=y, mode="markers", name="Samples")])
        fig.update_layout(xaxis_title=feature_name, yaxis_title='price', title=f'Pearson correlation {feature_name} / '
                                                                               f'price: {pearson}')

        pio.write_image(fig, os.path.join(output_path, f'{feature_name}.png'))


if __name__ == '__main__':
    np.random.seed(0)
    X = pd.read_csv("../datasets/house_prices.csv")
    y = X.pop('price')

    # Question 1 - split data into train and test sets
    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 2 - Preprocessing of housing prices dataset
    X, y = preprocess_data(X, y)
    train_X, train_y = preprocess_data(train_X, train_y)
    test_X, test_y = preprocess_data(test_X, test_y)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(X, y, './ex2/features')

    reg = LinearRegression(include_intercept=True)
    # Question 4 - Fit model over increasing percentages of the overall training data
    percentages = np.arange(10, 101, 1)
    losses_means_and_stds = []
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    losses_mean = []
    losses_std = []
    for percent in percentages:
        p = percent / 100
        p_losses = []

        for _ in range(10):
            #   1) Sample p% of the overall training data
            curr_train_X = train_X.sample(frac=p)
            curr_train_y = train_y[curr_train_X.index]

            #   2) Fit linear model (including intercept) over sampled set
            reg.fit(curr_train_X.values, curr_train_y.values)
            #   3) Test fitted model over test set
            loss = reg.loss(test_X.values, test_y.values)
            p_losses.append(loss)
        #   4) Store average and variance of loss over test set

        mean = np.mean(p_losses)
        std = np.std(p_losses)
        losses_mean.append(mean)
        losses_std.append(std)
        losses_means_and_stds.append((np.mean(p_losses), np.std(p_losses)))

    losses_mean = np.array(losses_mean)
    losses_std = np.array(losses_std)
    fig = go.Figure([go.Scatter(x=percentages, y=losses_mean, line=dict(color='rgb(0,100,80)'), mode='markers+lines',
                                name='Average loss as function of training size'),
                     go.Scatter(x=percentages,
                                y=losses_mean + (2 * losses_std),
                                fill=None, fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'),
                                hoverinfo="skip", showlegend=False),
                     go.Scatter(x=percentages,
                                y=losses_mean - (2 * losses_std),
                                fill='tonexty', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'),
                                hoverinfo="skip", showlegend=False)
                     ])
    fig.show()

# %%
