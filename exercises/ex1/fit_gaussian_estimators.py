from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu = 10
    sig = 1
    total_samples = 1000
    gaussian = UnivariateGaussian()
    samples = np.random.normal(mu, sig, size=total_samples)
    gaussian.fit(samples)
    print(f"({gaussian.mu_}, {gaussian.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    estimations = []
    sample_sizes = np.arange(10, total_samples + 1, 10)
    for n_samples in sample_sizes:
        # Fit according to first 1...10,20,30...1000 samples
        gaussian.fit(samples[:n_samples + 1])
        # for the graph
        estimations.append(gaussian.mu_)

    # transform data to show relative error
    errors = np.abs(np.array(estimations) - mu)

    # plot the graph
    fig = go.Figure([go.Scatter(x=sample_sizes, y=errors, name="Error")])
    fig.update_layout(xaxis_title='# Samples', yaxis_title='Error',
                      title='Difference between estimated and real expectation')
    fig.show()
    # Question 3 - Plot PDF of previously-drawn samples
    fig = go.Figure([go.Scatter(x=samples, y=gaussian.pdf(samples), mode='markers')])
    fig.update_layout(xaxis_title='Sample Value', yaxis_title='Density', title='Empirical PDF graph')

    fig.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    gaussian = MultivariateGaussian()
    mu = np.array([0, 0, 4, 0])
    sig = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])

    n_samples = 1000
    n_means_to_check = 200

    samples = np.random.multivariate_normal(mu, sig, size=n_samples)
    # Question 4 - Fit the multivariate gaussian according with the samples
    gaussian.fit(samples)
    print(gaussian.mu_)
    print(gaussian.cov_)

    # Question 5 + 6 - Log-likelihood evaluation
    means_to_check = np.linspace(-10, 10, n_means_to_check)
    data = np.array(
        [[gaussian.log_likelihood(np.array([f1, 0, f3, 0]), sig, samples) for f3 in means_to_check] for f1 in
         means_to_check])

    maximum_likelihood = data.max()
    # Find which index provided the maximal likelihood
    ((max_f1_i, max_f3_i),) = np.argwhere(data == maximum_likelihood)
    # Display the result in the graph
    max_description = f"Maximum likelihood is: {round(maximum_likelihood, 3)}, Achieved at f1=" \
                      f"{round(means_to_check[max_f1_i], 3)}, f3={round(means_to_check[max_f3_i], 3)}."
    # Plot the graph
    fig = go.Figure([go.Heatmap(x=means_to_check, y=means_to_check, z=data)])
    fig.update_layout(xaxis_title='Feature 3', yaxis_title='Feature 1',
                      title='Log likelihood of multivariate normal distribution with mean=[f1, 0, f3, 0]<br />'
                            f'<sup>{max_description}</sup>', )
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
test_univariate_gaussian()
test_multivariate_gaussian()
