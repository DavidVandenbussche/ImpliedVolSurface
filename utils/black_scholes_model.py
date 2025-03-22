import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.graph_objs as go

class BlackScholes:
    def __init__(self, r, s, k, t, sigma):
        self.r = r          # Risk-free rate
        self.k = k          # Strike price
        self.s = s          # Stock price
        self.t = t          # Time to expiration
        self.sigma = sigma  # Volatility

    def calculate_df(self):
        try:
            d1 = (np.log(self.s / self.k) + (self.r + 0.5 * self.sigma**2) * self.t) / (self.sigma * np.sqrt(self.t))
            d2 = d1 - self.sigma * np.sqrt(self.t)
            return d1, d2
        except ZeroDivisionError:
            raise ValueError("Time to expiration must be greater than 0")

    def option(self, option_type='Call'):
        d1, d2 = self.calculate_df()
        option_type = option_type.capitalize()
        if option_type == "Call":
            price = (self.s * norm.cdf(d1)) - (self.k * np.exp(-self.r * self.t) * norm.cdf(d2))
        elif option_type == "Put":
            price = (self.k * np.exp(-self.r * self.t) * norm.cdf(-d2)) - (self.s * norm.cdf(-d1))
        else:
            raise ValueError('Invalid option type. Use "Call" or "Put"')
        return round(price, 2)

    def greeks(self, option_type):
        d1, d2 = self.calculate_df()
        pdf_d1 = norm.pdf(d1)
        sqrt_T = np.sqrt(self.t)
        exp_neg_rt = np.exp(-self.r * self.t)

        gamma = pdf_d1 / (self.s * self.sigma * sqrt_T)
        vega = self.s * pdf_d1 * sqrt_T
        if option_type == "Call":
            delta = norm.cdf(d1)
            theta = (-self.s * pdf_d1 * self.sigma / (2 * sqrt_T)) - (self.r * self.k * exp_neg_rt * norm.cdf(d2))
            rho = self.k * self.t * exp_neg_rt * norm.cdf(d2)
        elif option_type == "Put":
            delta = -norm.cdf(-d1)
            theta = (-self.s * pdf_d1 * self.sigma / (2 * sqrt_T)) + (self.r * self.k * exp_neg_rt * norm.cdf(-d2))
            rho = -self.k * self.t * exp_neg_rt * norm.cdf(-d2)
        else:
            raise ValueError("Option type must be 'Call' or 'Put'")

        return {
            'delta': round(delta, 3),
            'gamma': round(gamma, 6),
            'theta': round(theta / 365, 6),
            'vega': round(vega * 0.01, 6),
            'rho': round(rho * 0.01, 6)
        }

    def greek_visualisation(self, option_type, greek):
        fig = go.Figure()
        line_color = '#FA7070' if option_type == 'Call' else '#799351'
        min_s = self.s * 0.92
        max_s = self.s * 1.09
        spot_values = np.linspace(min_s, max_s, 200)

        greek_values = [BlackScholes(self.r, s, self.k, self.t, self.sigma).greeks(option_type)[greek] for s in spot_values]
        current_greek_value = self.greeks(option_type)[greek]

        fig.add_trace(go.Scatter(x=spot_values, y=greek_values, mode='lines', name=greek.capitalize(), line=dict(color=line_color, width=3)))
        fig.add_trace(go.Scatter(x=[self.s], y=[current_greek_value], mode='markers', name=f'Current {greek.capitalize()}', marker=dict(color='black', size=7)))

        fig.update_layout(title=f'{greek.capitalize()} vs Spot Price ({option_type})', xaxis_title='Spot Price', yaxis_title=greek.capitalize())
        return fig
