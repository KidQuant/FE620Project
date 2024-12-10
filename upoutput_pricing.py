# delta_analysis.py
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import splu


# Define the BSmodel and BarrierOption classes
class BSmodel:
    def __init__(self, S0, K, T, v, r, q):
        self.S0 = S0
        self.K = K
        self.T = T
        self.v = v
        self.r = r
        self.q = q

    @property
    def d1(self):
        return (np.log(self.S0 / self.K) + (self.r - self.q + 0.5 * self.v**2) * self.T) / (self.v * np.sqrt(self.T))

    @property
    def d2(self):
        return self.d1 - self.v * np.sqrt(self.T)


class BarrierOption(BSmodel):
    def __init__(self, S0, K, T, v, r, q, barrier):
        super().__init__(S0, K, T, v, r, q)
        self.H = barrier

    def up_put(self, knock='out'):
        if knock not in ['in', 'out']:
            raise ValueError("knock must be 'in' or 'out'")
        if self.S0 >= self.H:
            return 0  # Knocked out
        vanilla_put = self.vanilla_european_put()
        if knock == 'out':
            return vanilla_put
        else:
            return vanilla_put  # Placeholder for up-and-in put option

    def vanilla_european_put(self):
        p = (self.K * np.exp(-self.r * self.T) * ss.norm.cdf(-self.d2) -
             self.S0 * np.exp(-self.q * self.T) * ss.norm.cdf(-self.d1))
        return p


# Define parameters
S0 = 40.0  # spot stock price
K = 50  # strike
T = 1.0  # maturity
r = 0.1  # risk free rate
sigma = 0.2  # diffusion coefficient or volatility
B = 100  # Barrier level

# Initialize barrier option pricing class
barrier_option = BarrierOption(S0=S0, K=K, T=T, v=sigma, r=r, q=0, barrier=B)
# Calculate baseline option price at the initial spot S0
baseline_price = barrier_option.up_put(knock='out')
print(f"Baseline option price at S0={S0}: {baseline_price:.4f}")
