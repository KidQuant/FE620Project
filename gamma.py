# gamma_analysis.py
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt


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
        return vanilla_put

    def vanilla_european_put(self):
        p = (self.K * np.exp(-self.r * self.T) * ss.norm.cdf(-self.d2) -
             self.S0 * np.exp(-self.q * self.T) * ss.norm.cdf(-self.d1))
        return p


# Calculate delta for a given stock price
def calculate_delta(S, K, B, T, sigma, r):
    """Calculate Delta using finite difference approximation."""
    epsilon = 1e-4  # A small stock price change
    # Change the stock price positively
    barrier_option_plus = BarrierOption(S + epsilon, K, T, sigma, r, 0, B)
    price_plus = barrier_option_plus.up_put(knock='out')
    # Change the stock price negatively
    barrier_option_minus = BarrierOption(S - epsilon, K, T, sigma, r, 0, B)
    price_minus = barrier_option_minus.up_put(knock='out')
    # Approximation of delta
    delta = (price_plus - price_minus) / (2 * epsilon)
    return delta


# Calculate gamma using finite difference
def calculate_gamma(S, K, B, T, sigma, r):
    """Calculate Gamma using finite difference."""
    epsilon = 1e-4  # Small change in stock price
    # Calculate deltas at stock price moved up and down
    delta_plus = calculate_delta(S + epsilon, K, B, T, sigma, r)
    delta_minus = calculate_delta(S - epsilon, K, B, T, sigma, r)
    # Compute gamma
    gamma = (delta_plus - delta_minus) / (2 * epsilon)
    return gamma


# Set parameters for simulation
S0 = 40.0  # Spot stock price
K = 50  # Strike price
T = 1.0  # Maturity
r = 0.1  # Risk-free rate
sigma = 0.2  # Volatility
B = 100  # Barrier level

# Range of stock prices to evaluate gamma across
S_values = np.linspace(K / 3, B - 1e-8, 500)

# Compute gamma values across stock prices
gammas = [calculate_gamma(S, K, B, T, sigma, r) for S in S_values]

# Plotting the Gamma values
plt.figure(figsize=(10, 6))
plt.plot(S_values, gammas, label="Gamma of Up-and-out Put Option")
plt.axvline(x=S0, color="g", linestyle="--", label="Spot Price (S)")
plt.axvline(x=K, color="b", linestyle="--", label="Strike Price (K)")
plt.axvline(x=B, color="r", linestyle="--", label="Barrier level (B)")
plt.xlabel("Stock Price $S$")
plt.ylabel("Gamma")
plt.ylim(-0.01, 0.05)  # Adjust y-axis to zoom in on the range of gamma
plt.title("Gamma of Up-and-out Put Option")
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

specific_stock_price = 40.0
gamma_value = calculate_gamma(specific_stock_price, K, B, T, sigma, r)

print(f"Gamma at S={specific_stock_price}: {gamma_value:.4f}")