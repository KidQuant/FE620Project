# rho_analysis.py
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


# Calculate option price at a given stock price and risk-free rate
def calculate_option_price(S, K, T, sigma, r, B):
    """Compute the price at a given stock price S with risk-free rate r."""
    barrier_option = BarrierOption(S, K, T, sigma, r, 0, B)
    return barrier_option.up_put(knock='out')


# Calculate rho using finite difference
def calculate_rho(S, K, B, T, sigma, r):
    """Calculate Rho by finite difference."""
    dr = 0.0001  # Small change in risk-free rate
    
    # Compute the option price at risk-free rate r + dr
    V_plus = calculate_option_price(S, K, T, sigma, r + dr, B)
    
    # Compute the option price at risk-free rate r - dr
    V_minus = calculate_option_price(S, K, T, sigma, r - dr, B)
    
    # Rho is the rate of change with respect to risk-free rate
    rho = (V_plus - V_minus) / (2 * dr)
    return rho/100


# Set parameters for simulation
S0 = 50  # Spot stock price
K = 50  # Strike price
T = 1.0  # Maturity
r = 0.1  # Risk-free rate
sigma = 0.2  # Volatility
B = 100  # Barrier level

# Range of stock prices to evaluate rho across
S_values = np.linspace(K / 3, B - 1e-8, 500)

# Compute rho values across stock prices
rhos = [calculate_rho(S, K, B, T, sigma, r) for S in S_values]

# Plotting the Rho values
plt.figure(figsize=(10, 6))
plt.plot(S_values, rhos, label="Rho of Up-and-out Put Option")
plt.axvline(x=S0, color="g", linestyle="--", label="Spot Price (S)")
plt.axvline(x=K, color="b", linestyle="--", label="Strike Price (K)")
plt.axvline(x=B, color="r", linestyle="--", label="Barrier level (B)")
plt.xlabel("Stock Price $S$")
plt.ylabel("Rho")
plt.title("Rho of Up-and-out Put Option")
plt.legend(loc='upper right')
plt.grid(True)
plt.show()


specific_stock_price = 50
rho_value = calculate_rho(specific_stock_price, K, B, T, sigma, r)

print(f"Rho at S={specific_stock_price}: {rho_value:.4f}")