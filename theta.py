from BSM_option_class import BarrierOption
import matplotlib.pyplot as plt
import numpy as np


# Define parameters
S0 = 40  # spot stock price
X0 = np.log(S0)
K = 50  # strike price
T = 1.0  # maturity
r = 0.1  # risk-free rate
sigma = 0.2  # volatility
B = 100  # Barrier level


# Calculate Theta using finite difference
def calculate_theta(S0, K, B, T, sigma, r, epsilon=1/252):
    """
    Calculate Theta using finite difference (change in time).
    epsilon: Small change in time (1 trading day expressed in years).
    """
    # Current time value
    barrier_option_now = BarrierOption(S0, K, T, sigma, r, 0, B)
    price_now = barrier_option_now.up_put(knock='out')
    
    # Slightly earlier time value
    barrier_option_earlier = BarrierOption(S0, K, T - epsilon, sigma, r, 0, B)
    price_earlier = barrier_option_earlier.up_put(knock='out')
    
    # Theta calculation
    theta = (price_earlier - price_now)
    return theta


# Compute specific Theta at S0=40
specific_theta = calculate_theta(S0, K, B, T, sigma, r)
print(f"Theta at S0={S0}, T={T}, sigma={sigma}: {specific_theta:.4f}")

# Define a range of stock prices for plotting
S_values = np.linspace(K / 3, B - 1e-8, 500)  # Range of stock prices

# Compute Theta across a range of stock prices
thetas = [calculate_theta(S, K, B, T, sigma, r) for S in S_values]

# Plotting Theta over the range
plt.figure(figsize=(10, 6))
plt.plot(S_values, thetas, label="Theta of Up-and-out Put Option")
plt.axvline(x=S0, color="g", linestyle="--", label=f"Spot Price (S0={S0})")
plt.axvline(x=K, color="b", linestyle="--", label="Strike Price (K)")
plt.axvline(x=B, color="r", linestyle="--", label="Barrier level (B)")
plt.xlabel("Stock Price $S$")
plt.ylabel("Theta")
plt.title("Theta of Up-and-out Put Option as a Function of Stock Price")
plt.legend(loc="upper right")
plt.grid(True)
plt.show()
