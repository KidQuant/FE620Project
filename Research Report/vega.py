from BSM_option_class import BarrierOption
import matplotlib.pyplot as plt
import numpy as np


# Define parameters
S0 = 40.0  # spot stock price
X0 = np.log(S0)
K = 50  # strike price
T = 1.0  # maturity
r = 0.1  # risk-free rate
sigma = 0.2  # volatility
B = 100  # Barrier level


# Calculate Vega using finite difference
def calculate_vega(S0, K, B, T, sigma, r, epsilon=1e-4):
    """
    Calculate Vega for an up-and-out put option using finite difference.
    epsilon: Small change in volatility.
    """
    # Increase volatility by a small amount
    price_plus_vol = BarrierOption(S0, K, T, sigma + epsilon, r, 0, B).up_put(knock='out')
    
    # Decrease volatility by a small amount
    price_minus_vol = BarrierOption(S0, K, T, sigma - epsilon, r, 0, B).up_put(knock='out')
    
    # Compute Vega
    vega = (price_plus_vol - price_minus_vol) / (2 * epsilon)
    
    return vega/100


# Compute specific Vega at S0=40
specific_vega = calculate_vega(S0, K, B, T, sigma, r)
print(f"Vega at S0={S0}, T={T}, sigma={sigma}: {specific_vega:.4f}")

# Define a range of stock prices for plotting
S_values = np.linspace(K / 3, B - 1e-8, 500)  # Range of stock prices

# Compute Vega across a range of stock prices
vegas = [calculate_vega(S, K, B, T, sigma, r) for S in S_values]

# Plotting Vega over the range
plt.figure(figsize=(10, 6))
plt.plot(S_values, vegas, label="Vega of Up-and-out Put Option")
plt.axvline(x=S0, color="g", linestyle="--", label=f"Spot Price (S0={S0})")
plt.axvline(x=K, color="b", linestyle="--", label="Strike Price (K)")
plt.axvline(x=B, color="r", linestyle="--", label="Barrier level (B)")
plt.xlabel("Stock Price $S$")
plt.ylabel("Vega")
plt.title("Vega of Up-and-out Put Option as a Function of Stock Price")
plt.legend(loc="upper right")
plt.grid(False)
plt.show()
