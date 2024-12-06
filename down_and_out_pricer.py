import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def down_and_out_option_price(S, K, H, r, q, sigma, T, option_type):
    """Price a down-and-out European call or put option using the Black-Scholes PDE solution."""
    if S <= H:
        return 0.0  # Option is worthless if the barrier is breached at inception

    # Compute lambda
    lambda_ = (r - q + 0.5 * sigma**2) / sigma**2

    # Compute variables for the pricing formula
    x1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    x2 = x1 - sigma * np.sqrt(T)
    y1 = (np.log((H**2) / (S * K)) + (r - q + 0.5 * sigma**2) * T) / (
        sigma * np.sqrt(T)
    )
    y2 = y1 - sigma * np.sqrt(T)

    if option_type == "call":
        A = S * np.exp(-q * T) * norm.cdf(x1)
        B = K * np.exp(-r * T) * norm.cdf(x2)
        C = S * np.exp(-q * T) * (H / S) ** (2 * lambda_) * norm.cdf(y1)
        D = K * np.exp(-r * T) * (H / S) ** (2 * lambda_ - 2) * norm.cdf(y2)
        price = A - B - C + D
    elif option_type == "put":
        A = K * np.exp(-r * T) * norm.cdf(-x2)
        B = S * np.exp(-q * T) * norm.cdf(-x1)
        C = K * np.exp(-r * T) * (H / S) ** (2 * lambda_ - 2) * norm.cdf(-y2)
        D = S * np.exp(-q * T) * (H / S) ** (2 * lambda_) * norm.cdf(-y1)
        price = A - B - C + D
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return max(price, 0.0)  # Ensure the price is non-negative


# Parameters
S0 = 120  # Initial stock price
K = 90  # Strike price
H = 110  # Barrier level (down-and-out)
r = 0.04  # Risk-free interest rate
q = 0.00  # Dividend yield
sigma = 0.25  # Volatility of the underlying asset
T = 0.5  # Time to maturity in years
option_type = "call"  # Option type: 'call' or 'put'

# Calculate option price
price = down_and_out_option_price(S0, K, H, r, q, sigma, T, option_type)
print(f"Down-and-out {option_type} option price: {price:.4f}")

# Plot option price as a function of underlying price
# S_values = np.linspace(H + 1e-8, 2 * S0, 500)
# option_prices = [
#     down_and_out_option_price(S, K, H, r, q, sigma, T, option_type) for S in S_values
# ]

# plt.figure(figsize=(10, 6))
# plt.plot(S_values, option_prices, label=f"Down-and-out {option_type} option price")
# plt.axvline(x=H, color="r", linestyle="--", label="Barrier level (H)")
# plt.xlabel("Stock Price $S$")
# plt.ylabel("Option Price")
# plt.title(f"Price of Down-and-out {option_type.capitalize()} Option")
# plt.legend()
# plt.grid(True)
# plt.show()
