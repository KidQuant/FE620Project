import numpy as np

def monte_carlo_barrier_down_and_out(
    S0, K, H, T, r, q, sigma, N, M, option_type="call"
):
    """
    Monte Carlo pricer for a Down-and-Out Barrier option with dividend yield.

    Parameters:
    - S0 (float): Initial stock price
    - K (float): Strike price
    - H (float): Barrier level
    - T (float): Time to maturity (in years)
    - r (float): Risk-free rate (annualized)
    - q (float): Dividend yield (annualized)
    - sigma (float): Volatility (annualized)
    - N (int): Number of time steps per path
    - M (int): Number of Monte Carlo simulations
    - option_type (str): "call" or "put"

    Returns:
    - option_price (float): Estimated price of the option
    """
    dt = T / N  # Time step size
    discount_factor = np.exp(-r * T)  # Discount factor
    payoff_sum = 0

    for _ in range(M):
        # Simulate a single path
        path = np.zeros(N + 1)
        path[0] = S0
        barrier_breached = False

        for t in range(1, N + 1):
            z = np.random.normal(0, 1)  # Standard normal random variable
            path[t] = path[t - 1] * np.exp(
                (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
            )

            if path[t] <= H:  # Check if barrier is breached
                barrier_breached = True
                break

        # If the barrier is breached, the payoff is 0 for this path
        if not barrier_breached:
            if option_type == "call":
                payoff = max(path[-1] - K, 0)
            elif option_type == "put":
                payoff = max(K - path[-1], 0)
            else:
                raise ValueError("Invalid option type. Use 'call' or 'put'.")

            payoff_sum += payoff

    # Average payoff and discount to present value
    option_price = discount_factor * (payoff_sum / M)
    return option_price


# Example Parameters with Dividend Yield
S0 = 100  # Initial stock price
K = 90    # Strike price
H = 95    # Barrier level
T = 0.5    # Time to maturity (1 year)
r = 0.04  # Risk-free rate (5%)
q = 0.08  # Dividend yield (2%)
sigma = 0.25  # Volatility (20%)
N = 252   # Number of time steps (daily)
M = 100  # Number of Monte Carlo simulations

# Calculate option price with dividend yield
MC_price = monte_carlo_barrier_down_and_out(
    S0, K, H, T, r, q, sigma, N, M, option_type="call"
)
MC_price
