import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from BSM_option_class import BSmodel, BarrierOption, VanillaEuro
from MC_barrier_options import MC_BarrierOption

from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import spsolve

S0 = 100.0  # spot stock price
X0 = np.log(S0)
K = 100.0  # strike
T = 1.0  # maturity
r = 0.1  # risk free rate
sigma = 0.2  # diffusion coefficient or volatility
B = 250  # Barrier 1
sim=10000
call = VanillaEuro(S0, K, T, sigma,r, 0)

call.vanilla_european_put()

barrier = BarrierOption(S0, K, T, sigma, r, 0, B)
BSM_price = barrier.up_put('out')


# BSM_price = 1.6735
print(f"Black Scholes Option Price: {BSM_price:.4f}")

MC_barrier = MC_BarrierOption("put", "up_and_out", S0, K, B, T, sigma, r)
MC_price = MC_barrier.monte_carlo_pricing(sim, sim, plot=False)

print(f"Monte Carlo Option Price: {MC_price:.4f}")

err = BSM_price - MC_price
print(f"MC error: {err:.4f}")


d1 = lambda t, s: 1 / (sigma * np.sqrt(t)) * (np.log(s) + (r + sigma**2 / 2) * t)
d2 = lambda t, s: 1 / (sigma * np.sqrt(t)) * (np.log(s) + (r - sigma**2 / 2) * t)

closed_barrier_u = (
    S0 * (ss.norm.cdf(d1(T, S0 / K)) - ss.norm.cdf(d1(T, S0 / B)))
    - np.exp(-r * T) * K * (ss.norm.cdf(d2(T, S0 / K)) - ss.norm.cdf(d2(T, S0 / B)))
    - B * (S0 / B) ** (-2 * r / sigma**2) * (ss.norm.cdf(d1(T, B**2 / (S0 * K))) - ss.norm.cdf(d1(T, B / S0)))
    + np.exp(-r * T)
    * K
    * (S0 / B) ** (-2 * r / sigma**2 + 1)
    * (ss.norm.cdf(d2(T, B**2 / (S0 * K))) - ss.norm.cdf(d2(T, B / S0)))
)

print("The price of the Up and Out call option by closed formula is: ", closed_barrier_u)

Nspace = 14000  # M space steps
Ntime = 10000  # N time steps
S_max = B  # The max of S corresponds to the Barrier
S_min = float(K) / 3
x_max = np.log(S_max)  # A2
x_min = np.log(S_min)  # A1

x, dx = np.linspace(x_min, x_max, Nspace, retstep=True)  # space discretization
T_array, dt = np.linspace(0, T, Ntime, retstep=True)  # time discretization
Payoff = np.maximum(np.exp(x) - K, 0)  # Call payoff

V = np.zeros((Nspace, Ntime))  # grid initialization
offset = np.zeros(Nspace - 2)  # vector to be used for the boundary terms

V[:, -1] = Payoff  # terminal conditions
V[-1, :] = 0  # boundary condition
V[0, :] = 0  # boundary condition

# construction of the tri-diagonal matrix D
sig2 = sigma * sigma
dxx = dx * dx
a = (dt / 2) * ((r - 0.5 * sig2) / dx - sig2 / dxx)
b = 1 + dt * (sig2 / dxx + r)
c = -(dt / 2) * ((r - 0.5 * sig2) / dx + sig2 / dxx)
D = sparse.diags([a, b, c], [-1, 0, 1], shape=(Nspace - 2, Nspace - 2)).tocsc()
DD = splu(D)

# Backward iteration
for i in range(Ntime - 2, -1, -1):
    offset[0] = a * V[0, i]
    offset[-1] = c * V[-1, i]
    V[1:-1, i] = DD.solve(V[1:-1, i + 1] - offset)

# = V * (oPrice / np.max(V))
oPrice = np.interp(X0, x, V[:, 0])
print("The price of the Up and Out option by PDE is: ", oPrice)


S = np.exp(x)
fig = plt.figure(figsize=(18, 6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection="3d")

ax1.plot(S, Payoff, color="blue", label="Payoff")
ax1.plot(S, V[:, 0], color="red", label="Barrier curve")
ax1.set_xlim(40, 130)
ax1.set_ylim(0, 1.9)
ax1.set_xlabel("S")
ax1.set_ylabel("V")
ax1.legend(loc="upper right")
ax1.set_title("Curve at t=0")

X, Y = np.meshgrid(T_array, S)
ax2.plot_surface(Y, X, V)
ax2.set_title("Barrier option Up and Out surface")
ax2.set_xlabel("S")
ax2.set_ylabel("t")
ax2.set_zlabel("V")
plt.show()