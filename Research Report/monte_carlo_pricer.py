
import numpy as np
import pandas as pd
import statistics
import scipy.stats as sp
import time
from datetime import timedelta, date
import warnings
from scipy.stats import norm
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def sim_MC_paths(S0, T, r, sigma, n, M):
    """
    Parameters:
    S0: initial price
    T: time to maturity 
    r: risk free rate
    sigma: volatility
    n: number of time steps
    M: number of paths
    """
    paths = []
    dt = T / n
    
    #simulate M paths
    for _ in range(M):
        s = [S0] #declare list of prices 
        #increment n timesteps
        for _ in range(n):
            z = np.random.normal(0,1)
            px = s[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z) #get next price based on prev. price in list
            s.append(px)
        paths.append(s)
    return paths

def calc_BS_Option_Price(So, sigma, t, K, r, optType):
    d_plus = (np.log(So/K) + (r + (sigma**2)/2)*t)/(sigma*np.sqrt(t))
    d_minus = d_plus - sigma * np.sqrt(t)
    if optType == "call":
        price = So*norm.cdf(d_plus) - K*np.exp(-r*t)*norm.cdf(d_minus)
    elif optType == "put":
        price = K*np.exp(-r*t)*norm.cdf(-d_minus) - So*norm.cdf(-d_plus)
    return price


results = pd.DataFrame(columns=['barrier_Scale','in_call','out_call','in_put','out_put','sim_call','sim_put', 'bs_call','bs_put'])

S0 = 100  # Initial stock price
K = 90  # Strike price
r = 0.05  # Risk-free interest rate
sigma = 0.25  # Volatility of the underlying asset
T = 0.5  # Time to maturity in years
n = 252
M = 100
paths = sim_MC_paths(S0,T,r,sigma,n,M)

S0Scales = [60]

for n in S0Scales:
    b =  n
    kicall = []
    kiput = []
    kocall = []
    koput = []
    vput = []
    vcall = []
    for s in paths:    
        if b > S0: #up options
            if max(s) > b: #knock in
                kicall.append(max(s[-1] - K, 0))
                kocall.append(0)
                kiput.append(max(K - s[-1], 0))
                koput.append(0)
            else: #knock out
                kicall.append(0)
                kocall.append(max(s[-1] - K,0))
                kiput.append(0)
                koput.append(max(K - s[-1],0))
        else: #down options
            if min(s) < b: #knock in 
                kicall.append(max(s[-1] - K, 0))
                kocall.append(0)
                kiput.append(max(K - s[-1], 0))
                koput.append(0)
            else: #knock out
                kicall.append(0)
                kocall.append(max(s[-1] - K,0))
                kiput.append(0)
                koput.append(max(K - s[-1],0))
        #vanilla options with k = b
        vcall.append(max(s[-1] - b, 0))
        vput.append(max(b - s[-1],0))

    price_kicall = np.exp(-r * T) * np.mean(kicall)
    price_kiput = np.exp(-r * T) * np.mean(kiput)
    price_kocall = np.exp(-r * T) * np.mean(kocall)
    price_koput = np.exp(-r * T) * np.mean(koput)
    price_vcall = np.exp(-r * T) * np.mean(vcall)
    price_vput = np.exp(-r * T) * np.mean(vput)
    price_bs_call = calc_BS_Option_Price(S0, sigma, T, b, r, "call")
    price_bs_put = calc_BS_Option_Price(S0, sigma, T, b, r, "put")
    results.loc[len(results)] = [n,price_kicall,price_kocall,price_kiput,price_koput,price_vcall,price_vput,price_bs_call,price_bs_put]


results