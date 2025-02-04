# Description: Simulate future price paths using Monte Carlo simulation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

df = pd.read_csv('data/test_df.csv', parse_dates=["datetime"])

# Compute log returns from Close prices
df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
df = df.dropna()

# Simulation Parameters
num_simulations = 1000  # Number of price paths
time_horizon = 20  # Simulated trading 15 minutes 
initial_price = df["Close"].iloc[-1]  # Last observed price
log_returns = df["Log_Return"].dropna().values  # Extract log returns
print(log_returns)
# Monte Carlo Simulation
simulated_prices = np.zeros((time_horizon, num_simulations))

# ✅ Rescale log returns to avoid numerical issues
scaling_factor = 1000  # Recommended by arch warning
log_returns_garch = df["Log_Return"].values * scaling_factor

# Fit GARCH Model
garch = arch_model(log_returns_garch, vol="Garch", p=1, q=1).fit(disp="off")
print(garch.summary())
# Forecast Volatility
forecast1 = garch.forecast(horizon=1, reindex=False)
# ✅ Extract the predicted return
predicted_return = forecast1.mean.iloc[-1] / scaling_factor  # Scale back

print('predicted_return ', predicted_return)
for sim in range(num_simulations):
    price = initial_price
    for t in range(time_horizon):
        daily_return = np.random.choice(log_returns)  # Randomly pick a historical log return
        price *= np.exp(daily_return)  # Apply exponential growth
        simulated_prices[t, sim] = price

# Buy at initial price, sell at the end of simulation
buy_price = initial_price

sell_prices = simulated_prices[-1, :]  # Final simulated prices
expected_sell_prices = np.mean(sell_prices)
expected_profit = expected_sell_prices - buy_price

# Expected return & risk
# expected_profit = np.mean(profits)
# profit_std = np.std(profits)

# Plot Monte Carlo Simulated Paths
plt.figure(figsize=(12, 5))

# 1. Plot simulated price paths
plt.subplot(1, 2, 1)
plt.plot(simulated_prices, color="blue", alpha=0.1)
plt.axhline(y=buy_price, color="red", linestyle="--", label="Buy Price")
plt.xlabel("Days")
plt.ylabel("Price")
plt.title("Monte Carlo Simulated Price Paths")
plt.legend()

# 2. Plot final sell prices as a histogram
plt.subplot(1, 2, 2)
plt.hist(sell_prices, bins=50, color="green", alpha=0.7, edgecolor="black")
plt.axvline(x=buy_price, color="red", linestyle="--", label="Buy Price")
plt.xlabel("Sell Price")
plt.ylabel("Frequency")
plt.title("Distribution of Sell Prices")
plt.legend()

plt.tight_layout()
plt.show()

# Print summary statistics
print('********************************')
print("Monte Carlo Simulation Results")
print('Buy Price:', buy_price)
print('sell_prices:', expected_sell_prices)
print(f"Expected Profit: ${expected_profit:.2f}")
# print(f"Risk (Standard Deviation): ${profit_std:.2f}")
print(f"Probability of Profit: {np.mean(expected_profit > 0) * 100:.2f}%")
