import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# Load your data (OHLC data)
data = pd.read_csv("data/test_df.csv")
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)
print(data['Close'])
# Example: GARCH model for volatility prediction
returns = np.log(data['Close'] / data['Close'].shift(1)).dropna() * 1000  # Compute log returns
model = arch_model(returns, vol='Garch', p=1, q=1)
model_fit = model.fit()

# Forecast the next return and volatility
forecast = model_fit.forecast(horizon=10)
predicted_volatility = forecast.variance.values[-1, 0]  # Volatility for the next period
predicted_return = forecast.mean.values[-1, 0]  # Predicted return for the next period
print(f"Predicted Volatility: {predicted_volatility}, Predicted Return: {predicted_return}")
# Define threshold for Buy/Sell decision
buy_threshold = 0.0005  # Example: Buy if predicted return is greater than 0.05%
sell_threshold = -0.0005  # Example: Sell if predicted return is less than -0.05%
print("data['Close'].iloc[-1", data['Close'].iloc[-1])
# Example logic for Buy/Sell signals based on predicted return
if predicted_return > buy_threshold:
    print(f"Buy signal at predicted return: {predicted_return}")
    buy_price = data['Close'].iloc[-1]  # Set the buy price at the latest close price
    stop_loss = buy_price - (predicted_volatility * 2)  # Set stop-loss based on predicted volatility
    take_profit = buy_price + (predicted_volatility * 3)  # Set take-profit based on predicted volatility
    print(f"Buy Price {buy_price}  Stop-loss: {stop_loss}, Take-profit: {take_profit}")
elif predicted_return < sell_threshold:
    print(f"Sell signal at predicted return: {predicted_return}")
    sell_price = data['Close'].iloc[-1]  # Set the sell price at the latest close price
    stop_loss = sell_price + (predicted_volatility * 2)  # Set stop-loss based on predicted volatility
    take_profit = sell_price - (predicted_volatility * 3)  # Set take-profit based on predicted volatility
    print(f"Stop-loss: {stop_loss}, Take-profit: {take_profit}")
else:
    print("No action: predicted return is within the neutral range")

# Optional: Plot the Buy/Sell signals on a price chart
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Close'], label='Price', color='blue')
if predicted_return > buy_threshold:
    plt.scatter(data.index[-1], buy_price, color='green', label='Buy Signal', marker='^', s=100)
    plt.scatter(data.index[-1], stop_loss, color='red', label='Stop Loss', marker='v', s=100)
    plt.scatter(data.index[-1], take_profit, color='orange', label='Take Profit', marker='x', s=100)
elif predicted_return < sell_threshold:
    plt.scatter(data.index[-1], sell_price, color='red', label='Sell Signal', marker='v', s=100)
    plt.scatter(data.index[-1], stop_loss, color='green', label='Stop Loss', marker='^', s=100)
    plt.scatter(data.index[-1], take_profit, color='orange', label='Take Profit', marker='x', s=100)

plt.legend()
plt.title("Buy/Sell Signals with GARCH Model Prediction")
plt.show()
