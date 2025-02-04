import pandas as pd 
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv('data/train_df.csv')


df['returns'] = df['Close'].pct_change()
df['volatility'] = df['returns'].rolling(10).std()
df['momentum'] = df['returns'].rolling(10).mean()
df['ma'] = df['Close'].rolling(10).mean()
df['rsi'] = 100 - (100 / (1 + df['momentum']))
df['ema'] = df['Close'].ewm(span=10, adjust=False).mean()
df['macd'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
df['lag_1'] = df['Close'].shift(1)
df['lag_2'] = df['Close'].shift(2)
df['lag_3'] = df['Close'].shift(3)
df['lag_4'] = df['Close'].shift(4)
df['lag_5'] = df['Close'].shift(5)
df= df.dropna()


df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

X = df.drop(['target', 'Close', 'returns', 'Unnamed: 0', 'datetime'], axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(objective='binary:logistic', 
                          max_depth = 5, 
                          learning_rate = 0.001,
                          n_estimators=1000,
                          
                          eval_metric='logloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

df['predicted_signal'] = model.predict(X)
df['strategy_returns'] = df['returns'] * df['predicted_signal'].shift(1)
(1 + df['strategy_returns']).cumprod().plot()
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
plt.show()

def predict_trading_signal(data_row):
    signal =  model.predict(data_row.values.reshape(1, -1))[0]
    return "Buy" if signal == 1 else "Sell"


latest_data = X_test.iloc[0]


model.save_model('model/trade_xgboost.json')

last_10_entries = X_test.tail(1000)
last_10_predictions = last_10_entries.apply(predict_trading_signal, axis=1)


last_10_signals = pd.DataFrame({'Timestamp':df['datetime'].tail(1000), 'close_price':df['Close'].tail(1000),
                                'Signal':last_10_predictions.values})

last_10_signals.to_csv('result/last_10_signals.csv', index=False)

all_predictions = X_test.apply(predict_trading_signal, axis=1)

# Create a DataFrame with the corresponding signals
all_signals = pd.DataFrame({'Timestamp': df['datetime'].iloc[:len(X_test)], 
                            'close_price': df['Close'].iloc[:len(X_test)],
                            'Signal': all_predictions.values})
print(all_signals.shape, len(X_test))
# Save the result to a CSV file
all_signals.to_csv('result/all_signals.csv', index=False)

# Load the data
data = pd.read_csv('result/all_signals.csv', parse_dates=['Timestamp'])

# Initialize variables
first_buy = None
sell_triggered = False
profit = []
filtered_data = []
gross_profit = 0
gross_loss = 0
win_trades = 0
loss_trades = 0
total_trades = 0

# Loop through the data to process it
for index, row in data.iterrows():
    if row['Signal'] == 'Buy':
        if not first_buy:
            # Store the first Buy signal price
            first_buy = row['close_price']
            filtered_data.append({
            'Timestamp': row['Timestamp'],
            'close_price': row['close_price'],
            'Signal': row['Signal'],
            'Profit': ''
        })
        # Ignore subsequent "Buy" signals after the first one
        else:
            continue
    elif row['Signal'] == 'Sell' and first_buy:
        # Calculate profit when a Sell is triggered
        profit_value = row['close_price'] - first_buy
        profit.append(profit_value)
        
        # Store the data with the "Sell" signal
        filtered_data.append({
            'Timestamp': row['Timestamp'],
            'close_price': row['close_price'],
            'Signal': row['Signal'],
            'Profit': profit_value
        })
        # Update gross profit or loss, win/loss trade count
        if profit_value > 0:
            gross_profit += profit_value
            win_trades += 1
        else:
            gross_loss += profit_value
            loss_trades += 1
        
        # Increment total trade count
        total_trades += 1
        
        # Reset for next cycle
        first_buy = None

# Create a DataFrame from the filtered data
filtered_df = pd.DataFrame(filtered_data)
net_profit = gross_profit + gross_loss
# Save the result to a new CSV file
filtered_df.to_csv('filtered_data_with_profit.csv', index=False)

# Print the results

print(f"\nGross Profit: {gross_profit}")
print(f"Gross Loss: {gross_loss}")
print(f"Net Profit: {net_profit}")
print(f"Total Trades: {total_trades}")
print(f"Winning Trades: {win_trades}")
print(f"Losing Trades: {loss_trades}")

print('**********************************************************')

data = pd.read_csv('result/last_10_signals.csv', parse_dates=['Timestamp'])

# Initialize variables
first_buy = None
sell_triggered = False
profit = []
filtered_data = []
gross_profit = 0
gross_loss = 0
win_trades = 0
loss_trades = 0
total_trades = 0

# Loop through the data to process it
for index, row in data.iterrows():
    if row['Signal'] == 'Buy':
        if not first_buy:
            # Store the first Buy signal price
            first_buy = row['close_price']
            filtered_data.append({
            'Timestamp': row['Timestamp'],
            'close_price': row['close_price'],
            'Signal': row['Signal'],
            'Profit': ''
        })
        # Ignore subsequent "Buy" signals after the first one
        else:
            continue
    elif row['Signal'] == 'Sell' and first_buy:
        # Calculate profit when a Sell is triggered
        profit_value = row['close_price'] - first_buy
        profit.append(profit_value)
        
        # Store the data with the "Sell" signal
        filtered_data.append({
            'Timestamp': row['Timestamp'],
            'close_price': row['close_price'],
            'Signal': row['Signal'],
            'Profit': profit_value
        })
        # Update gross profit or loss, win/loss trade count
        if profit_value > 0:
            gross_profit += profit_value
            win_trades += 1
        else:
            gross_loss += profit_value
            loss_trades += 1
        
        # Increment total trade count
        total_trades += 1
        
        # Reset for next cycle
        first_buy = None

# Create a DataFrame from the filtered data
filtered_df = pd.DataFrame(filtered_data)
net_profit = gross_profit + gross_loss
# Save the result to a new CSV file
filtered_df.to_csv('filtered_data_with_1000_profit.csv', index=False)

# Print the results

print(f"\nGross Profit: {gross_profit}")
print(f"Gross Loss: {gross_loss}")
print(f"Net Profit: {net_profit}")
print(f"Total Trades: {total_trades}")
print(f"Winning Trades: {win_trades}")
print(f"Losing Trades: {loss_trades}")



'''  
Accuracy: 52.78% , depth = 4
Gross Profit: 21730.760000000002
Gross Loss: -22526.790000000066
Net Profit: -796.0300000000643
Total Trades: 253
Winning Trades: 122
Losing Trades: 131

Accuracy: 52.98% , depth = 6
Gross Profit: 22977.289999999906
Gross Loss: -18738.949999999997
Net Profit: 4238.339999999909
Total Trades: 254
Winning Trades: 136
Losing Trades: 118

Accuracy: 52.98%
(16373, 3) 16373

Gross Profit: 173423.67008000013
Gross Loss: -193424.99471999984
Net Profit: -20001.324639999715
Total Trades: 4115
Winning Trades: 2058
Losing Trades: 2057

Accuracy: 51.99% , depth = 8
(16373, 3) 16373

Gross Profit: 164822.64635999984
Gross Loss: -190870.81406000012
Net Profit: -26048.167700000282
Total Trades: 4146
Winning Trades: 2045
Losing Trades: 2101


Accuracy: 52.81%, depth = 4, learning_rate = 0.01
(16373, 3) 16373
Gross Profit: 171883.93953000003
Gross Loss: -187556.52495
Net Profit: -15672.58541999996
Total Trades: 4127
Winning Trades: 2056
Losing Trades: 2071

Accuracy: 52.63%, depth = 6, learning_rate = 0.01
Accuracy: 52.63%
(16373, 3) 16373

Gross Profit: 173525.01404999994
Gross Loss: -184695.15857999976
Net Profit: -11170.144529999816
Total Trades: 4116
Winning Trades: 2060
Losing Trades: 2056

Accuracy: 52.84%, depth = 6, learning_rate = 0.01, n_estimators=1000
(16373, 3) 16373

Gross Profit: 174750.8409500001
Gross Loss: -191289.90190000035
Net Profit: -16539.060950000247
Total Trades: 4103
Winning Trades: 2062
Losing Trades: 2041
'''
