import numpy as np
# simple moving average , inputs are microprices and a window size
def simple_moving_average(prices, window):
    return (1/window) *np.sum(prices[window:])

#exponential_moving_average , smoothing constant theta between 0 and 1
def exponential_moving_average(prices, window,theta):
    recent_prices = prices[-window:]
    weights = theta ** (np.arange(1, window + 1))
    norm = np.sum(weights)
    weighted_avg = np.sum(recent_prices[::-1] * weights) / norm
    return np.sum(weighted_avg) / norm
prices= [1,2,3,4,5,6]
#SMA rule
parameter_upper=0.1
parameter_lower=0.01
"""
window=2
if simple_moving_average(prices,window)/prices[-1]>parameter_upper:
    #sell
elif simple_moving_average(prices,window)/prices[-1]<parameter_lower:
    #buy
else:
    #do nothing and wait
theta=0.5
if exponential_moving_average(prices,window,theta)/prices[-1]>parameter_upper:
    #sell
elif exponential_moving_average(prices,window,theta)/prices[-1]<parameter_lower:
    #buy
    
else:
    #do nothing
"""

# bollinger weigthed moving average (BMWA)
def bollinger_moving_average(prices,window):
    recent_prices = prices[-window:]
    weights = np.arange(window, 0, -1)  # weights = [m, m-1, ..., 1]
    norm = np.sum(weights)
    return np.dot(recent_prices, weights) / norm

#BMWA rule to trade:
def bmwa_std(prices, window):
    # Extract the last m prices
    recent_prices = prices[-window:]
    # Compute the BWMA
    weights = np.arange(window, 0, -1)
    norm = np.sum(weights)
    bwma = np.dot(recent_prices, weights) / norm
    # Compute squared deviations from BWMA
    squared_deviations = (recent_prices - bwma) ** 2
    # Sample standard deviation (divide by m - 1)
    return np.sqrt(np.sum(squared_deviations) / (window - 1))

"""
#BMWA rule for trading:
window=2
if prices[-1]>(bollinger_moving_average(prices,window)+2*bmwa_std(prices,window)):
    #sell
elif prices[-1]< bollinger_moving_average(prices,window)-2*bmwa_std(prices,window):
    #buy


"""


#moving average oscillator, let m<n short and long averages (EMA(window=5) , EMA(window=10) e.g.)

#Oscillator rule SIGNALS a BREAK in the trend
#enter if:
small=2
large=5
theta=0.5
"""
if (exponential_moving_average(prices,small-1,theta)<exponential_moving_average(prices,large-1,theta) ) and (exponential_moving_average(prices,small,theta)>exponential_moving_average(prices,large,theta)):
    #then enter
# enter the market
#if  EMA(window=small,until time t-1)<EMA(window=large,until time t-1)   and EMA(window=small,until time t)> EMA(window=large,until time t):
#we can do EMA(prices, window=small-1) and EMA(prices,window=large-1) to achieve the until time t-1, a bit dirty but should work

#exit if
if ((exponential_moving_average(prices,small-1,theta)>exponential_moving_average(prices,large-1,0.5)) and exponential_moving_average(prices,small,theta)<exponential_moving_average(prices,large,theta)):
    #then exit 

"""

#RSI Ocillator
def calculate_RSI(prices, window):
    diffs = np.diff(prices[-(window + 1):])  # Get m differences
    gains = np.where(diffs > 0, diffs, 0)
    losses = np.where(diffs < 0, -diffs, 0)
    U_t = np.sum(gains)
    D_t = np.sum(losses)
    # Prevent division by zero
    if D_t == 0:
        return 100.0  # Max RSI if no losses
    RS = U_t / D_t
    RSI = 100 - (100 / (1 + RS))
    return RSI

#RSI rule:
window=3
"""
if calculate_RSI(prices,window)>70:   #market overbought
    #sell or exit buy position
if calculate_RSI(prices,window)<30: #market oversold
    #buy or exit short position

"""