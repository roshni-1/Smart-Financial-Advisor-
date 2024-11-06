import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")  # Ignore warnings

# Fetch historical data (up to the last available date)
def fetch_historical_data(stock_symbol):
    stock_symbol = stock_symbol.strip() + ".NS"  # Append '.NS' for NSE stocks
    stock_data = yf.download(stock_symbol, progress=False)
    if stock_data.empty:
        raise ValueError(f"No data found for stock symbol: {stock_symbol}")
    return stock_data

# Fetch real-time data
def fetch_real_time_data(stock_symbol):
    stock_symbol = stock_symbol.strip() + ".NS"  # Append '.NS' for NSE stocks
    ticker = yf.Ticker(stock_symbol)
    real_time_price = ticker.history(period="1d")['Close'][-1]
    return real_time_price

# Train ARIMA model
def train_arima_model(stock_data):
    close_prices = stock_data['Close']
    model = ARIMA(close_prices, order=(5, 1, 2))
    arima_model = model.fit()
    return arima_model

# Make prediction with error checking
def make_prediction(arima_model):
    try:
        forecast = arima_model.forecast(steps=1)
        if not forecast.empty:
            return forecast.iloc[0]  # Ensure correct access for a single forecast value
        else:
            raise ValueError("ARIMA forecast returned no data.")
    except Exception as e:
        raise ValueError(f"Failed to make prediction: {e}")

# Main program logic
def main():
    # User input for multiple stock symbols and quantities
    stock_symbols = input("Enter stock symbols (comma-separated): ").upper().split(',')
    stock_quantities = input("Enter stock quantities (comma-separated, in the same order): ").split(',')
    
    # Convert stock quantities to integers
    stock_quantities = [int(quantity.strip()) for quantity in stock_quantities]
    
    if len(stock_symbols) != len(stock_quantities):
        print("Error: The number of symbols and quantities must match.")
        return
    
    for symbol, quantity in zip(stock_symbols, stock_quantities):
        symbol = symbol.strip()  # Remove any extra whitespace
        try:
            # Fetch historical data and real-time price
            stock_data = fetch_historical_data(symbol)
            real_time_price = fetch_real_time_data(symbol)
            
            # Train the ARIMA model
            arima_model = train_arima_model(stock_data)
            
            # Make prediction for the next day
            predicted_price = make_prediction(arima_model)
            
            # Calculate potential profit/loss
            potential_profit_loss = (predicted_price - real_time_price) * quantity
            
            print(f"\nStock: {symbol}")
            print(f"Real-time price: {real_time_price:.2f}")
            print(f"Predicted price for the next day: {predicted_price:.2f}")
            print(f"Potential profit/loss for {quantity} shares: {potential_profit_loss:.2f}")
        
        except ValueError as e:
            print(e)

# Run the main function
if __name__ == "__main__":
    main()
