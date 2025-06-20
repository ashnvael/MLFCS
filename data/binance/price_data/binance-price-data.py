import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import hmac
import hashlib
from urllib.parse import urlencode
import os
import json

class BinanceAPI:
    def __init__(self, api_key=None, api_secret=None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.binance.com"
        
        # List of coins we consider
        self.popular_tokens = [
            "ETH", "UNI", "LINK", "SHIB", "AAVE"
        ]

    def _get_signature(self, params):
        """Generate HMAC SHA256 signature for authenticated endpoints"""
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _make_request(self, endpoint, params=None, method="GET", signed=False):
        url = f"{self.base_url}{endpoint}"
        headers = {}
        
        if self.api_key and (signed or endpoint.startswith('/api/v3/account')):
            headers['X-MBX-APIKEY'] = self.api_key
            
        if signed and self.api_secret:
            if params is None:
                params = {}
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._get_signature(params)
            
        try:
            if method == "GET":
                response = requests.get(url, headers=headers, params=params)
            elif method == "POST":
                response = requests.post(url, headers=headers, params=params)
            elif method == "DELETE":
                response = requests.delete(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response: {e.response.text}")
            return None
    
    def get_exchange_info(self):
        """Get exchange info including all trading pairs"""
        return self._make_request("/api/v3/exchangeInfo")
    
    def get_available_usdc_pairs(self):
        """Get all available pairs with USDC for the popular tokens"""
        exchange_info = self.get_exchange_info()
        available_pairs = []
        
        if exchange_info and 'symbols' in exchange_info:
            for symbol_info in exchange_info['symbols']:
                symbol = symbol_info['symbol']
                base_asset = symbol_info['baseAsset']
                quote_asset = symbol_info['quoteAsset']
                
                # Check if pair is one of our popular tokens with USDC
                if (base_asset in self.popular_tokens and quote_asset == "USDC") or \
                   (quote_asset in self.popular_tokens and base_asset == "USDC"):
                    if symbol_info['status'] == 'TRADING':
                        available_pairs.append(symbol)
        
        return available_pairs
    
    def get_klines(self, symbol, interval="1m", limit=1000, start_time=None, end_time=None):
        """
        Get candlestick/kline data for a symbol
        
        Intervals:
        - For minute data: 1m, 3m, 5m, 15m, 30m
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        if start_time:
            params["startTime"] = int(start_time * 1000) if start_time > 10**10 else int(start_time)
        if end_time:
            params["endTime"] = int(end_time * 1000) if end_time > 10**10 else int(end_time)
            
        kline_data = self._make_request("/api/v3/klines", params=params)
        
        if kline_data:
            df = pd.DataFrame(kline_data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                              'quote_asset_volume', 'taker_buy_base_asset_volume', 
                              'taker_buy_quote_asset_volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            return df
        return None
    
    def fetch_historical_data(self, symbol, interval="1m", look_back_days=365):
        """
        Fetch historical data for the past year in chunks to handle API limitations
        """

        end_time = int(time.time() * 1000)  # Current time in milliseconds
        start_time = end_time - (look_back_days * 24 * 60 * 60 * 1000)  # go back X days
        
        # Binance limit is 1000 candles per request!
        # For 1m data, each day has 1440 minutes, so we need to fetch in chunks, 
        # each chunk should cover 1000 minutes
        chunk_size = 1000 * 60 * 1000  # 1000 minutes in milliseconds
        
        all_df = pd.DataFrame()
        current_start = start_time
        
        while current_start < end_time:
            current_end = min(current_start + chunk_size, end_time)
            
            print(f"Fetching {symbol} data from {datetime.fromtimestamp(current_start/1000)} to {datetime.fromtimestamp(current_end/1000)}")
            
            chunk_df = self.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=current_end,
                limit=1000
            )
            
            if chunk_df is not None and not chunk_df.empty:
                all_df = pd.concat([all_df, chunk_df])
                if not chunk_df.empty:
                    # Add 1 minute
                    current_start = int(chunk_df['close_time'].iloc[-1].timestamp() * 1000) + 60000
                else:
                    # If no data, move forward
                    current_start = current_end
            else:
                # If API call failed continue
                current_start = current_end
            
            # Sleep to avoid hitting rate limits
            time.sleep(0.3)
        
        # Remove any duplicate rows based on open_time
        if not all_df.empty:
            all_df = all_df.drop_duplicates(subset=['open_time'])
            all_df = all_df.sort_values('open_time')
            
        return all_df
    
    def fetch_all_token_prices_historical(self, interval="1m", look_back_days=365):
        """Fetch historical price data for all popular tokens paired with USDC"""
        available_pairs = self.get_available_usdc_pairs()
        results = {}
        
        for pair in available_pairs:
            print(f"\n--- Fetching historical data for {pair} ---")
            data = self.fetch_historical_data(pair, interval=interval, look_back_days=look_back_days)
            if data is not None and not data.empty:
                results[pair] = data
                print(f"Successfully fetched {len(data)} records for {pair}")
            else:
                print(f"No data found for {pair}")
            
            # Sleep so we dont get banned :) 
            time.sleep(1)
            
        return results
    
    def save_to_csv(self, data_dict, folder="coinUSDC-price-data"):
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created directory: {folder}")
            
        timestamp = datetime.now().strftime("%Y%m%d")
        
        for symbol, df in data_dict.items():
            if df is not None and not df.empty:
                filename = f"{folder}/{symbol}_{timestamp}.csv"
                df.to_csv(filename)
                print(f"Saved {symbol} data to {filename} ({len(df)} records)")
            else:
                print(f"No data to save for {symbol}")

if __name__ == "__main__":
    api = BinanceAPI()
    
    # Check which pairs are available
    pairs = api.get_available_usdc_pairs()
    print(f"Available USDC pairs for popular tokens: {pairs}")
    
    # Fetch historical minute data for the past year for all available pairs
    print("\nFetching historical minute data for the past year...")
    historical_data = api.fetch_all_token_prices_historical(interval="1m", look_back_days=365)
    
    # Save the data to CSV files
    print("\nSaving data to CSV files...")
    api.save_to_csv(historical_data, folder="coinUSDC-price-data")
    
    print("\nProcess completed!")
