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
    
    def get_klines(self, symbol, interval="1m", limit=1000, start_time=None, end_time=None):
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        if start_time:
            params["startTime"] = int(start_time)
        if end_time:
            params["endTime"] = int(end_time)
            
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
    
    def fetch_historical_data(self, symbol, interval="1m", start_time=None, end_time=None):
        if start_time is None or end_time is None:
            raise ValueError("start_time and end_time must be provided (in ms)")
        
        chunk_size = 1000 * 60 * 1000  # 1000 minutes in milliseconds
        all_df = pd.DataFrame()
        current_start = start_time
        
        while current_start < end_time:
            current_end = min(current_start + chunk_size, end_time)
            print(f"Fetching {symbol} data from {datetime.fromtimestamp(current_start/1000)} to {datetime.fromtimestamp(current_end/1000)}")
            
            chunk_df = self.get_klines(
                symbol=symbol,
                interval=interval,
                limit=1000,
                start_time=current_start,
                end_time=current_end
            )
            
            if chunk_df is not None and not chunk_df.empty:
                all_df = pd.concat([all_df, chunk_df])
                last_close = int(chunk_df['close_time'].iloc[-1].timestamp() * 1000)
                current_start = last_close + 60 * 1000
            else:
                current_start = current_end
            
            time.sleep(0.3)
        
        if not all_df.empty:
            all_df = all_df.drop_duplicates(subset=['open_time']).sort_values('open_time')
        
        return all_df
    
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
    api = BinanceAPI(
        api_key="Qp9qC8T682Tv0Jp9sadltkTPrfSsAYRrgUw9aGdPG6t0nCr8xT1xOrIqPqAZUxcT"
    )

    # Define fixed date range: August 1, 2021 to July 31, 2022
    start_dt = datetime(2021, 8, 1)
    end_dt = datetime(2022, 7, 31, 23, 59, 59)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    
    symbol = "ETHUSDC"
    print(f"\nFetching historical minute data for {symbol} from {start_dt.date()} to {end_dt.date()}...")
    data = api.fetch_historical_data(
        symbol=symbol,
        interval="1m",
        start_time=start_ms,
        end_time=end_ms
    )
    
    historical_data = {symbol: data} if data is not None else {}
    
    print("\nSaving data to CSV files...")
    api.save_to_csv(historical_data, folder="nikolai-price-data")
    
    print("\nProcess completed!")
