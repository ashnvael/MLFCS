import pandas as pd
import numpy as np
import time
import hmac
import hashlib
import requests
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("binance_data_fetcher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BinanceDataFetcher:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.binance.com"
        self.fapi_url = "https://fapi.binance.com"  # Futures API
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
    def _get_signature(self, params: Dict) -> str:
        """Generate signature for authenticated requests."""
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _make_request(self, endpoint: str, params: Dict = None, signed: bool = False, futures: bool = False) -> Dict:
        """Make request to Binance API with rate limit handling."""
        url = self.fapi_url if futures else self.base_url
        full_url = f"{url}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key}
        
        if params is None:
            params = {}
            
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._get_signature(params)
        
        try:
            response = requests.get(full_url, headers=headers, params=params)
            data = response.json()
            
            # Handle API errors
            if response.status_code != 200:
                logger.error(f"API Error: {data}")
                if 'code' in data and data['code'] == -1003:  # Rate limit error
                    logger.info("Rate limit hit, sleeping for 60 seconds")
                    time.sleep(60)
                    return self._make_request(endpoint, params, signed, futures)
                return data
            
            return data
        except Exception as e:
            logger.error(f"Request error: {e}")
            time.sleep(10)  # Sleep on error
            return self._make_request(endpoint, params, signed, futures)
    
    def get_account_info(self) -> Dict:
        """Get account information including fee tier."""
        endpoint = "/api/v3/account"
        return self._make_request(endpoint, signed=True)
    
    def get_futures_account_info(self) -> Dict:
        """Get futures account information."""
        endpoint = "/fapi/v2/account"
        return self._make_request(endpoint, signed=True, futures=True)
    
    def get_klines(self, symbol: str, interval: str, start_time: int, end_time: int, futures: bool = False) -> List:
        """Get klines/candlestick data."""
        endpoint = "/fapi/v1/klines" if futures else "/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000  # Maximum allowed
        }
        
        return self._make_request(endpoint, params, futures=futures)
    
    def get_all_klines(self, symbol: str, interval: str, start_date: datetime, end_date: datetime, futures: bool = False) -> pd.DataFrame:
        """Get all klines between start and end date, handling pagination."""
        all_klines = []
        current_start = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)
        
        while current_start < end_timestamp:
            # Calculate end time for this batch
            current_end = min(current_start + (1000 * 60 * 1000), end_timestamp)  # 1000 minutes in ms
            
            klines = self.get_klines(symbol, interval, current_start, current_end, futures)
            if not klines:
                break
                
            all_klines.extend(klines)
            logger.info(f"Fetched {len(klines)} klines from {datetime.fromtimestamp(current_start/1000)}")
            
            # Prepare for next batch, add 1ms to avoid duplicates
            current_start = int(klines[-1][0]) + 1
            
            # Avoid rate limits
            time.sleep(0.5)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert timestamps to datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Convert string values to numeric
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_asset_volume', 'taker_buy_base_asset_volume', 
                          'taker_buy_quote_asset_volume']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
        
        return df
    
    def get_funding_rate_history(self, symbol: str, start_time: int, end_time: int) -> List:
        """Get funding rate history for a perpetual futures symbol."""
        endpoint = "/fapi/v1/fundingRate"
        params = {
            "symbol": symbol,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000  # Maximum allowed
        }
        
        return self._make_request(endpoint, params, futures=True)
    
    def get_all_funding_rates(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get all funding rates between start and end date, handling pagination."""
        all_funding_rates = []
        current_start = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)
        
        while current_start < end_timestamp:
            # Calculate end time for this batch (30 days at a time to respect limits)
            current_end = min(current_start + (30 * 24 * 60 * 60 * 1000), end_timestamp)
            
            funding_rates = self.get_funding_rate_history(symbol, current_start, current_end)
            if not funding_rates:
                break
                
            all_funding_rates.extend(funding_rates)
            logger.info(f"Fetched {len(funding_rates)} funding rates from {datetime.fromtimestamp(current_start/1000)}")
            
            # Prepare for next batch, add 1ms to avoid duplicates
            if funding_rates:
                current_start = int(funding_rates[-1]['fundingTime']) + 1
            else:
                current_start = current_end + 1
            
            # Avoid rate limits
            time.sleep(0.5)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_funding_rates)
        
        if not df.empty:
            # Convert timestamps to datetime
            df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
            
            # Convert string values to numeric
            df['fundingRate'] = pd.to_numeric(df['fundingRate'])
        
        return df
    
    def get_fee_tier_info(self) -> Dict:
        """Get fee tier information for the account."""
        endpoint = "/sapi/v1/asset/tradeFee"
        params = {}
        return self._make_request(endpoint, params, signed=True)
    
    def get_perpetual_trading_data(self, symbol_pair: str, lookback_days: int = 365):
        """
        Fetch all relevant data for analyzing perpetual futures hedging strategy.
        
        Args:
            symbol_pair: Trading pair (e.g., 'ETHUSDC')
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary containing all the fetched data
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Ensure symbol format is correct for different endpoints
        spot_symbol = symbol_pair
        futures_symbol = symbol_pair
        
        # For ETH/USDC specifically
        if symbol_pair == "ETHUSDC":
            # Some endpoints may require different formats, adjust as needed
            spot_symbol = "ETHUSDC"
            futures_symbol = "ETHUSDC"  # or "ETHUSDC_PERP" depending on Binance's current format
        
        logger.info(f"Fetching data for {symbol_pair} from {start_date} to {end_date}")
        
        # 1. Get spot market minute-level data
        logger.info("Fetching spot market data...")
        spot_data = self.get_all_klines(spot_symbol, "1m", start_date, end_date, futures=False)
        spot_data.to_csv(f"data/{symbol_pair}_spot_minute_data.csv", index=False)
        
        # 2. Get perpetual futures minute-level data
        logger.info("Fetching futures market data...")
        futures_data = self.get_all_klines(futures_symbol, "1m", start_date, end_date, futures=True)
        futures_data.to_csv(f"data/{symbol_pair}_futures_minute_data.csv", index=False)
        
        # 3. Get funding rate history (8-hour intervals)
        logger.info("Fetching funding rate history...")
        funding_data = self.get_all_funding_rates(futures_symbol, start_date, end_date)
        funding_data.to_csv(f"data/{symbol_pair}_funding_rate_history.csv", index=False)
        
        # 4. Get account fee information
        logger.info("Fetching account fee information...")
        try:
            fee_info = self.get_fee_tier_info()
            with open(f"data/{symbol_pair}_fee_info.json", 'w') as f:
                import json
                json.dump(fee_info, f)
        except Exception as e:
            logger.error(f"Failed to fetch fee information: {e}")
        
        # 5. Get account information including margin requirements
        logger.info("Fetching account information...")
        try:
            account_info = self.get_account_info()
            futures_account_info = self.get_futures_account_info()
            
            with open(f"data/account_info.json", 'w') as f:
                import json
                json.dump(account_info, f)
                
            with open(f"data/futures_account_info.json", 'w') as f:
                import json
                json.dump(futures_account_info, f)
        except Exception as e:
            logger.error(f"Failed to fetch account information: {e}")
        
        return {
            "spot_data": spot_data,
            "futures_data": futures_data,
            "funding_data": funding_data
        }
    
    def analyze_data(self, symbol_pair: str):
        """
        Analyze the fetched data to determine costs and effectiveness of hedging.
        
        Args:
            symbol_pair: Trading pair (e.g., 'ETHUSDC')
        """
        # Load the saved data
        spot_data = pd.read_csv(f"data/{symbol_pair}_spot_minute_data.csv")
        futures_data = pd.read_csv(f"data/{symbol_pair}_futures_minute_data.csv")
        funding_data = pd.read_csv(f"data/{symbol_pair}_funding_rate_history.csv")
        
        # Convert timestamps to datetime
        spot_data['open_time'] = pd.to_datetime(spot_data['open_time'])
        futures_data['open_time'] = pd.to_datetime(futures_data['open_time'])
        funding_data['fundingTime'] = pd.to_datetime(funding_data['fundingTime'])
        
        # Calculate price differences between spot and futures
        # First, align the data by timestamp
        merged_data = pd.merge_asof(
            spot_data[['open_time', 'close']].rename(columns={'close': 'spot_price'}),
            futures_data[['open_time', 'close']].rename(columns={'close': 'futures_price'}),
            on='open_time',
            direction='nearest'
        )
        
        # Calculate basis (difference between futures and spot)
        merged_data['basis'] = merged_data['futures_price'] - merged_data['spot_price']
        merged_data['basis_percent'] = (merged_data['basis'] / merged_data['spot_price']) * 100
        
        # Calculate volatility (standard deviation of returns)
        spot_data['returns'] = spot_data['close'].pct_change()
        spot_volatility = spot_data['returns'].std() * (365 * 24 * 60) ** 0.5  # Annualized volatility
        
        # Calculate average funding rate and annualized cost
        avg_funding_rate = funding_data['fundingRate'].mean()
        annualized_funding_cost = avg_funding_rate * (365 * 3)  # 3 fundings per day, 365 days
        
        # Calculate rebalancing frequency based on price movements
        # Assume rebalance when price moves by more than 2%
        spot_data['price_move'] = spot_data['close'].pct_change().abs()
        rebalance_events = spot_data[spot_data['price_move'] > 0.02]
        rebalances_per_day = len(rebalance_events) / (len(spot_data) / (24 * 60))
        
        # Create summary report
        summary = {
            "average_spot_price": spot_data['close'].mean(),
            "average_futures_price": futures_data['close'].mean(),
            "average_basis_percent": merged_data['basis_percent'].mean(),
            "annualized_volatility": spot_volatility * 100,  # as percentage
            "average_funding_rate": avg_funding_rate,
            "annualized_funding_cost": annualized_funding_cost * 100,  # as percentage
            "estimated_rebalances_per_day": rebalances_per_day,
            "total_days_analyzed": len(spot_data) / (24 * 60)
        }
        
        # Save summary to file
        pd.DataFrame([summary]).to_csv(f"data/{symbol_pair}_analysis_summary.csv", index=False)
        
        # Create some basic plots
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Spot vs Futures Price
        plt.subplot(2, 2, 1)
        plt.plot(spot_data['open_time'], spot_data['close'], label='Spot Price')
        plt.plot(futures_data['open_time'], futures_data['close'], label='Futures Price')
        plt.title('Spot vs Futures Price')
        plt.legend()
        plt.xticks(rotation=45)
        
        # Plot 2: Basis Percentage
        plt.subplot(2, 2, 2)
        plt.plot(merged_data['open_time'], merged_data['basis_percent'])
        plt.title('Basis Percentage')
        plt.xticks(rotation=45)
        
        # Plot 3: Funding Rates
        plt.subplot(2, 2, 3)
        plt.plot(funding_data['fundingTime'], funding_data['fundingRate'] * 100)
        plt.title('Funding Rates (%)')
        plt.xticks(rotation=45)
        
        # Plot 4: Cumulative Funding Cost
        plt.subplot(2, 2, 4)
        funding_data['cumulative_cost'] = funding_data['fundingRate'].cumsum() * 100
        plt.plot(funding_data['fundingTime'], funding_data['cumulative_cost'])
        plt.title('Cumulative Funding Cost (%)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"data/{symbol_pair}_analysis_plots.png")
        
        return summary


# Example usage
if __name__ == "__main__":
    # Replace with your API key and secret
    api_key = "get it on binance"
    api_secret = "get it on binance"
    
    fetcher = BinanceDataFetcher(api_key, api_secret)
    
    # Fetch data for ETH/USDC pair for the past year
    data = fetcher.get_perpetual_trading_data("ETHUSDC", lookback_days=365)
    
    # Analyze the data
    summary = fetcher.analyze_data("ETHUSDC")
    
    print("Analysis complete! Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
