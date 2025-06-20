import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import glob
from config.env_config import Config

HEDGING_FUTURES_PRICE_PATH = "data/binance/hedging_data/data/ETHUSDC_futures_minute_data.csv"
HEDGING_FUNDING_RATE_PATH = "data/binance/hedging_data/data/ETHUSDC_funding_rate_history.csv"
UNISWAP_POOL_EVENTS_PATH = "uniswap_lp_data"
ETH_USDC_PRICE_PATH = "data/binance/price_data/coinUSDC-price-data/ETHUSDC_20250316.csv"


class UniswapV3LPGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config: Config = None, feat_num: int = 19):
        super().__init__()
        self.config = config if config is not None else Config()
        self.initial_wealth = self.config.WEALTH
        self.wealth = None # It is set during the reset function, pd.Series with one obs. 
        self.tau = self.config.TAU
        self.FEAT_NUM = feat_num
        self.prev_action = np.array([0, 0, 0])

        self.load_data()
        self.initialize_decision_grid()

        self.action_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([1, np.finfo(np.float32).max, np.finfo(np.float32).max],
                          dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.FEAT_NUM,),
            dtype=np.float32
        )

        self.current_time_index = 0
        self.last_timestamp = None
        self.cumulative_pnl = 0.0

    def load_data(self):
        self.uniswap_lp_data = pd.read_csv("data/uniswap/uniswap_lp_data_1.csv")
        self.uniswap_lp_data['timestamp'] = pd.to_datetime(self.uniswap_lp_data['timestamp'], unit='s')

        # Load Binance futures and spot data. For these, we'll assume open_time is already in a string format.
        self.binance_futures_data = pd.read_csv(
            "data/binance/hedging_data/data/ETHUSDC_futures_minute_data.csv",
            parse_dates=["open_time"]
        )
        self.binance_spot_data = pd.read_csv(
            "data/binance/hedging_data/data/ETHUSDC_spot_minute_data.csv",
            parse_dates=["open_time"]
        )

    def initialize_decision_grid(self):
        start_dt = self.uniswap_lp_data['timestamp'].min()
        end_dt = self.uniswap_lp_data['timestamp'].max()
        self.decision_grid = pd.date_range(start=start_dt, end=end_dt, freq='1min')

    def form_observable_data(self, timestamp):
        binance_futures = self.binance_futures_data[self.binance_futures_data['open_time'] <= timestamp]
        if binance_futures.empty:
            raise ValueError(f"No binance_futures data available up to timestamp {timestamp}")
        binance_spot = self.binance_spot_data[self.binance_spot_data['open_time'] <= timestamp]
        if binance_spot.empty:
            raise ValueError(f"No binance_spot data available up to timestamp {timestamp}")
        uniswap_lp = self.uniswap_lp_data[self.uniswap_lp_data['timestamp'] <= timestamp]
        if uniswap_lp.empty:
            raise ValueError(f"No uniswap_lp_data available up to timestamp {timestamp}")

        return (binance_futures, binance_spot, uniswap_lp)

    def form_observable_features(self, timestamp):
        features = np.zeros(self.FEAT_NUM, dtype=np.float32)

        (binance_futures, binance_spot, uniswap_lp) = self.form_observable_data(timestamp)
        
        ## now code which takes existing data and fills in features data
        ## e.g.
        features[0] = np.log(binance_futures.close.iloc[-1] / binance_futures.close.iloc[-1])
        features[1] = 3
        return features
    
    @staticmethod
    def _event_gas_fees(event_name, current_timestamp, num_of_events=20):
        """
            Calculate the average gas fees in USDC for a given event.
        """
        
        file_paths = glob.glob(f"{UNISWAP_POOL_EVENTS_PATH}/uniswap_lp_data_*.csv")
        columns_to_use = ["timestamp", "event", "gas_eth"]
        events_data = []

        for path in file_paths:
            # Get first timestamp
            first_row = pd.read_csv(path, usecols=["timestamp"], nrows=1)
            first_ts = pd.to_datetime(first_row["timestamp"].iloc[0])

            # Get last timestamp 
            last_row = pd.read_csv(path, usecols=["timestamp"], skiprows=lambda i: i != 0 and i != sum(1 for _ in open(path)) - 1)
            last_ts = pd.to_datetime(last_row["timestamp"].iloc[-1])

            # If current_timestamp is within range, read and filter
            if first_ts <= current_timestamp and last_ts >= current_timestamp:
                df = pd.read_csv(path, usecols=columns_to_use)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                filtered = df[
                    (df["timestamp"] <= current_timestamp) &
                    (df['event'] == event_name)
                ]
                events_data.append(filtered)

        # Filter most recent events
        result_df = pd.concat(events_data, ignore_index=True)
        recent_events = result_df.sort_values("timestamp", ascending=False).head(num_of_events)

        avg_gas_fee = recent_events["gas_eth"].mean()

        # ETH price in USDT from Binance
        price_data = pd.read_csv(ETH_USDC_PRICE_PATH, usecols=["open_time", "open"])
        price_data["open_time"] = pd.to_datetime(price_data["open_time"])
        matched_row = price_data[price_data["open_time"] == current_timestamp]
        current_eth_price = matched_row["open"].iloc[0]

        return avg_gas_fee * current_eth_price

    @staticmethod
    def _hedging_transaction_costs(current_timestamp, eth_position_size):
        # Get current ETH USDC futures price
        eth_futures_price = pd.read_csv(HEDGING_FUTURES_PRICE_PATH, usecols=["open_time", "open"])
        eth_futures_price["open_time"] = pd.to_datetime(eth_futures_price["open_time"])
        matched_row = eth_futures_price[eth_futures_price["open_time"] == current_timestamp]
        current_eth_futures_price = matched_row["open"].iloc[0]

        return - eth_position_size * current_eth_futures_price * 0.0004 # Price taker transaction costs
        
    @staticmethod
    def _hedging_funding_rate(current_timestamp, eth_position_size):
        funding_times = ["00:00", "08:00", "16:00"]

        # Transform current_timestamp to correct format
        hour = current_timestamp.hour
        minute = current_timestamp.minute
        time = f"{hour:02}:{minute:02}"

        if time in funding_times:
            funding_rate = pd.read_csv(HEDGING_FUNDING_RATE_PATH)
            funding_rate["fundingTime"] = pd.to_datetime(funding_rate["fundingTime"])
            funding_rate = funding_rate[funding_rate['fundingTime'] == current_timestamp]
            return - eth_position_size * funding_rate["fundingRate"] * funding_rate["markPrice"]
        return 0 
        
    @staticmethod
    def _hedging_rebalancing_costs(current_timestamp, eth_position_size, previous_eth_position_size):

        # Get current ETH USDC futures price
        eth_futures_price = pd.read_csv(HEDGING_FUTURES_PRICE_PATH, usecols=["open_time", "open"])
        eth_futures_price["open_time"] = pd.to_datetime(eth_futures_price["open_time"])
        matched_row = eth_futures_price[eth_futures_price["open_time"] == current_timestamp]
        current_eth_futures_price = matched_row["open"].iloc[0]

        # Compute the change in our ETH holdings 
        change_eth_hodlings = abs(eth_position_size - previous_eth_position_size)

        return - change_eth_hodlings * current_eth_futures_price * 0.0004 # Price taker transaction costs
            

    @staticmethod   
    def _impermanent_loss():
        # We have to store fraction of the pool we own when we initially provide liquidity and check the current composition and compare the money we would have un usd 
        # We calculate the deltaImpermanentLoss over a minute
        previous_imp_loss = 0
        inital_ETH = 0
        initial_USDC = 0
        pool_share = 0
        current_pool_composition = 0 
        current_ETH = 0
        current_USDC = 0
        conversion_rate = 0 # ETH to USDC conversion rate 
        return previous_imp_loss - ((current_ETH * conversion_rate + current_USDC) - (inital_ETH * conversion_rate + initial_USDC))

    @staticmethod
    def _pool_fees():
        return 0

    def compute_pnl(self, current_timestamp, action):
        """
            Compute the total PnL over the last minute based on the action taken.

            There are four possible cases:

            1. From No Position to No Position:
               - `action_{t-1} = (0, ., .)` → `action_t = (0, ., .)`
               - No change in position, so PnL is 0.

            2. From No Position to Providing Liquidity:
               - `action_{t-1} = (0, ., .)` → `action_t = (1, tick_l, tick_u)`
               - Actions:
                   • Pay gas fees for adding liquidity.
                   • Open hedging position → pay transaction costs and possibly funding rate.
                   • Earn fees from swap events.
                   • Incur impermanent loss.

            3. From Providing Liquidity to Same Position:
               - `action_{t-1} = (1, tick_l, tick_u)` → `action_t = (1, tick_l, tick_u)`
               - Actions:
                   • Earn fees from position.
                   • Pay funding rate for hedging (if applicable).
                   • Pay transaction costs for rebalancing hedge.
                   • Incur impermanent loss.

            4. From Providing Liquidity to No Position:
               - `action_{t-1} = (1, tick_l, tick_u)` → `action_t = (0, ., .)`
               - Actions:
                   • Pay gas fees for removing liquidity (Burn event).
                   • Close hedge → pay transaction costs.
                   • Pay gas fees related to fee collection over the position lifetime.
        """

        # Previous decision 
        previous_action = self.prev_action[0]
        previous_lower_tick = self.prev_action[1]
        previous_upper_tick = self.prev_action[2]

        next_action = action[0]
        next_lower_tick = action[1]
        next_upper_tick = action[2]

        if previous_action == 0:
            if next_action == 0:
                # Case 1. 
                return 0
            # Case 2.
            return (self._pool_fees() 
                    - self._event_gas_fees("Mint", current_timestamp) 
                    - self._hedging_transaction_costs(current_timestamp, self.eth_position_size[-1]) 
                    - self._hedging_funding_rate(current_timestamp, self.eth_position_size[-1])
                    - self._impermanent_loss()
                )

        # When history is 1
        if previous_action == 1:
            if next_action == 1:
                # Case 3.
                return (self._pool_fees() 
                        - self._hedging_funding_rate(current_timestamp, self.eth_position_size[-1]) 
                        - self._hedging_rebalancing_costs(current_timestamp, self.eth_position_size[-1], self.eth_position_size[-2])
                        - self._impermanent_loss()
                )
            # Case 4. 
            return (- self._event_gas_fees("Burn", current_timestamp) 
                    - self._event_gas_fees("Collect", current_timestamp) 
                    - self._hedging_transaction_costs(current_timestamp, self.eth_position_size[-1])
                )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        total_decisions = len(self.decision_grid)
        self.episode_length = max(1, int(0.1 * total_decisions))
        max_start = total_decisions - self.episode_length

        self.current_time_index = np.random.randint(0, max_start + 1)

        self.wealth = pd.Series([self.initial_wealth], 
                                index=[self.decision_grid[self.current_time_index]])


        current_timestamp = self.decision_grid[self.current_time_index]
        self.last_timestamp = current_timestamp
        self.cumulative_pnl = 0.0

        observable_features = self.form_observable_features(current_timestamp)
        return observable_features, {}

    def step(self, action):

        prev_timestamp = self.last_timestamp
        current_timestamp = self.decision_grid[self.current_time_index]
        
        pnl = self.compute_pnl(current_timestamp, action)
        self.cumulative_pnl += pnl
        
        self.last_timestamp = current_timestamp

        self.current_time_index += 1
        done = self.current_time_index >= len(self.decision_grid)
        next_observation = None
        if not done:
            next_timestamp = self.decision_grid[self.current_time_index]
            new_s = pd.Series([self.wealth.iloc[-1] + pnl], index=[next_timestamp])
            self.wealth = pd.concat([self.wealth, new_s])
            next_observation = self.form_observable_features(next_timestamp)
            self.prev_action = action

        return next_observation, pnl, done, False, {}

    def render(self, mode="human"):
        print(f"Time Index: {self.current_time_index}, Cumulative PnL: {self.cumulative_pnl}")

    def close(self):
        pass
