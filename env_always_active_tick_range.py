import glob
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from config.env_config import Config

ETH_USDC_PRICE_PATH = "data/binance/price_data/coinUSDC-price-data/ETHUSDC_20250316.csv"
UNISWAP_SAMPLE_PATH = "uniswap_lp_data/sorted_uniswap_data1_truncated.csv"
FEE_TABLE_PATH = "data/uniswap/fee_table.parquet"
FUNDING_RATE_PATH = "data/binance/hedging_data/data/ETHUSDC_funding_rate_history.csv"
ETH_USDC_FUTURES_PATH = "data/binance/hedging_data/data/ETHUSDC_futures_minute_data.csv"
POOL_FEE_TIER = 0.0005
HEDGING_COST = 0.00036
FUNDING_TIMES = [
    pd.Timestamp("2025-06-28 00:00:00"),
    pd.Timestamp("2025-06-28 08:00:00"),
    pd.Timestamp("2025-06-28 16:00:00")
]
DECIMALS_TOKEN0, DECIMALS_TOKEN1 = 6, 18          # USDC / WETH
DEC_FACTOR = 10 ** (DECIMALS_TOKEN1 - DECIMALS_TOKEN0)  # 1_000_000_000_000
SQRTDEC_FACTOR = 10 ** ((DECIMALS_TOKEN1 - DECIMALS_TOKEN0) // 2)  

LOG_1P0001 = np.log(1.0001)

def tick_to_price(tick: int | float) -> float:
    """Tick → dollar price (USDC per ETH)."""
    return DEC_FACTOR / (1.0001 ** tick)

def ticks_to_sqrtp(tick: int) -> float:
    """Uniswap √-price corresponding to *token1/token0* (WETH/USDC)."""
    # print(tick_to_price(tick))
    return SQRTDEC_FACTOR / (1.0001 ** (tick//2))  

def price_to_sqrtPriceX96(p):               
    return np.sqrt(p)*2**96

# def _group_concat(frames):
#     return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def _build_fee_table(csv_path: str) -> pd.DataFrame:
    """
    Build a 1-minute fee grid from *sorted_uniswap_data1.csv*.
    Only rows whose event_type == 'Swap' are used.
    """
    raw = (
        pd.read_csv(
            csv_path,
            usecols=["timestamp", "event_type",
                     "amount0", "amount1", "liquidity", "tick"],
            low_memory=False,
        )
        .query("event_type == 'Swap'")
        .assign(timestamp=lambda df: pd.to_datetime(df["timestamp"], unit="s"))
    )

    for col in ("amount0", "amount1", "liquidity"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw.dropna(subset=["amount0", "amount1", "liquidity"], inplace=True)

    raw["fee0"] = raw["amount0"].clip(lower=0) * POOL_FEE_TIER
    raw["fee1"] = raw["amount1"].clip(lower=0) * POOL_FEE_TIER

    fee_grid = (
        raw.set_index("timestamp")
            .resample("1min")
            .agg({"fee0": "sum",
                  "fee1": "sum",
                  "liquidity": "mean",
                  "tick": "mean"}) # TODO: think, use last?
            .rename(columns={"liquidity": "liquidity_pool",
                             "tick": "tick_close"})
    )

    # forward–fill pool liquidity & last tick, fill missing fees with 0
    fee_grid["liquidity_pool"] = fee_grid["liquidity_pool"].ffill()
    fee_grid["tick_close"]     = fee_grid["tick_close"].ffill()
    fee_grid.fillna({"fee0": 0.0, "fee1": 0.0}, inplace=True)
    return fee_grid


class UniswapV3LPGymEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, config: Config | None = None, feat_num: int | None = None):
        super().__init__()

        self.config = config or Config()
        self.current_wealth_in_USDC = self.config.WEALTH
        self.FEAT_NUM = feat_num or 6 
        self.EPISODE_LEN = 1000

        self.cumulative_pnl = 0.0
        self.active = False
        self.L = 0.0
        self.tick_l = 0
        self.tick_u = 0
        self.x_prev = 0.0
        self.y_prev = 0.0

        self._load_data()
        self._build_decision_grid()

        self.action_space = spaces.Box(
            low=np.array([50.0, 50.0],  dtype=np.float32),
            high=np.array([100.0, 100.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.FEAT_NUM,), dtype=np.float32
        )

        self.idx = 0
        self.steps_left = 0

    def _load_data(self):

        cex = pd.read_csv(ETH_USDC_PRICE_PATH, usecols=["open_time", "close"])
        cex["open_time"] = pd.to_datetime(cex["open_time"])        
        self.eth_px = cex.set_index("open_time")

        dex_tick = pd.read_csv(UNISWAP_SAMPLE_PATH, usecols=["timestamp", "tick"])
        dex_tick["timestamp"] = pd.to_datetime(dex_tick["timestamp"])
        self.lp_span = dex_tick.set_index("timestamp")

        futures_data = pd.read_csv(ETH_USDC_FUTURES_PATH, usecols=["open_time", "close"])
        funding_rate_data = pd.read_csv(FUNDING_RATE_PATH, usecols=["fundingTime","fundingRate","markPrice"])
        funding_rate_data["fundingTime"] = pd.to_datetime(funding_rate_data["fundingTime"])
        futures_data["open_time"] = pd.to_datetime(futures_data["open_time"])
        self.futures_data = pd.merge(
            futures_data,
            funding_rate_data,
            left_on="open_time",
            right_on="fundingTime",
            how="left"
        )
        self.futures_data = self.futures_data.set_index("open_time")

        # full raw dataframe (all event types) – we'll slice it later
        dex_raw = pd.read_csv(UNISWAP_SAMPLE_PATH, low_memory=False)
        dex_raw["timestamp"] = pd.to_datetime(dex_raw["timestamp"])
        self.uniswap_lp_data = dex_raw

        start = max(self.eth_px.index.min(), self.lp_span.index.min())
        end   = min(self.eth_px.index.max(), self.lp_span.index.max())

        self.eth_px = self.eth_px.loc[start:end]
        self.lp_span = self.lp_span.loc[start:end]
        self.uniswap_lp_data = self.uniswap_lp_data[
            (self.uniswap_lp_data["timestamp"] >= start)
            & (self.uniswap_lp_data["timestamp"] <= end)
        ]

        fee_frames: dict[str, pd.DataFrame] = {}
        for evt, grp in self.uniswap_lp_data[["timestamp", "event_type", "gas_eth"]].groupby("event_type"):
            fee_frames[evt] = (
                grp.set_index("timestamp")
                .sort_index()
            )
        self.gas_fee = fee_frames

        fee_tbl_path = Path(FEE_TABLE_PATH)
        if fee_tbl_path.exists():
            grid = pd.read_parquet(fee_tbl_path)
        else:
            grid = _build_fee_table(UNISWAP_SAMPLE_PATH)
            fee_tbl_path.parent.mkdir(parents=True, exist_ok=True)
            grid.to_parquet(fee_tbl_path, compression="zstd")
        self.fee_grid = grid.loc[start:end]

    def _build_decision_grid(self):
        start = self.lp_span.index.min()
        end = self.lp_span.index.max()
        self.decision_grid = pd.date_range(start=start, end=end, freq="10min")

    # ------------------------ price & gas helpers ------------------------
    def _eth_price(self, ts: pd.Timestamp) -> float:
        if ts in self.eth_px.index:
            return float(self.eth_px.loc[ts, "close"])
        pos = self.eth_px.index.searchsorted(ts, side="right") - 1
        if pos < 0:
            raise ValueError(f"ETH price not available before {ts}.")
        return float(self.eth_px.iloc[pos]["close"])

    def _gas_cost(self, evt: str, ts) -> float:
        df = self.gas_fee.get(evt)
        if df is None:
            print("Warning: gas fee not available for event type:", evt)
            return 0.0
        last20 = df.loc[:ts].tail(20)
        if last20.empty:
            print("Warning: gas fee not available for event type:", evt)
            return 0.0

        return float(last20["gas_eth"].mean() * self._eth_price(ts))
    
    def _dex_tick(self, ts: pd.Timestamp) -> int:
        """
        Get the average tick from Swap events at or before *ts*.
        If multiple swaps happen at *ts*, their ticks are averaged.
        """
        # Slice only Swap events with valid ticks
        swap_df = self.uniswap_lp_data
        if "event_type" in swap_df.columns:
            swap_df = swap_df[swap_df["event_type"] == "Swap"]

        swap_df = swap_df.dropna(subset=["tick"])
        swap_df = swap_df.set_index("timestamp").sort_index()

        # Exact match: take average of all swaps at this second
        if ts in swap_df.index:
            ticks_at_ts = swap_df.loc[ts, "tick"]
            if isinstance(ticks_at_ts, pd.Series):
                return int(np.nanmean(ticks_at_ts))
            return int(ticks_at_ts)

        # Otherwise: fallback to most recent tick before ts
        pos = swap_df.index.searchsorted(ts, side="right") - 1
        if pos < 0:
            raise ValueError(f"No swap tick data available before {ts}.")
        tick = swap_df.iloc[pos]["tick"]
        return int(tick)


    # ------------------------ pool fee helpers ------------------------
    def _pool_fees(self, ts):
        if ts in self.fee_grid.index:
            row = self.fee_grid.loc[ts]
        else:                        # fetch the last known minute bar
            pos = self.fee_grid.index.searchsorted(ts, side="right") - 1
            if pos < 0:
                return 0.0, 0.0, np.nan
            row = self.fee_grid.iloc[pos]
        tick_close = row.tick_close
        if pd.isna(tick_close):
            return float(row.fee0), float(row.fee1), float(row.liquidity_pool), None
        return float(row.fee0), float(row.fee1), float(row.liquidity_pool), int(tick_close)

    # def _accrue_fees(self, ts: pd.Timestamp):
    #     if not self.active:
    #         return 0.0, 0.0

    #     fee0_pool, fee1_pool, L_pool, tick_close = self._pool_fees(ts)
    #     if (tick_close is None) or (tick_close <= self.tick_l) or (tick_close >= self.tick_u):
    #         return 0.0, 0.0

    #     # not in range → no fees
    #     if not (self.tick_l < tick_close < self.tick_u):
    #         return 0.0, 0.0
    #     if np.isnan(L_pool) or L_pool == 0.0:
    #         return 0.0, 0.0

    #     L_eff = self.L if self.tick_l < tick_close < self.tick_u else 0.0
    #     share = L_eff / L_pool
    #     return share * fee0_pool, share * fee1_pool

    def _accrue_fees(self, ts):
        if not self.active:
            return 0.0, 0.0

        # grab the raw swaps for *that* minute only
        t1 = ts.floor("1min")
        t0 = t1 - pd.Timedelta(minutes=1)
        minute_swaps = self.uniswap_lp_data.query(
            "(event_type == 'Swap') and @t0 <= timestamp < @t1"
        )

        in_range = minute_swaps[
            (minute_swaps["tick"] < self.tick_l) & (minute_swaps["tick"] > self.tick_u)
        ]
        if in_range.empty:
            return 0.0, 0.0

        # fees on the positive leg only (Y = USDC, X = ETH)
        feeY = in_range["amount0"].clip(lower=0).to_numpy() * POOL_FEE_TIER
        feeX = in_range["amount1"].clip(lower=0).to_numpy() * POOL_FEE_TIER

        share = (self.L / (in_range["liquidity"] + self.L)).clip(upper=1.0).to_numpy()
        # print(in_range.shape)
        # print("fee0:", fee0, "fee1", fee1)

        # if np.dot(share, fee0) > 1:
        #     print("total fee0", np.dot(share, fee0), "total fee1", np.dot(share, fee1))
        #     print("num of swaps", in_range.shape[0])
        #     print("transactions:", in_range)

        # print(in_range.shape, in_range["amount1"].sum(), in_range["amount0"].sum())
        # print("fees ETH, USDC", np.dot(share, feeX), np.dot(share, feeY))
        return np.dot(share, feeX), np.dot(share, feeY)

    def _hedge_postion(self, ts):
        pass


    # ------------------------ feature engineering ------------------------
    def form_observable_features(
        self, timestamp: pd.Timestamp, lookback_period: pd.Timedelta = pd.Timedelta(hours=1)
    ) -> np.ndarray:
        beginning = timestamp - lookback_period
        features = np.zeros(self.FEAT_NUM, dtype=np.float32)

        period_df = (
            self.uniswap_lp_data
                .loc[(self.uniswap_lp_data["timestamp"] >= beginning)
                    & (self.uniswap_lp_data["timestamp"] <= timestamp)]
                .copy()              
        )

        period_df["sqrtPriceX96"] = pd.to_numeric(
            period_df["sqrtPriceX96"], errors="coerce"
        )
        swap_df = period_df[period_df["event_type"] == "Swap"].copy()
        swap_df.dropna(subset=["sqrtPriceX96"], inplace=True)
        swap_df["price"] = (swap_df["sqrtPriceX96"] / (2 ** 96)) ** 2

        features[0] = swap_df["price"].mean() if not swap_df.empty else 0.0
        features[1] = swap_df["price"].std(ddof=0) if not swap_df.empty else 0.0
        if len(swap_df) > 1:
            features[2] = swap_df["price"].iloc[-1] - swap_df["price"].iloc[0]
        else:
            features[2] = 0.0

        swap_df["amount0"] = pd.to_numeric(swap_df["amount0"], errors="coerce")
        swap_df["amount1"] = pd.to_numeric(swap_df["amount1"], errors="coerce")
        features[3] = swap_df["amount0"].abs().sum() + swap_df["amount1"].abs().sum()


        def _lp_fee(row):
            dx, dy = row["amount0"], row["amount1"]
            act_liquidity = float(row["liquidity"])
            if dy > 0:
                return POOL_FEE_TIER * dy * self.L / act_liquidity
            if dx > 0:
                return POOL_FEE_TIER * dx * self.L / act_liquidity
            return 0.0

        swap_df["lp_fee"] = swap_df.apply(_lp_fee, axis=1)
        features[4] = swap_df["lp_fee"].sum()

        period_df["gas_eth"] = pd.to_numeric(period_df["gas_eth"], errors="coerce")
        features[5] = period_df["gas_eth"].mean() if not period_df.empty else 0.0
        
        features[6] = self.x_prev
        features[7] = self.y_prev
        
        features[8] = self.L
        features[9] = self.x_prev*self._eth_price(timestamp) + self.y_prev
        features[10] = ticks_to_sqrtp(self.tick_l)
        features[11] = ticks_to_sqrtp(self.tick_u)
        

        return features

    # wrapper to keep original naming used elsewhere
    def _features(self, ts: pd.Timestamp) -> np.ndarray:
        return self.form_observable_features(ts)

    @staticmethod
    def _calculate_initial_liquidity(Pc, Pl, Pu, total_value_to_invest, p_cex):
        """
        Calculates the liquidity L and the initial amounts of x and y to deposit.
        Pc, Pl, Pu are sqrtPriceX96 values.
        total_value_to_invest is in USDC.
        """
        sp = Pc / (2**96)
        sa = Pl / (2**96)
        sb = Pu / (2**96)

        if sa < sp < sb:
            # This formula calculates L based on the desired value to invest
            L = total_value_to_invest / ((sp - sa) + (1/sp - 1/sb) * p_cex)

            y_amount = L * (sp - sa)
            x_amount = L * (1/sp - 1/sb)
            return L, x_amount, y_amount
        elif sp <= sa: # All USDC
            return 0, 0, total_value_to_invest
        else: # All ETH
            return 0, total_value_to_invest / p_cex, 0

    @staticmethod
    def _calculate_assets_from_liquidity(L, Pc, Pl, Pu):
        """
        Calculates the current theoretical amounts of asset x and y for a given
        liquidity L and price range. Pc, Pl, Pu are sqrtPriceX96 values.
        """
        sp = Pc / (2**96)
        sa = Pl / (2**96)
        sb = Pu / (2**96)

        if sp <= sa: # Price is below the range, position is all in asset X (ETH)
            x_amount = L * (1/sa - 1/sb)
            y_amount = 0.0
        elif sa < sp < sb: # Price is within the range
            x_amount = L * (1/sp - 1/sb)
            y_amount = L * (sp - sa)
        else: # Price is above the range, position is all in asset Y (USDC)
            x_amount = 0.0
            y_amount = L * (sb - sa)

        return x_amount, y_amount

    # ------------------------ Gym API ------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        max_start = len(self.decision_grid) - self.EPISODE_LEN - 1
        self.idx = self.np_random.integers(0, max_start + 1)
        self.steps_left = self.EPISODE_LEN

        self.active = False
        self.current_wealth_in_USDC = self.config.WEALTH
        self.L = 0.0
        self.cumulative_pnl = 0.0

        ts = self.decision_grid[self.idx]
        return self._features(ts), {}

    def step(self, action):
        ts = self.decision_grid[self.idx]
        curr_tick = self._dex_tick(ts)
        p_curr = tick_to_price(curr_tick)
        
        # --- 1. Initialize variables for the step ---
        value_before = self.current_wealth_in_USDC

        new_total_value = value_before 
        reward = 0.0

        tick_l = int(action[0]*10)
        tick_u = int(action[1]*10)

        price_upper = tick_to_price(self.tick_u)
        price_lower = tick_to_price(self.tick_l)
        Pu = price_to_sqrtPriceX96(price_upper)
        Pl = price_to_sqrtPriceX96(price_lower)
        Pc = price_to_sqrtPriceX96(p_curr)

        # === CASE 1: ENTER a position if the price is out of range ===
        if Pc<Pl or Pc>Pu:
            gas_cost_for_step = self._gas_cost("Mint", ts) # New position cost
            gas_cost_for_step += self._gas_cost("Burn", ts) + self._gas_cost("Collect", ts) # Cost of closing old position

            
            self.tick_u = curr_tick - tick_u
            self.tick_l = curr_tick + tick_l


            price_upper = tick_to_price(self.tick_u)
            price_lower = tick_to_price(self.tick_l)
            Pu = price_to_sqrtPriceX96(price_upper)
            Pl = price_to_sqrtPriceX96(price_lower)
            

            # Calculate initial deposit based on total wealth
            # We invest the full available wealth minus the gas cost for this step
            investable_wealth = value_before - gas_cost_for_step
            L, x, y = self._calculate_initial_liquidity(Pc, Pl, Pu, investable_wealth, p_curr)

            self.L = L
            self.x_prev = x 
            self.y_prev = y
            self.active = True

            # Reset total fee trackers for the new position
            self.total_fees_x = 0.0
            self.total_fees_y = 0.0

            new_total_value = investable_wealth
            reward = new_total_value - value_before

        # === CASE 3: HOLD an active position if price is in range ===
        else:

            # Calculate PnL from fees for this step
            dx_fee, dy_fee = self._accrue_fees(ts)
            fee_pnl = p_curr * dx_fee + dy_fee
            
            # Calculate current base assets
            xt, yt = self._calculate_assets_from_liquidity(self.L, Pc, Pl, Pu)

            # Calculate PnL from impermanent loss (the hedged component)
            impermanent_loss_pnl = (xt - self.x_prev) * p_curr + (yt - self.y_prev)
            # if impermanent_loss_pnl + fee_pnl < 0:
            #     print("impermanent loss:", impermanent_loss_pnl)
            #     print("fee gain:", fee_pnl)
            #     print("reward:", fee_pnl + impermanent_loss_pnl)

            # The REWARD for the agent is the HEDGED PnL
            reward = fee_pnl + impermanent_loss_pnl

            # The REAL portfolio value is still the unhedged value. We update it here.
            self.total_fees_x += dx_fee
            self.total_fees_y += dy_fee
            total_x_in_pool = xt + self.total_fees_x
            total_y_in_pool = yt + self.total_fees_y
            new_total_value = p_curr * total_x_in_pool + total_y_in_pool
            
            # Update previous state for the next step's IL calculation
            self.x_prev, self.y_prev = xt, yt

        # --- 2. Update state ---
        self.current_wealth_in_USDC = new_total_value
        self.cumulative_pnl += reward
        
        self.idx += 1
        self.steps_left -= 1
        done = self.steps_left == 0 or self.idx >= len(self.decision_grid)
        obs = self._features(self.decision_grid[self.idx]) if not done else None

        return obs, reward, done, False, {}


    def render(self, mode="human"):
        ts = self.decision_grid[self.idx]
        print(f"{ts} | cumPnL = {self.cumulative_pnl:,.2f}")

    def close(self):
        pass
