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
POOL_FEE_TIER = 0.0005
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
        self.initial_wealth = self.config.WEALTH
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
            low=np.array([0.0, 1.0],  dtype=np.float32),
            high=np.array([1.0, 5.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.FEAT_NUM,), dtype=np.float32
        )

        self.idx = 0
        self.steps_left = 0

    def _load_data(self):

        cex_cols = ["open_time", "open"]
        cex = pd.read_csv(ETH_USDC_PRICE_PATH, usecols=cex_cols)
        cex["open_time"] = pd.to_datetime(cex["open_time"])        
        self.eth_px = cex.set_index("open_time")

        dex_tick = pd.read_csv(UNISWAP_SAMPLE_PATH, usecols=["timestamp", "tick"])
        dex_tick["timestamp"] = pd.to_datetime(dex_tick["timestamp"])
        self.lp_span = dex_tick.set_index("timestamp")

        # full raw dataframe (all event types) – we'll slice it later
        dex_raw = pd.read_csv(UNISWAP_SAMPLE_PATH, low_memory=False)
        dex_raw["timestamp"] = pd.to_datetime(dex_raw["timestamp"])
        self.uniswap_lp_data = dex_raw

        start = max(self.eth_px.index.min(), self.lp_span.index.min())
        end   = min(self.eth_px.index.max(), self.lp_span.index.max())

        self.eth_px        = self.eth_px.loc[start:end]
        self.lp_span       = self.lp_span.loc[start:end]
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
        self.decision_grid = pd.date_range(start=start, end=end, freq="1min")

    # ------------------------ price & gas helpers ------------------------
    def _eth_price(self, ts: pd.Timestamp) -> float:
        if ts in self.eth_px.index:
            return float(self.eth_px.loc[ts, "open"])
        pos = self.eth_px.index.searchsorted(ts, side="right") - 1
        if pos < 0:
            raise ValueError(f"ETH price not available before {ts}.")
        return float(self.eth_px.iloc[pos]["open"])

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
        t0 = ts.floor("1min")
        t1 = t0 + pd.Timedelta(minutes=1)
        minute_swaps = self.uniswap_lp_data.query(
            "(event_type == 'Swap') and @t0 <= timestamp < @t1"
        )

        in_range = minute_swaps[
            (minute_swaps["tick"] < self.tick_l) & (minute_swaps["tick"] > self.tick_u)
        ]
        if in_range.empty:
            return 0.0, 0.0

        # fees on the positive leg only
        fee0 = (in_range["amount0"].clip(lower=0) * POOL_FEE_TIER).sum()
        fee1 = (in_range["amount1"].clip(lower=0) * POOL_FEE_TIER).sum()

        L_pool = in_range["liquidity"].mean()
        share  = min(self.L / (L_pool + self.L), 1.0) if L_pool > 0 else 0.0
        return share * fee0, share * fee1


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

    # ------------------------ Gym API ------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        max_start = len(self.decision_grid) - self.EPISODE_LEN - 1
        self.idx = self.np_random.integers(0, max_start + 1)
        self.steps_left = self.EPISODE_LEN

        self.active = False
        self.L = 0.0
        self.cumulative_pnl = 0.0

        ts = self.decision_grid[self.idx]
        return self._features(ts), {}

    def step(self, action):
        ts = self.decision_grid[self.idx]
        engage = 1 if action[0] >= 0.5 else 0
        width  = int(round(np.clip(action[1], 0, 5)))

        p_cex      = self._eth_price(ts)          # USDC per ETH
        curr_tick  = self._dex_tick(ts)

        tick_l     = curr_tick + 10 * width
        tick_u     = curr_tick - 10 * width

        sqrt_pl    = ticks_to_sqrtp(tick_l)
        sqrt_pu    = ticks_to_sqrtp(tick_u)
        sqrt_pc    = ticks_to_sqrtp(curr_tick)

        dx_fee, dy_fee = self._accrue_fees(ts)    # token0 = ETH, token1 = USDC
        # print(dx_fee, dy_fee)
        self.x_prev += dx_fee
        self.y_prev += dy_fee
        # print(self.active, act, f"dx_fee = {dx_fee}, dy_fee = {dy_fee}", "FEE ACCRUED", p * dx_fee + dy_fee)
        # for this one, use CEX price as a more reliable proxy
        reward = (p_cex * max(dx_fee, 0) + max(dy_fee, 0))
        # print(f"reward = {reward:,.2f}")

        if not self.active and engage == 1:
            # TODO: revisit
            self.tick_l, self.tick_u = tick_l, tick_u

            denom = sqrt_pc * (sqrt_pu - sqrt_pc) + sqrt_pl * (sqrt_pc - sqrt_pl)
            self.L = max(0.0, self.initial_wealth / denom)

            x0 = self.L * (sqrt_pu - sqrt_pc) / (sqrt_pc * sqrt_pu)
            y0 = self.L * (sqrt_pc - sqrt_pl)
            self.x_prev, self.y_prev = x0, y0

            self.active = True
            reward -= self._gas_cost("Mint", ts)

        elif self.active and engage == 0:
            reward -= self._gas_cost("Burn", ts)
            reward -= self._gas_cost("Collect", ts)
            self.active = False
            self.L = 0.0

        elif self.active:
            sqrt_pl = ticks_to_sqrtp(self.tick_l)
            sqrt_pu = ticks_to_sqrtp(self.tick_u)
            sqrt_pc = ticks_to_sqrtp(curr_tick)

            if sqrt_pc <= sqrt_pl:
                xt, yt = self.L * (sqrt_pu - sqrt_pl) / (sqrt_pl * sqrt_pu), 0.0

            elif sqrt_pc < sqrt_pu:
                xt = self.L * (sqrt_pu - sqrt_pc) / (sqrt_pc * sqrt_pu)
                yt = self.L * (sqrt_pc - sqrt_pl)
            else:
                xt, yt = 0.0, self.L * (sqrt_pu - sqrt_pl)
            # print(f"xt = {xt}, yt = {yt}")
            # print(f"{xt - self.x_prev}", f"{yt - self.y_prev}")
            # print(f"{p_cex * (xt - self.x_prev)}", f"{yt - self.y_prev}")
            reward += p_cex * (xt - self.x_prev) + (yt - self.y_prev)
            self.x_prev, self.y_prev = xt, yt

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
