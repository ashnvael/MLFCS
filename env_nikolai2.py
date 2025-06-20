import glob
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from config.env_config import Config

"""
LVR - gas fees + collected fees
"""


###############################################
# Constants & Paths
###############################################
UNISWAP_POOL_EVENTS_PATH = "uniswap_lp_data"  # folder with all on-chain CSVs
ETH_USDC_PRICE_PATH = "data/binance/price_data/coinUSDC-price-data/ETHUSDC_20250316.csv"
UNISWAP_SAMPLE_PATH = "data/uniswap/uniswap_lp_data_1.csv"
FEE_TABLE_PATH = "data/uniswap/fee_table.parquet"   # pre-computed minute fee grid
POOL_FEE_TIER = 0.0005   # 5 bp 

def ticks_to_sqrtp(tick: int) -> float:
    """Convert a Uniswap-V3 tick index to √P."""
    return 1.0001 ** (tick / 2)


def _group_concat(frames):
    """Utility: concat only if list not empty."""
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _build_fee_table(swap_paths: list[str]) -> pd.DataFrame:
    """Parse raw Swap CSVs and return a **minute-level** fee grid.

    Columns returned: fee0, fee1, liquidity_pool, tick_close
    Index: pandas.DatetimeIndex rounded to minute.
    """
    dfs = []
    for p in swap_paths:
        df = pd.read_csv(
            p,
            usecols=["timestamp", "amount0", "amount1", "liquidity", "tick"],
            low_memory=False,          # avoids dtype guessing chunks
        )

        # 👉 force to float; bad rows become NaN, which we then drop
        for col in ("amount0", "amount1", "liquidity"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(subset=["amount0", "amount1", "liquidity"], inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        # swap amounts can be +/-; fee is on *notional* traded
        df["fee0"] = df["amount0"].abs() * POOL_FEE_TIER
        df["fee1"] = df["amount1"].abs() * POOL_FEE_TIER
        dfs.append(df[["timestamp", "fee0", "fee1", "liquidity", "tick"]])

    raw = pd.concat(dfs, ignore_index=True)

    # resample to 1-minute grid (to match env.decision_grid)
    fee_grid = (raw
                .set_index("timestamp")
                .resample("1min")
                .agg({
                    "fee0": "sum",
                    "fee1": "sum",
                    "liquidity": "mean",
                    "tick": "last",
                }))
    fee_grid.rename(columns={"liquidity": "liquidity_pool", "tick": "tick_close"}, inplace=True)
    fee_grid.fillna({"fee0": 0.0, "fee1": 0.0, "liquidity_pool": np.nan}, inplace=True)
    return fee_grid

class UniswapV3LPGymEnv(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self, config: Config | None = None, feat_num: int = 19):
        super().__init__()
        self.config = config or Config()
        self.initial_wealth = self.config.WEALTH
        self.FEAT_NUM = feat_num
        self.EPISODE_LEN  = 1000

        # state vars
        self.cumulative_pnl: float = 0.0
        self.active = False
        self.L = 0.0           # liquidity of our position
        self.tick_l = 0        # lower tick bound
        self.tick_u = 0        # upper tick bound
        self.x_prev = 0.0      # token0 inventory *after* last step
        self.y_prev = 0.0      # token1 inventory *after* last step

        # one-time loads
        self._load_data()
        self._build_decision_grid()

        # RL spaces
        self.action_space = spaces.Box(
            low=np.array([0, -887272, -887272], dtype=np.float32),
            high=np.array([1, 887272, 887272], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.FEAT_NUM,), dtype=np.float32
        )

        self.idx = 0  # pointer on decision_grid
        self.steps_left = 0

    def _load_data(self):
        """Load prices, gas stats, *and* the minute-level fee grid."""
        # 1) ETH-USDC prices (minute open)
        cols = ["open_time", "open"]
        px = pd.read_csv(ETH_USDC_PRICE_PATH, usecols=cols)
        px["open_time"] = pd.to_datetime(px["open_time"])
        self.eth_px = px.set_index("open_time")

        # 2) sample LP events – for date span only
        lp = pd.read_csv(UNISWAP_SAMPLE_PATH, usecols=["timestamp", "tick"])
        lp["timestamp"] = pd.to_datetime(lp["timestamp"], unit="s")
        self.lp_span = lp.set_index("timestamp")

        fee_frames: dict[str, list[pd.DataFrame]] = {}
        for path in glob.glob(f"{UNISWAP_POOL_EVENTS_PATH}/uniswap_lp_data_*.csv"):
            df = pd.read_csv(path, usecols=["timestamp", "event_type", "gas_eth"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            for evt, grp in df.groupby("event_type"):
                fee_frames.setdefault(evt, []).append(grp)
        self.gas_fee = {}
        for evt, lst in fee_frames.items():
            g = _group_concat(lst).sort_values("timestamp")
            self.gas_fee[evt] = g.set_index("timestamp")

        # 4) minute-level pool fee grid
        fee_tbl_path = Path(FEE_TABLE_PATH)
        if fee_tbl_path.exists():
            self.fee_grid = pd.read_parquet(fee_tbl_path)
        else:
            swap_paths = glob.glob(f"{UNISWAP_POOL_EVENTS_PATH}/uniswap_lp_data_*.csv")
            self.fee_grid = _build_fee_table(swap_paths)
            fee_tbl_path.parent.mkdir(parents=True, exist_ok=True)
            self.fee_grid.to_parquet(fee_tbl_path, compression="zstd")

    def _build_decision_grid(self):
        start = self.lp_span.index.min()
        end = self.lp_span.index.max()
        self.decision_grid = pd.date_range(start=start, end=end, freq="1min")

    def _eth_price(self, ts: pd.Timestamp) -> float:
        if ts in self.eth_px.index:
            return float(self.eth_px.loc[ts, "open"])
        pos = self.eth_px.index.searchsorted(ts, side="right") - 1
        if pos < 0:
            raise ValueError(f"ETH price not available before {ts}.")
        return float(self.eth_px.iloc[pos]["open"])

    def _gas_cost(self, evt: str, ts) -> float:
        df = self.gas_fee.get(evt)
        if df is None or ts not in self.eth_px.index:
            return 0.0
        last20 = df.loc[:ts].tail(20)
        if last20.empty:
            return 0.0
        return float(last20["gas_eth"].mean() * self._eth_price(ts))

    # ---------- fee helpers ----------
    def _pool_fees(self, ts: pd.Timestamp):
        """Return (fee0, fee1, L_pool) for this minute."""
        if ts not in self.fee_grid.index:
            return 0.0, 0.0, np.nan
        row = self.fee_grid.loc[ts]
        return float(row.fee0), float(row.fee1), float(row.liquidity_pool)

    def _accrue_fees(self, ts: pd.Timestamp):
        """Return fee amounts accruing to **our position** for this minute."""
        if not self.active:
            return 0.0, 0.0
        fee0_pool, fee1_pool, L_pool = self._pool_fees(ts)
        if np.isnan(L_pool) or L_pool == 0.0:
            return 0.0, 0.0
        share = self.L / L_pool
        return share * fee0_pool, share * fee1_pool

    def _features(self, ts):
        prev = ts - pd.Timedelta(minutes=1)
        p_now = self._eth_price(ts)
        p_prev = self._eth_price(prev) if prev in self.eth_px.index else p_now
        out = np.zeros(self.FEAT_NUM, dtype=np.float32)
        out[0] = np.log(p_now / p_prev)
        out[-1] = 1.0  # bias term
        return out

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
        act, t_l, t_u = map(int, action)

        # ---------- 1) Fee accrual ----------
        p = self._eth_price(ts)
        dx_fee, dy_fee = self._accrue_fees(ts)
        self.x_prev += dx_fee
        self.y_prev += dy_fee
        reward = p * dx_fee + dy_fee   # USD value of fees for this minute

        # ---------- 2) Action logic ----------
        if not self.active and act == 1:  # Mint
            self.tick_l, self.tick_u = t_l, t_u
            sqrt_pl, sqrt_pu = ticks_to_sqrtp(t_l), ticks_to_sqrtp(t_u)
            sqrt_pc = np.sqrt(p)
            denom = sqrt_pc * (sqrt_pu - sqrt_pc) + sqrt_pl * (sqrt_pc - sqrt_pl)
            self.L = self.initial_wealth / denom
            x0 = self.L * (sqrt_pu - sqrt_pc) / (sqrt_pc * sqrt_pu)
            y0 = self.L * (sqrt_pc - sqrt_pl)
            self.x_prev, self.y_prev = x0, y0
            self.active = True
            reward -= self._gas_cost("Mint", ts)

        elif self.active and act == 0:  # Burn
            reward -= self._gas_cost("Burn", ts) + self._gas_cost("Collect", ts)
            self.active = False
            self.L = 0.0

        elif self.active:  # Hold position → Δ LVR
            sqrt_pl, sqrt_pu = ticks_to_sqrtp(self.tick_l), ticks_to_sqrtp(self.tick_u)
            sqrt_pc = np.sqrt(p)
            if sqrt_pc <= sqrt_pl:
                xt, yt = self.L * (sqrt_pu - sqrt_pl) / (sqrt_pl * sqrt_pu), 0.0
            elif sqrt_pc < sqrt_pu:
                xt = self.L * (sqrt_pu - sqrt_pc) / (sqrt_pc * sqrt_pu)
                yt = self.L * (sqrt_pc - sqrt_pl)
            else:
                xt, yt = 0.0, self.L * (sqrt_pu - sqrt_pl)
            reward += p * (xt - self.x_prev) + (yt - self.y_prev)  # Δ LVR
            self.x_prev, self.y_prev = xt, yt

        # ---------- 3) Bookkeeping ----------
        self.cumulative_pnl += reward
        self.idx += 1
        self.steps_left -= 1

        done = (self.steps_left == 0) or (self.idx >= len(self.decision_grid))
        obs = self._features(self.decision_grid[self.idx]) if not done else None
        
        return obs, reward, done, False, {}

    def render(self, mode="human"):
        ts = self.decision_grid[self.idx]
        print(f"{ts} | cumPnL = {self.cumulative_pnl:,.2f}")

    def close(self):
        pass

import glob
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from config.env_config import Config

"""
LVR - gas fees + collected fees
"""


###############################################
# Constants & Paths
###############################################
UNISWAP_POOL_EVENTS_PATH = "uniswap_lp_data"  # folder with all on-chain CSVs
ETH_USDC_PRICE_PATH = "data/binance/price_data/coinUSDC-price-data/ETHUSDC_20250316.csv"
UNISWAP_SAMPLE_PATH = "data/uniswap/uniswap_lp_data_1.csv"
FEE_TABLE_PATH = "data/uniswap/fee_table.parquet"   # pre-computed minute fee grid
POOL_FEE_TIER = 0.0005   # 5 bp 


def ticks_to_sqrtp(tick: int) -> float:
    """Convert a Uniswap-V3 tick index to √P."""
    return 1.0001 ** (tick / 2)


def _group_concat(frames):
    """Utility: concat only if list not empty."""
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _build_fee_table(swap_paths: list[str]) -> pd.DataFrame:
    """Parse raw Swap CSVs and return a **minute-level** fee grid.

    Columns returned: fee0, fee1, liquidity_pool, tick_close
    Index: pandas.DatetimeIndex rounded to minute.
    """
    dfs = []
    for p in swap_paths:
        df = pd.read_csv(
            p,
            usecols=["timestamp", "amount0", "amount1", "liquidity", "tick"],
            low_memory=False,          # avoids dtype guessing chunks
        )

        # 👉 force to float; bad rows become NaN, which we then drop
        for col in ("amount0", "amount1", "liquidity"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(subset=["amount0", "amount1", "liquidity"], inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        # swap amounts can be +/-; fee is on *notional* traded
        df["fee0"] = df["amount0"].abs() * POOL_FEE_TIER
        df["fee1"] = df["amount1"].abs() * POOL_FEE_TIER
        dfs.append(df[["timestamp", "fee0", "fee1", "liquidity", "tick"]])

    raw = pd.concat(dfs, ignore_index=True)

    # resample to 1-minute grid (to match env.decision_grid)
    fee_grid = (raw
                .set_index("timestamp")
                .resample("1min")
                .agg({
                    "fee0": "sum",
                    "fee1": "sum",
                    "liquidity": "mean",
                    "tick": "last",
                }))
    fee_grid.rename(columns={"liquidity": "liquidity_pool", "tick": "tick_close"}, inplace=True)
    fee_grid.fillna({"fee0": 0.0, "fee1": 0.0, "liquidity_pool": np.nan}, inplace=True)
    return fee_grid

class UniswapV3LPGymEnv(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self, config: Config | None = None, feat_num: int = 19):
        super().__init__()
        self.config = config or Config()
        self.initial_wealth = self.config.WEALTH
        self.FEAT_NUM = feat_num

        # state vars
        self.cumulative_pnl: float = 0.0
        self.active = False
        self.L = 0.0           # liquidity of our position
        self.tick_l = 0        # lower tick bound
        self.tick_u = 0        # upper tick bound
        self.x_prev = 0.0      # token0 inventory *after* last step
        self.y_prev = 0.0      # token1 inventory *after* last step

        # one-time loads
        self._load_data()
        self._build_decision_grid()

        # RL spaces
        self.action_space = spaces.Box(
            low=np.array([0, -887272, -887272], dtype=np.float32),
            high=np.array([1, 887272, 887272], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.FEAT_NUM,), dtype=np.float32
        )

        self.idx = 0  # pointer on decision_grid

    def _load_data(self):
        """Load prices, gas stats, *and* the minute-level fee grid."""
        # 1) ETH-USDC prices (minute open)
        cols = ["open_time", "open"]
        px = pd.read_csv(ETH_USDC_PRICE_PATH, usecols=cols)
        px["open_time"] = pd.to_datetime(px["open_time"])
        self.eth_px = px.set_index("open_time")

        # 2) sample LP events – for date span only
        lp = pd.read_csv(UNISWAP_SAMPLE_PATH, usecols=["timestamp", "tick"])
        lp["timestamp"] = pd.to_datetime(lp["timestamp"], unit="s")
        self.lp_span = lp.set_index("timestamp")

        fee_frames: dict[str, list[pd.DataFrame]] = {}
        for path in glob.glob(f"{UNISWAP_POOL_EVENTS_PATH}/uniswap_lp_data_*.csv"):
            df = pd.read_csv(path, usecols=["timestamp", "event_type", "gas_eth"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            for evt, grp in df.groupby("event_type"):
                fee_frames.setdefault(evt, []).append(grp)
        self.gas_fee = {}
        for evt, lst in fee_frames.items():
            g = _group_concat(lst).sort_values("timestamp")
            self.gas_fee[evt] = g.set_index("timestamp")

        # 4) minute-level pool fee grid
        fee_tbl_path = Path(FEE_TABLE_PATH)
        if fee_tbl_path.exists():
            self.fee_grid = pd.read_parquet(fee_tbl_path)
        else:
            swap_paths = glob.glob(f"{UNISWAP_POOL_EVENTS_PATH}/uniswap_lp_data_*.csv")
            self.fee_grid = _build_fee_table(swap_paths)
            fee_tbl_path.parent.mkdir(parents=True, exist_ok=True)
            self.fee_grid.to_parquet(fee_tbl_path, compression="zstd")

    def _build_decision_grid(self):
        start = self.lp_span.index.min()
        end = self.lp_span.index.max()
        self.decision_grid = pd.date_range(start=start, end=end, freq="1min")

    def _eth_price(self, ts: pd.Timestamp) -> float:
        if ts in self.eth_px.index:
            return float(self.eth_px.loc[ts, "open"])
        pos = self.eth_px.index.searchsorted(ts, side="right") - 1
        if pos < 0:
            raise ValueError(f"ETH price not available before {ts}.")
        return float(self.eth_px.iloc[pos]["open"])

    def _gas_cost(self, evt: str, ts) -> float:
        df = self.gas_fee.get(evt)
        if df is None or ts not in self.eth_px.index:
            return 0.0
        last20 = df.loc[:ts].tail(20)
        if last20.empty:
            return 0.0
        return float(last20["gas_eth"].mean() * self._eth_price(ts))

    # ---------- fee helpers ----------
    def _pool_fees(self, ts: pd.Timestamp):
        """Return (fee0, fee1, L_pool) for this minute."""
        if ts not in self.fee_grid.index:
            return 0.0, 0.0, np.nan
        row = self.fee_grid.loc[ts]
        return float(row.fee0), float(row.fee1), float(row.liquidity_pool)

    def _accrue_fees(self, ts: pd.Timestamp):
        """Return fee amounts accruing to **our position** for this minute."""
        if not self.active:
            return 0.0, 0.0
        fee0_pool, fee1_pool, L_pool = self._pool_fees(ts)
        if np.isnan(L_pool) or L_pool == 0.0:
            return 0.0, 0.0
        share = self.L / L_pool
        return share * fee0_pool, share * fee1_pool

    def _features(self, ts):
        prev = ts - pd.Timedelta(minutes=1)
        p_now = self._eth_price(ts)
        p_prev = self._eth_price(prev) if prev in self.eth_px.index else p_now
        out = np.zeros(self.FEAT_NUM, dtype=np.float32)
        out[0] = np.log(p_now / p_prev)
        out[-1] = 1.0  # bias term
        return out

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        self.active = False
        self.L = 0.0
        self.cumulative_pnl = 0.0
        ts = self.decision_grid[self.idx]
        return self._features(ts), {}

    def step(self, action):
        ts = self.decision_grid[self.idx]
        act, t_l, t_u = map(int, action)

        # ---------- 1) Fee accrual ----------
        p = self._eth_price(ts)
        dx_fee, dy_fee = self._accrue_fees(ts)
        self.x_prev += dx_fee
        self.y_prev += dy_fee
        reward = p * dx_fee + dy_fee   # USD value of fees for this minute

        # ---------- 2) Action logic ----------
        if not self.active and act == 1:  # Mint
            self.tick_l, self.tick_u = t_l, t_u
            sqrt_pl, sqrt_pu = ticks_to_sqrtp(t_l), ticks_to_sqrtp(t_u)
            sqrt_pc = np.sqrt(p)
            denom = sqrt_pc * (sqrt_pu - sqrt_pc) + sqrt_pl * (sqrt_pc - sqrt_pl)
            self.L = self.initial_wealth / denom
            x0 = self.L * (sqrt_pu - sqrt_pc) / (sqrt_pc * sqrt_pu)
            y0 = self.L * (sqrt_pc - sqrt_pl)
            self.x_prev, self.y_prev = x0, y0
            self.active = True
            reward -= self._gas_cost("Mint", ts)

        elif self.active and act == 0:  # Burn
            reward -= self._gas_cost("Burn", ts) + self._gas_cost("Collect", ts)
            self.active = False
            self.L = 0.0

        elif self.active:  # Hold position → Δ LVR
            sqrt_pl, sqrt_pu = ticks_to_sqrtp(self.tick_l), ticks_to_sqrtp(self.tick_u)
            sqrt_pc = np.sqrt(p)
            if sqrt_pc <= sqrt_pl:
                xt, yt = self.L * (sqrt_pu - sqrt_pl) / (sqrt_pl * sqrt_pu), 0.0
            elif sqrt_pc < sqrt_pu:
                xt = self.L * (sqrt_pu - sqrt_pc) / (sqrt_pc * sqrt_pu)
                yt = self.L * (sqrt_pc - sqrt_pl)
            else:
                xt, yt = 0.0, self.L * (sqrt_pu - sqrt_pl)
            reward += p * (xt - self.x_prev) + (yt - self.y_prev)  # Δ LVR
            self.x_prev, self.y_prev = xt, yt

        # ---------- 3) Bookkeeping ----------
        self.cumulative_pnl += reward
        self.idx += 1
        done = self.idx >= len(self.decision_grid)
        obs = self._features(self.decision_grid[self.idx]) if not done else None
        return obs, reward, done, False, {}

    def render(self, mode="human"):
        ts = self.decision_grid[self.idx]
        print(f"{ts} | cumPnL = {self.cumulative_pnl:,.2f}")

    def close(self):
        pass
