import glob
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from config.env_config import Config



"""
LVR - gas fees PNL
"""

###############################################
# Constants & Paths
###############################################
UNISWAP_POOL_EVENTS_PATH = "uniswap_lp_data"
ETH_USDC_PRICE_PATH = "data/binance/price_data/coinUSDC-price-data/ETHUSDC_20250316.csv"
UNISWAP_SAMPLE_PATH = "data/uniswap/uniswap_lp_data_1.csv"


def ticks_to_sqrtp(tick: int) -> float:
    """Convert a Uniswap-V3 tick index to √P."""
    return 1.0001 ** (tick / 2)


def _group_concat(frames):
    """Utility: concat only if list not empty."""
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


class UniswapV3LPGymEnv(gym.Env):

    metadata = {"render.modes": ["human"]}


    def __init__(self, config: Config | None = None, feat_num: int = 19):
        super().__init__()
        self.config = config or Config()
        self.initial_wealth = self.config.WEALTH
        self.FEAT_NUM = feat_num

        self.cumulative_pnl: float = 0.0

        self.active = False
        self.L = 0.0
        self.tick_l = 0
        self.tick_u = 0
        self.x_prev = 0.0
        self.y_prev = 0.0

        # I/O once
        self._load_data()
        self._build_decision_grid()

        self.action_space = spaces.Box(
            low=np.array([0, -887272, -887272], dtype=np.float32),
            high=np.array([1, 887272, 887272], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.FEAT_NUM,), dtype=np.float32
        )

        self.idx = 0 

    def _load_data(self):
        # ETH-USDC minute close prices
        cols = ["open_time", "open"]
        px = pd.read_csv(ETH_USDC_PRICE_PATH, usecols=cols)
        px["open_time"] = pd.to_datetime(px["open_time"])
        self.eth_px = px.set_index("open_time")

        # sample LP state (tick, sqrtPriceX96 etc.) - used only for date span
        lp = pd.read_csv(UNISWAP_SAMPLE_PATH, usecols=["timestamp", "tick"])
        lp["timestamp"] = pd.to_datetime(lp["timestamp"], unit="s")
        self.lp_span = lp.set_index("timestamp")

        # --- pre-compute gas-fee tables per event_type ---
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

    def _build_decision_grid(self):
        start = self.lp_span.index.min()
        end = self.lp_span.index.max()
        self.decision_grid = pd.date_range(start=start, end=end, freq="1min")


    def _eth_price(self, ts: pd.Timestamp) -> float:

        if ts in self.eth_px.index:
            return float(self.eth_px.loc[ts, "open"])
        # binary-search insertion point
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

    def _features(self, ts):
        prev = ts - pd.Timedelta(minutes=1)
        p_now = self._eth_price(ts)
        p_prev = self._eth_price(prev) if prev in self.eth_px.index else p_now
        out = np.zeros(self.FEAT_NUM, dtype=np.float32)
        out[0] = np.log(p_now / p_prev)
        out[-1] = 1.0
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
        reward = 0.0

        if not self.active and act == 1:  # Mint
            self.tick_l, self.tick_u = t_l, t_u
            sqrt_pl, sqrt_pu = ticks_to_sqrtp(t_l), ticks_to_sqrtp(t_u)
            p = self._eth_price(ts)
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

        elif self.active:  # Hold position → accrue Δ LVR
            sqrt_pl, sqrt_pu = ticks_to_sqrtp(self.tick_l), ticks_to_sqrtp(self.tick_u)
            p = self._eth_price(ts)
            sqrt_pc = np.sqrt(p)
            if sqrt_pc <= sqrt_pl:
                xt, yt = self.L * (sqrt_pu - sqrt_pl) / (sqrt_pl * sqrt_pu), 0.0
            elif sqrt_pc < sqrt_pu:
                xt = self.L * (sqrt_pu - sqrt_pc) / (sqrt_pc * sqrt_pu)
                yt = self.L * (sqrt_pc - sqrt_pl)
            else:
                xt, yt = 0.0, self.L * (sqrt_pu - sqrt_pl)
            reward += p * (xt - self.x_prev) + (yt - self.y_prev)
            self.x_prev, self.y_prev = xt, yt

        # flat→flat delivers 0 reward

        # ――― bookkeeping & next obs ―――
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
