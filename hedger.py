from __future__ import annotations
import numpy as np
import pandas as pd


class DeltaHedger:

    def __init__(
        self,
        price_series: pd.Series,
        rebalance_freq: str | pd.Timedelta | None = "1min",
        initial_cash: float = 0.0,
    ) -> None:
        self.price_series = price_series.sort_index()
        self.initial_cash = initial_cash

        if rebalance_freq is None:
            self.rebalance_freq = None
        else:
            self.rebalance_freq = (
                pd.Timedelta(rebalance_freq)
                if not isinstance(rebalance_freq, pd.Timedelta)
                else rebalance_freq
            )
            if self.rebalance_freq <= pd.Timedelta(0):
                raise ValueError("rebalance_freq must be positive.")
        self.reset()

    def reset(self) -> None:
        self.position_eth: float = 0.0
        self.cash_usdc:   float = self.initial_cash
        self._last_price  = np.nan
        self._last_rebalance_ts: pd.Timestamp | None = None

    def update(self, ts: pd.Timestamp, target_eth_delta: float) -> float:

        p_now = self._price(ts)

        mtm_pnl = 0.0
        if not np.isnan(self._last_price):
            mtm_pnl = self.position_eth * (p_now - self._last_price)

        should_rebalance = (
            (self.rebalance_freq is None)                           
            or (self._last_rebalance_ts is None)                  
            or (ts - self._last_rebalance_ts >= self.rebalance_freq)  
        )

        if should_rebalance:
            trade_size = target_eth_delta - self.position_eth
            self.cash_usdc -= trade_size * p_now
            self.position_eth = target_eth_delta
            self._last_rebalance_ts = ts

        self._last_price = p_now
        return mtm_pnl

    def _price(self, ts: pd.Timestamp) -> float:
        """Exact match else use previous tick."""
        if ts in self.price_series.index:
            return float(self.price_series.loc[ts])

        pos = self.price_series.index.searchsorted(ts, side="right") - 1
        if pos < 0:
            raise ValueError(f"Binance price not available before {ts}.")
        return float(self.price_series.iloc[pos])
