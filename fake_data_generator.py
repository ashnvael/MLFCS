from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

from config.env_config import Config

def generate_fake_data(config: Config):
    base_dir = config.PATH_FAKE_DATA
    os.makedirs(base_dir, exist_ok=True)
    n = config.N_DAYS
    start_time = datetime.now() - timedelta(days=n)
    timestamps = [start_time + timedelta(days=i) for i in range(n)]
    
    eth_reserve = np.random.uniform(10, 100, size=n)
    usdc_reserve = np.random.uniform(20000, 200000, size=n)
    sqrt_price = np.sqrt(usdc_reserve / eth_reserve)
    tick = (np.log(sqrt_price) / np.log(np.sqrt(1.0001))).astype(int)
    total_liquidity = np.sqrt(eth_reserve * usdc_reserve)
    trading_volume = np.random.uniform(10000, 50000, size=n)
    
    dune_data = pd.DataFrame({
        "timestamp": timestamps,
        "eth_reserve": eth_reserve,
        "usdc_reserve": usdc_reserve,
        "sqrt_price": sqrt_price,
        "tick": tick,
        "total_liquidity": total_liquidity,
        "trading_volume": trading_volume
    })
    dune_data.to_csv(os.path.join(base_dir, "dune_data.csv"), index=False)
    
    eth_fees = np.random.uniform(0.01, 0.1, size=n)
    gas_fee = np.random.uniform(0.001, 0.005, size=n)
    fees_data = pd.DataFrame({
        "timestamp": timestamps,
        "eth_fees": eth_fees,
        "gas_fee": gas_fee
    })
    fees_data.to_csv(os.path.join(base_dir, "fees_data.csv"), index=False)
    
    current_sqrt_price = sqrt_price
    lower_sqrt_price = current_sqrt_price * 0.98
    upper_sqrt_price = current_sqrt_price * 1.02
    current_tick = tick
    lower_tick = (np.log(lower_sqrt_price) / np.log(np.sqrt(1.0001))).astype(int)
    upper_tick = (np.log(upper_sqrt_price) / np.log(np.sqrt(1.0001))).astype(int)
    delta_x = np.random.uniform(0.1, 5, size=n)
    delta_y = np.random.uniform(100, 1000, size=n)
    L_calc_1 = delta_x * current_sqrt_price * upper_sqrt_price / (upper_sqrt_price - current_sqrt_price)
    L_calc_2 = delta_y / (current_sqrt_price - lower_sqrt_price)
    liquidity_provided = np.minimum(L_calc_1, L_calc_2) * (2**96)
    computed_delta_x = liquidity_provided * (upper_sqrt_price - current_sqrt_price) / (upper_sqrt_price * current_sqrt_price)
    computed_delta_y = liquidity_provided * (current_sqrt_price - lower_sqrt_price)
    
    uniswap_v3_params = pd.DataFrame({
        "timestamp": timestamps,
        "current_sqrt_price": current_sqrt_price,
        "lower_sqrt_price": lower_sqrt_price,
        "upper_sqrt_price": upper_sqrt_price,
        "current_tick": current_tick,
        "lower_tick": lower_tick,
        "upper_tick": upper_tick,
        "delta_x": delta_x,
        "delta_y": delta_y,
        "liquidity_provided": liquidity_provided,
        "computed_delta_x": computed_delta_x,
        "computed_delta_y": computed_delta_y
    })
    uniswap_v3_params.to_csv(os.path.join(base_dir, "uniswap_v3_params.csv"), index=False)

if __name__ == "__main__":
    config = LiquidityExperimentConfig()
    generate_fake_data(config)