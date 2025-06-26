import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv("uniswap_lp_data/merged_uniswap_data.csv")

# Keep only the specified columns
swaps = df[[
    "timestamp", "amount0", "amount1", "sqrtPriceX96", "liquidity", "tick", "token0_balance", "token1_balance"
]].copy()

# Convert columns to appropriate types
swaps["timestamp"] = pd.to_datetime(swaps["timestamp"])
swaps["amount0"] = pd.to_numeric(swaps["amount0"], errors="coerce")
swaps["amount1"] = pd.to_numeric(swaps["amount1"], errors="coerce")
swaps["sqrtPriceX96"] = pd.to_numeric(swaps["sqrtPriceX96"], errors="coerce")
swaps["liquidity"] = pd.to_numeric(swaps["liquidity"], errors="coerce")
swaps["tick"] = pd.to_numeric(swaps["tick"], errors="coerce")
swaps["token0_balance"] = pd.to_numeric(swaps["token0_balance"], errors="coerce")
swaps["token1_balance"] = pd.to_numeric(swaps["token1_balance"], errors="coerce")

# Constants
FEE_TIER = 0.0005  # 0.05%
ETH_DECIMALS = 18
USDC_DECIMALS = 6
ETH_AMOUNT = 10  # Amount of ETH provided

# Helper functions
def tick_to_price(tick):
    return 1.0001 ** tick

def get_liquidity_from_eth(Pa, Pb, eth_amount):
    # L = y / (sqrt(Pb) - sqrt(Pa)), y = ETH
    sqrt_Pa = np.sqrt(Pa)
    sqrt_Pb = np.sqrt(Pb)
    if sqrt_Pb <= sqrt_Pa:
        return 0.0
    return eth_amount / (sqrt_Pb - sqrt_Pa)

def get_usdc_amount(L, Pa, Pb, Pc):
    # x = L * (sqrt(Pb) - sqrt(Pc)) / (sqrt(Pc) * sqrt(Pb)), x = USDC
    sqrt_Pb = np.sqrt(Pb)
    sqrt_Pc = np.sqrt(Pc)
    if sqrt_Pc == 0 or sqrt_Pb == 0:
        return 0.0
    return L * (sqrt_Pb - sqrt_Pc) / (sqrt_Pc * sqrt_Pb)

daily_fees = []

for date, day_df in swaps.groupby(swaps["timestamp"].dt.date):
    min_tick = day_df["tick"].min()
    max_tick = day_df["tick"].max()
    Pa = tick_to_price(min_tick)
    Pb = tick_to_price(max_tick)
    Pc = tick_to_price(day_df["tick"].iloc[0])  # Use first tick of the day as current price

    # Calculate liquidity provided in ETH
    L = get_liquidity_from_eth(Pa, Pb, ETH_AMOUNT)
    # Calculate corresponding USDC amount
    usdc_amount = get_usdc_amount(L, Pa, Pb, Pc)

    fee_eth = 0.0
    fee_usdc = 0.0
    for _, row in day_df.iterrows():
        tick = row["tick"]
        price = tick_to_price(tick)
        sqrt_Pa = np.sqrt(Pa)
        sqrt_Pb = np.sqrt(Pb)
        sqrt_P = np.sqrt(price)
        # Only earn fees if tick is within range
        if sqrt_Pa < sqrt_P < sqrt_Pb:
            pool_liquidity = row["liquidity"]
            if pool_liquidity > 0:
                share = L / pool_liquidity
                # Earn fees from positive amounts
                amt0 = max(row["amount0"], 0)
                amt1 = max(row["amount1"], 0)
                fee_eth += share * amt0 * FEE_TIER
                fee_usdc += share * amt1 * FEE_TIER
    daily_fees.append({
        "date": date,
        "fee_eth": fee_eth,
        "fee_usdc": fee_usdc,
        "min_tick": min_tick,
        "max_tick": max_tick,
        "L": L,
        "usdc_amount": usdc_amount
    })

# Save daily fees to a DataFrame
fees_df = pd.DataFrame(daily_fees)
fees_df.to_csv("daily_fees.csv", index=False)