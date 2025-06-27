import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Define the path to the CSV file
csv_path = os.path.join(os.path.dirname(__file__), 'uniswap_lp_data_1.csv')

# Read the CSV file into a pandas DataFrame
all_events = pd.read_csv(csv_path)

# Convert timestamp to datetime format
all_events['timestamp'] = pd.to_datetime(all_events['timestamp'], unit='s')

mint_events = all_events[all_events['event_type'] == 'Mint'].copy()
burn_events = all_events[all_events['event_type'] == 'Burn'].copy()
swap_events = all_events[all_events['event_type'] == 'Swap'].copy()
collect_events = all_events[all_events['event_type'] == 'Collect'].copy()


