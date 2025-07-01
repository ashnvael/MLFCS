# Optimizing Liquidity Provision in Uniswap V3 Using Reinforcement Learning
Repository for course: Machine Learning for finance and complex systems at ETH Zurich. <br>

This repository contains the official implementation of our Deep Reinforcement Learning framework for optimizing liquidity provision in Uniswap v3. We propose a Proximal Policy Optimization (PPO) agent that operates in a continuous DeFi environment and learns to manage LP positions using price and volume data. Our method is benchmarked on real blockchain data, and we replicate theoretical results from Milionis et al., 2022. Automated market making and loss-versus rebalancing to provide intuition in a V2 setting.

## Features

- **Custom Gym Environments**: Simulate Uniswap V3 liquidity provision with realistic tick/range mechanics.
- **RL Training**: Train agents using Stable Baselines3 (PPO) to optimize liquidity provision strategies.
- **Data**: Includes historical price and data for Uniswap V3 0.05% fee ETH/USDC pool.
- **Analysis & Visualization**: Notebooks and scripts for analyzing agent performance and market data.

- **LVR Replication**: Reproduction of Loss-Versus-Rebalancing theoretical results.

## Project Structure

```
config/           # Environment and config files
data/             # Market and pool data (Binance, Uniswap)
models/           # Saved RL agent models
images/           # Images from our test runs (Also accessible in the notebooks)
logs/             # Training logs and monitor files
uniswap_lp_data/  # Additional Uniswap LP data that needs to be downloaded
*.ipynb           # Main notebooks for training, evaluation, and analysis
env_*.py          # Custom Gym environment definitions
```

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ashnvael/MLFCS.git
   cd MLFCS
   ```
2. (Recommended) Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Data Setup

**Important**: This project requires large datasets that are not included in the repository due to size constraints.

1. **Download Uniswap LP Data**:
   - Download the `uniswap_lp_data` folder from the provided [Polybox link](https://polybox.ethz.ch/index.php/s/DpSoBdsXtPR2Z4q)
   - Place it in the project root directory
   - The folder contains historical Uniswap V2 and V3 events and pool data needed to run the agents

2. **Optional: Collect Fresh Data (Data from 2024-2025 is included in the repository)**:
   - Set up Binance API keys in environment variables:
     ```bash
     export BINANCE_API_KEY="your_api_key"
     export BINANCE_SECRET_KEY="your_secret_key"
     ```
   - Run data collection scripts:
     ```bash
     # Price data collection
     python data/binance/price_data/binance-price-data.py
     
     # Hedging data collection (requires API keys)
     python data/binance/hedging_data/binance-hedging-data.py
     ```

### Usage

- **Training an RL Agent**:  
  Use the provided notebooks (e.g., `deep_rl_agent_semi_active_tick_range.ipynb`) to train and evaluate agents.  
  Adjust hyperparameters and environment settings as needed.

- **Data Preprocessing**:  
  Use `data_preprocessing.ipynb` to prepare and analyze different chunks of raw data (not necessary if downloading full uniswap_lp_data folder).

### LVR Implementation
The `LVR_replication.ipynb` implements the Loss-Versus-Rebalancing framework from Milionis et al., 2022, showing:
- Theoretical vs. empirical LVR calculations
- Hedging effectiveness at different frequencies
- Fee income vs. impermanent loss analysis

## Requirements

- Python 3.8+
- stable-baselines3
- gymnasium
- pandas, numpy, matplotlib, tqdm
- torch
- requests
- seaborn

(See `requirements.txt` for the full list.)

## Troubleshooting

### Common Issues

1. **Missing Data Files**:
   - Ensure `uniswap_lp_data` folder is downloaded and placed correctly
   - Check that all required CSV files are present

2. **API Rate Limits**:
   - Data collection scripts include rate limiting
   - Increase sleep intervals if hitting Binance API limits

3. **Memory Issues**:
   - Large datasets may require significant RAM
   - Consider using smaller data subsets for testing

## Authors
Kurlovich Nikolai, Rozman Anej, Hachimi Mehdi, Joly Julien - Team Eigenforce

## License

This project is licensed under the MIT License. 
