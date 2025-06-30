# Optimizing Liquidity Provision in Uniswap V3 Using Reinforcement Learning
Repository for course: Machine Learning for finance and complex systems at ETH Zurich. <br>
**Authors**: KURLOVICH Nikolaï, ROZMAN Anej, HACHIMI Mehdi, JOLY Julien

This repository contains the official implementation of our Deep Reinforcement Learning framework for optimal liquidity provision in Uniswap v3. We propose a Proximal Policy Optimization (PPO) agent that operates in a continuous DeFi environment and learns to manage LP positions using price and volume data. Our method is benchmarked on real blockchain data, and we replicate theoretical results from Milionis et al., 2022. Automated market making and loss-versus rebalancing to provide intuition in a V2 setting.

## Features

- **Custom Gym Environments**: Simulate Uniswap V3 liquidity provision with realistic tick/range mechanics.
- **RL Training**: Train agents using Stable Baselines3 (PPO) to optimize liquidity provision strategies.
- **Data**: Includes historical price and pool data for ETH/USDC and other pairs.
- **Analysis & Visualization**: Notebooks and scripts for analyzing agent performance and market data.

## Project Structure

```
config/           # Environment and config files
data/             # Market and pool data (Binance, Uniswap)
models/           # Saved RL agent models
logs/             # Training logs and monitor files
uniswap_lp_data/  # Additional Uniswap LP data
*.ipynb           # Main notebooks for training, evaluation, and analysis
env_*.py          # Custom Gym environment definitions
```

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd MLFCS
   ```
2. (Recommended) Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install the `uniswap_lp_data` folder from polybox here: link

### Usage

- **Training an RL Agent**:  
  Use the provided notebooks (e.g., `RLagent_semi_active_tick_range.ipynb`) to train and evaluate agents.  
  Adjust hyperparameters and environment settings as needed.

- **Data Preprocessing**:  
  Use `data_preprocessing.ipynb` to prepare and analyze raw data.

- **Analysis**:  
  Visualize agent behavior and compare CEX/DEX prices using the plotting sections in the notebooks.

## Requirements

- Python 3.8+
- stable-baselines3
- gymnasium
- pandas, numpy, matplotlib, tqdm
- torch

(See `requirements.txt` for the full list.)

## Reproducing the results
To run the replication of the paper Milionis et al., 2022. Automated market making and loss-versus rebalancing, the Jupyter Notebook replication_lvr.ipynb needs to be run. The data needed can be found in "_replication_lvr_" data file in Polybox. It should contain the following csv files:
* swap0.csv
* swap1.csv
* swap2.csv
* swap3.csv
* swap4.csv
* tradesv2.csv
* binance_ETH_data.csv
* univ2_ETH_USDC_data.csv

## License

This project is licensed under the MIT License. 
