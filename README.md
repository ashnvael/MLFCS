# Machine-learning-finance-and-complex-systems

This repository contains research code and experiments for training and evaluating reinforcement learning (RL) agents to provide liquidity on Uniswap V3. The project includes custom OpenAI Gym environments, data preprocessing scripts, and RL training/evaluation notebooks.

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

## Citation

If you use this codebase in your research, please cite appropriately.

## License

This project is licensed under the MIT License. 
