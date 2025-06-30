# Optimizing Liquidity Provision in Uniswap V3 Using Reinforcement Learning
Repository for course: Machine Learning for finance and complex systems at ETH Zurich. 
Authors: KURLOVICH Nikolaï, ROZMAN Anej, HACHIMI Mehdi, JOLY Julien

This repository contains the official implementation of our Deep Reinforcement Learning framework for optimal liquidity provision in Uniswap v3. We propose a Proximal Policy Optimization (PPO) agent that operates in a continuous DeFi environment and learns to manage LP positions using price and volume data. Our method is benchmarked on real blockchain data, and we replicate theoretical results from _Milionis et al., 2022. Automated market making and loss-versus rebalancing_ to provide intuition in a V2 setting.

## Data

## 📁 Project Structure

To run the replication of the paper _Milionis et al., 2022. Automated market making and loss-versus rebalancing_, the Jupyter Notebook _replication_lvr.ipynb_ needs to be run. The data needed can be found in ??? file. It should contain the following csv files:
--> swap0.csv  --> swap1csv  --> swap2.cs --> swap3.csv --> swap4.csv --> tradesv2.csv -->binance_ETH_data.csv --> univ2_ETH_USDC_data.csv

For the Reinforcement Learning experiments, we implemented the following structure:

Needs to be changed.

├── data/                 # Raw and processed data
├── models/               # Neural network architectures
├── environments/         # Custom gym environments              
├── scripts/              # Training and evaluation scripts
├── results/              # Output figures and logs
├── utils/                # Helper functions
├── main.py               # Entry point for training
├── requirements.txt
└── README.md

## Installation
List dependencies and installation instructions.




