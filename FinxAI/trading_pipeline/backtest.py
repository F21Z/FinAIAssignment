import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from finrl.agents.stablebaselines3.models import DRLAgent
from config import INDICATORS, TRAINED_MODEL_DIR, ACC_AMT_DATA_DIR, DATASETS_DIR, STATIC_DIR
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader


class BacktestAgents:

	def __init__(self, train_data = None, trade_data = None):

		self.train = train_data
		self.trade = trade_data

		self.if_using_a2c  = True
		self.if_using_ddpg = True
		self.if_using_ppo  = True
		self.if_using_td3  = True
		self.if_using_sac  = True

		self.trained_a2c   = A2C.load(TRAINED_MODEL_DIR + "/agent_a2c") if self.if_using_a2c else None
		self.trained_ddpg  = DDPG.load(TRAINED_MODEL_DIR + "/agent_ddpg") if self.if_using_ddpg else None
		self.trained_ppo   = PPO.load(TRAINED_MODEL_DIR + "/agent_ppo") if self.if_using_ppo else None
		self.trained_td3   = TD3.load(TRAINED_MODEL_DIR + "/agent_td3") if self.if_using_td3 else None
		self.trained_sac   = SAC.load(TRAINED_MODEL_DIR + "/agent_sac") if self.if_using_sac else None

		self.stock_dim     = len(self.trade.tic.unique())
		self.state_space   = 1 + 2 * self.stock_dim + len(INDICATORS) * self.stock_dim

		self.buy_cost_list    = [0.001] * self.stock_dim
		self.sell_cost_list   = [0.001] * self.stock_dim
		self.num_stock_shares = [0] * self.stock_dim

		self.env_kwargs       = {
									"hmax": 100,
									"initial_amount": 10000,
									"num_stock_shares": self.num_stock_shares,
									"buy_cost_pct": self.buy_cost_list,
									"sell_cost_pct": self.sell_cost_list,
									"state_space": self.state_space,
									"stock_dim": self.stock_dim,
									"tech_indicator_list": INDICATORS,
									"action_space": self.stock_dim,
									"reward_scaling": 1e-4
								}

		self.e_trade_gym = StockTradingEnv(df = self.trade, turbulence_threshold = 70, risk_indicator_col='vix', **self.env_kwargs)

	def getAccActionDataAndSaveCSV(self):

		# get A2C, DDPG, PPO, TD3 & SAC account/action values
		df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction( model=self.trained_a2c, 
																		environment = self.e_trade_gym) if self.if_using_a2c else (None, None)

		df_account_value_ddpg, df_actions_ddpg = DRLAgent.DRL_prediction( model=self.trained_ddpg, 
																		  environment = self.e_trade_gym) if self.if_using_ddpg else (None, None)

		df_account_value_ppo, df_actions_ppo = DRLAgent.DRL_prediction( model=self.trained_ppo, 
		                                       							environment = self.e_trade_gym) if self.if_using_ppo else (None, None)

		df_account_value_td3, df_actions_td3 = DRLAgent.DRL_prediction( model=self.trained_td3, 
		                                       							environment = self.e_trade_gym) if self.if_using_td3 else (None, None)

		df_account_value_sac, df_actions_sac = DRLAgent.DRL_prediction( model=self.trained_sac, 
		                                       							environment = self.e_trade_gym) if self.if_using_sac else (None, None)


		# Save the account values and corresponding action values to the CSV file
		df_account_value_a2c.to_csv(ACC_AMT_DATA_DIR+"/a2c/df_account_value_a2c.csv")
		df_actions_a2c.to_csv(ACC_AMT_DATA_DIR+"/a2c/df_actions_a2c.csv")

		df_account_value_ddpg.to_csv(ACC_AMT_DATA_DIR+"/ddpg/df_account_value_ddpg.csv")
		df_actions_ddpg.to_csv(ACC_AMT_DATA_DIR+"/ddpg/df_actions_ddpg.csv")

		df_account_value_ppo.to_csv(ACC_AMT_DATA_DIR+"/ppo/df_account_value_ppo.csv")
		df_actions_ppo.to_csv(ACC_AMT_DATA_DIR+"/ppo/df_actions_ppo.csv")

		df_account_value_td3.to_csv(ACC_AMT_DATA_DIR+"/td3/df_account_value_td3.csv")
		df_actions_td3.to_csv(ACC_AMT_DATA_DIR+"/td3/df_actions_td3.csv")

		df_account_value_sac.to_csv(ACC_AMT_DATA_DIR+"/sac/df_account_value_sac.csv")
		df_actions_sac.to_csv(ACC_AMT_DATA_DIR+"/sac/df_actions_sac.csv")

		df_result_a2c = (
		    df_account_value_a2c.set_index(df_account_value_a2c.columns[0])
		    if self.if_using_a2c
		    else None
		)
		df_result_ddpg = (
		    df_account_value_ddpg.set_index(df_account_value_ddpg.columns[0])
		    if self.if_using_ddpg
		    else None
		)
		df_result_ppo = (
		    df_account_value_ppo.set_index(df_account_value_ppo.columns[0])
		    if self.if_using_ppo
		    else None
		)
		df_result_td3 = (
		    df_account_value_td3.set_index(df_account_value_td3.columns[0])
		    if self.if_using_td3
		    else None
		)
		df_result_sac = (
		    df_account_value_sac.set_index(df_account_value_sac.columns[0])
		    if self.if_using_sac
		    else None
		)

		result = pd.DataFrame(
		    {
		        "a2c": df_result_a2c["account_value"] if self.if_using_a2c else None,
		        "ddpg": df_result_ddpg["account_value"] if self.if_using_ddpg else None,
		        "ppo": df_result_ppo["account_value"] if self.if_using_ppo else None,
		        "td3": df_result_td3["account_value"] if self.if_using_td3 else None,
		        "sac": df_result_sac["account_value"] if self.if_using_sac else None,
		    }
		)

		plt.rcParams["figure.figsize"] = (15,5)
		plt.figure()
		result.plot(grid='on')
		plt.title('Portfolio Value for DOW 30 Tickers\n')
		plt.ylabel('Net Portfolio Value (in Millions)')
		plt.savefig(STATIC_DIR+'/pf_val.png')



if __name__=="__main__":

	check_and_make_directories([ACC_AMT_DATA_DIR+'/a2c', 
								ACC_AMT_DATA_DIR+'/ddpg',
								ACC_AMT_DATA_DIR+'/ppo',
								ACC_AMT_DATA_DIR+'/td3',
								ACC_AMT_DATA_DIR+'/sac'])

	train_data = pd.read_csv(DATASETS_DIR+'/train_data.csv')
	trade_data = pd.read_csv(DATASETS_DIR+'/trade_data.csv')

	train_data = train_data.set_index(train_data.columns[0])
	train_data.index.names = ['']
	trade_data = trade_data.set_index(trade_data.columns[0])
	trade_data.index.names = ['']

	backtest_object = BacktestAgents(train_data, trade_data)
	backtest_object.getAccActionDataAndSaveCSV()
