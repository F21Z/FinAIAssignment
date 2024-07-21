import pandas as pd
import sys
from stable_baselines3.common.logger import configure

from finrl.agents.stablebaselines3.models import DRLAgent
from config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR, TENSORBOARD_LOG_DIR, DATASETS_DIR
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

class DRLTrainClass:

	def __init__(self, data = None, init_amt = 0):

		if data is None or init_amt == 0:
			print(">> Enter Valid Data!! or 0 amt passed!!")
			sys.exit(0)

		self.data             = data
		self.initial_amount   = init_amt
		self.stock_dim        = len(self.data.tic.unique())
		self.state_space      = 1 + 2*self.stock_dim + len(INDICATORS)*self.stock_dim
		self.buy_cost_list    = [0.01]*self.stock_dim
		self.sell_cost_list   = [0.01]*self.stock_dim
		self.num_stock_shares = [0]*self.stock_dim
		self.env_kwargs       = {
								    "hmax": 100,
								    "initial_amount": self.initial_amount,
								    "num_stock_shares": self.num_stock_shares,
								    "buy_cost_pct": self.buy_cost_list,
								    "sell_cost_pct": self.sell_cost_list,
								    "state_space": self.state_space,
								    "stock_dim": self.stock_dim,
								    "tech_indicator_list": INDICATORS,
								    "action_space": self.stock_dim,
								    "reward_scaling": 1e-4
								}

		self.e_train_gym      = StockTradingEnv(df = self.data, **self.env_kwargs)
		self.env_train, _     = self.e_train_gym.get_sb_env()
		self.agent            = DRLAgent(env = self.env_train)
		
		self.if_using_a2c     = True
		self.if_using_ddpg    = True
		self.if_using_ppo     = True
		self.if_using_td3     = True
		self.if_using_sac     = True

	def printClassParameters(self):

		print(">> Class Parameters ::")
		print("Data Shape :: {}".format(self.data.shape))
		print("Initial Amount :: ${}.00/-".format(self.initial_amount))
		print("Stocks :: {}".format(self.data.tic.unique()))

	def trainAndSaveA2C(self):

		print(">> A2C :: Training Begins!!")
		model_a2c = self.agent.get_model("a2c")

		if self.if_using_a2c:
			# set up logger
			tmp_path = RESULTS_DIR + '/a2c'
			new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
			# Set new logger
			model_a2c.set_logger(new_logger_a2c)

		# Train A2C DRL Agent
		trained_a2c = self.agent.train_model(model=model_a2c, 
                                			 tb_log_name='a2c',
                                			 total_timesteps=50000) if self.if_using_a2c else None

		# Save the trained agent
		trained_a2c.save(TRAINED_MODEL_DIR + "/agent_a2c") if self.if_using_a2c else None
		print(">> A2C :: Training End -> trained model saved @ {}".format(TRAINED_MODEL_DIR + "/agent_a2c"))


	def trainAndSaveDDPG(self):

		print(">> DDPG :: Training Begins!!")
		model_ddpg = self.agent.get_model("ddpg")

		if self.if_using_ddpg:
			# set up logger
			tmp_path = RESULTS_DIR + '/ddpg'
			new_logger_ddpg = configure(tmp_path, ["stdout", "csv", "tensorboard"])
			# Set new logger
			model_ddpg.set_logger(new_logger_ddpg)

		# Train DDPG DRL Agent
		trained_ddpg = self.agent.train_model(model=model_ddpg, 
		                                 	  tb_log_name='ddpg',
		                                 	  total_timesteps=50000) if self.if_using_ddpg else None

		# Save the trained agent
		trained_ddpg.save(TRAINED_MODEL_DIR + "/agent_ddpg") if self.if_using_ddpg else None
		print(">> DDPG :: Training End -> trained model saved @ {}".format(TRAINED_MODEL_DIR + "/agent_ddpg"))

	def trainAndSavePPO(self):

		print(">> PPO :: Training Begins!!")

		PPO_PARAMS = {
		    "n_steps": 2048,
		    "ent_coef": 0.01,
		    "learning_rate": 0.00025,
		    "batch_size": 128,
		}

		model_ppo = self.agent.get_model("ppo", model_kwargs = PPO_PARAMS)

		if self.if_using_ppo:
			# set up logger
			tmp_path = RESULTS_DIR + '/ppo'
			new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
			# Set new logger
			model_ppo.set_logger(new_logger_ppo)

		# Train PPO DRL Agent
		trained_ppo = self.agent.train_model(model=model_ppo, 
                                			 tb_log_name='ppo',
                                			 total_timesteps=200000) if self.if_using_ppo else None

		# Save the trained agent
		trained_ppo.save(TRAINED_MODEL_DIR + "/agent_ppo") if self.if_using_ppo else None
		print(">> PPO :: Training End -> trained model saved @ {}".format(TRAINED_MODEL_DIR + "/agent_ppo"))


	def trainAndSaveTD3(self):
		
		print(">> TD3 :: Training Begins!!")

		TD3_PARAMS = {
			"batch_size": 100, 
            "buffer_size": 1000000, 
            "learning_rate": 0.001
        }

		model_td3 = self.agent.get_model("td3",model_kwargs = TD3_PARAMS)

		if self.if_using_td3:
			# set up logger
			tmp_path = RESULTS_DIR + '/td3'
			new_logger_td3 = configure(tmp_path, ["stdout", "csv", "tensorboard"])
			# Set new logger
			model_td3.set_logger(new_logger_td3)

		# Train TD3 DRL Agent
		trained_td3 = self.agent.train_model(model=model_td3, 
                                		tb_log_name='td3',
                                		total_timesteps=50000) if self.if_using_td3 else None

		# Save the trained agent
		trained_td3.save(TRAINED_MODEL_DIR + "/agent_td3") if self.if_using_td3 else None

		print(">> TD3 :: Training End -> trained model saved @ {}".format(TRAINED_MODEL_DIR + "/agent_td3"))


	def trainAndSaveSAC(self):
		
		print(">> SAC :: Training Begins!!")

		SAC_PARAMS = {
		    "batch_size": 128,
		    "buffer_size": 100000,
		    "learning_rate": 0.0001,
		    "learning_starts": 100,
		    "ent_coef": "auto_0.1",
		}

		model_sac = self.agent.get_model("sac",model_kwargs = SAC_PARAMS)

		if self.if_using_sac:
			# set up logger
			tmp_path = RESULTS_DIR + '/sac'
			new_logger_sac = configure(tmp_path, ["stdout", "csv", "tensorboard"])
			# Set new logger
			model_sac.set_logger(new_logger_sac)

		# Train SAC DRL Agent
		trained_sac = self.agent.train_model(model=model_sac, 
                                			 tb_log_name='sac',
                                			 total_timesteps=70000) if self.if_using_sac else None

		# Save the trained agent
		trained_sac.save(TRAINED_MODEL_DIR + "/agent_sac") if self.if_using_sac else None

		print(">> SAC :: Training End -> trained model saved @ {}".format(TRAINED_MODEL_DIR + "/agent_sac"))


if __name__ == '__main__':

	check_and_make_directories([TRAINED_MODEL_DIR, RESULTS_DIR, TENSORBOARD_LOG_DIR])

	train_data = pd.read_csv(DATASETS_DIR+'/train_data.csv')
	initial_amount = 1000000

	if train_data.shape[0]:
		train_data = train_data.set_index(train_data.columns[0])
		train_data.index.names = ['']
	else:
		print(">> Error in loading training data from CSV!!")
		sys.exit(0)

	train_class = DRLTrainClass(train_data, initial_amount)
	train_class.printClassParameters()
	train_class.trainAndSaveA2C()
	train_class.trainAndSaveDDPG()
	train_class.trainAndSavePPO()
	train_class.trainAndSaveTD3()
	train_class.trainAndSaveSAC()
