TRAINED_MODEL_DIR = "../tech_tickers/trained_models"
TENSORBOARD_LOG_DIR = "../tech_tickers/tensorboard_log"
RESULTS_DIR = "../tech_tickers/results"
DATASETS_DIR = "../tech_tickers/datasets"
ACC_AMT_DATA_DIR = "../tech_tickers/action_val_data"
STATIC_DIR = "../tech_tickers/static"

# date format: '%Y-%m-%d'
TRAIN_START_DATE = "2010-01-01"
TRAIN_END_DATE = "2023-05-01"

TRADE_START_DATE = "2023-05-01"
TRADE_END_DATE = "2024-05-01"

# stockstats technical indicator column names
# check https://pypi.org/project/stockstats/ for different names
INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]