import pandas as pd
import numpy as np
import datetime
import yfinance as yf

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from ticker_list import TECH_TICKERS, DOW_30_TICKER
from config import INDICATORS, TRAIN_START_DATE, TRAIN_END_DATE, TRADE_START_DATE, TRADE_END_DATE

import itertools
import warnings

warnings.filterwarnings("ignore")

def getData(tic_list):
	print("Start::{} -> End::{}".format(TRAIN_START_DATE, TRADE_END_DATE))
	print("Tickers:: {}".format(tic_list))
	df_raw = YahooDownloader(start_date = TRAIN_START_DATE,
                             end_date = TRADE_END_DATE,
                             ticker_list = tic_list).fetch_data()

	return df_raw

def performFeatureEngineering(fin_data):

	fe = FeatureEngineer(use_technical_indicator=True,
                         tech_indicator_list = INDICATORS,
                         use_vix=True,
                         use_turbulence=True,
                         user_defined_feature = False)

	processed_data = fe.preprocess_data(fin_data)

	return processed_data

def restructureAndSplit(processed_data):

	# PART1: Re-structuring
	list_ticker = processed_data["tic"].unique().tolist()
	list_date = list(pd.date_range(processed_data['date'].min(),processed_data['date'].max()).astype(str))
	combination = list(itertools.product(list_date,list_ticker))

	processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed_data,on=["date","tic"],how="left")
	processed_full = processed_full[processed_full['date'].isin(processed_data['date'])]
	processed_full = processed_full.sort_values(['date','tic'])

	processed_full = processed_full.fillna(0)

	# PART2: Split Data

	train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
	trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)

	return (train, trade)

if __name__ == '__main__':
	
	fin_data = getData(TECH_TICKERS)
	processed_data = performFeatureEngineering(fin_data)
	train, trade = restructureAndSplit(processed_data)

	# Save as CSV
	train.to_csv('../tech_tickers/datasets/train_data.csv')
	trade.to_csv('../tech_tickers/datasets/trade_data.csv')
	print(">> Fetch & Save Operation Successful!!")
