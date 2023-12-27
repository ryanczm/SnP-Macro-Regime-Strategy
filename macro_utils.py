import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
sns.set_style('darkgrid')

def retrieve_sheets(path):
    """
    Retrieve data from Excel sheets and process it.
    Loads data from specified Excel sheets, performs necessary preprocessing, and returns three DataFrames.

    :param path: Path to the Excel file
    :type path: str
    :return: Three DataFrames containing macroeconomic data, prices, and yields
    :rtype: tuple (pd.DataFrame, pd.DataFrame, pd.DataFrame)
    """
    
    dfs = pd.read_excel(path, parse_dates=True, sheet_name=None, index_col='Date')
    macro = dfs['Macro'].fillna(method='bfill').resample('Q').mean().round(2)
    prices = dfs['Prices'].fillna(method='bfill')
    yields = dfs['Yield']

    return macro, prices, yields


def create_returns(prices, yields):
    """
    Calculate quarterly returns based on price and yield data.
    Calculates the log returns for the S&P 500, Gold, and US 10YR Bonds, and then sums them up on a quarterly basis.

    :param prices: DataFrame containing price data for S&P 500 and Gold
    :type prices: pd.DataFrame
    :param yields: DataFrame containing yield data for US 10YR Bonds
    :type yields: pd.DataFrame
    :return: DataFrame with quarterly sum of log returns for S&P 500, Gold, and US 10YR Bonds
    :rtype: pd.DataFrame
    """
    prices['SR'] = np.log(prices['S&P 500']) - np.log(prices['S&P 500'].shift(1))
    prices['GR'] = np.log(prices['Gold']) - np.log(prices['Gold'].shift(1))
    yields['YR'] = np.log(yields['US 10YR Bonds']) - np.log(yields['US 10YR Bonds'].shift(1))
    
    rets = pd.merge(prices, yields['YR'], left_index=True, right_index=True)[['SR', 'GR', 'YR']]
    return rets.resample('Q').sum()


def assign_regimes(df, macro):
    """
    Assign economic regimes based on macroeconomic indicators.
    Assigns economic regimes based on changes in year-over-year GDP and CPI growth rates.

    :param df: DataFrame containing returns data
    :type df: pd.DataFrame
    :param macro: DataFrame containing macroeconomic indicators (e.g., GDP YOY, CPI YOY)
    :type macro: pd.DataFrame
    :return: DataFrame with assigned economic regimes
    :rtype: pd.DataFrame
    """
    regimes = pd.DataFrame(index=macro.index)
    regimes['g'] = np.where(macro['GDP YOY'] > (macro['GDP YOY'].shift(1)), 1, 0)
    regimes['i'] = np.where(macro['CPI YOY'] > (macro['CPI YOY'].shift(1)), 1, 0)
    regimes['regime'] = (regimes.g.astype(str) + regimes.i.astype(str)).apply(lambda x: int(x, 2))
    print(regimes.groupby('regime').count().g)
    return df.merge(regimes.regime, left_index=True, right_index=True, how='inner').dropna()


def split_rets(rets, date):
    """
    Split returns data into training and testing sets based on a specified date.

    :param rets: DataFrame containing returns data
    :type rets: pd.DataFrame
    :param date: Date used for splitting the data into training and testing sets
    :type date: pd.Timestamp or str
    :return: Two DataFrames representing the training and testing sets
    :rtype: tuple (pd.DataFrame, pd.DataFrame)
    """
    train, test = rets[rets.index < date], rets[rets.index >= date]
    return train, test


def sharpe(return_series, n, rf):
    """
    Calculate the Sharpe ratio for a given return series.

    :param return_series: Series containing the returns
    :type return_series: pd.Series
    :param n: Number of periods per year
    :type n: int
    :param rf: Risk-free rate
    :type rf: float
    :return: Sharpe ratio
    :rtype: float
    """
    mean = (return_series.mean() * n) - rf
    sigma = return_series.std() * np.sqrt(n) 
    return mean/sigma


def create_weights(rets):
    """
    Create weights based on Sharpe ratios for different economic regimes.

    :param rets: DataFrame containing returns and economic regimes
    :type rets: pd.DataFrame
    :return: Two DataFrames representing Sharpe ratios and corresponding weights
    :rtype: tuple (pd.DataFrame, pd.DataFrame)
    """
    alphas = rets.groupby('regime')[['SR', 'GR', 'YR']].apply(sharpe, n=4, rf=0.00) 
    weights = alphas.div(abs(alphas).sum(axis=1), axis=0)
    return alphas, weights


def quarterly_rets(row, weights):
    """
    Calculate quarterly returns for the strategy based on assigned weights.

    :param row: Row containing returns and economic regime
    :type row: pd.Series
    :param weights: DataFrame containing weights for different economic regimes
    :type weights: pd.DataFrame
    :return: Quarterly returns for the strategy
    :rtype: float
    """
    regime_weights = weights.iloc[int(row[-1])]
    return regime_weights.dot(row[:-1])


def get_strat_rets(df, weights):
    """
    Get strategy returns based on assigned weights.

    :param df: DataFrame containing returns and economic regimes
    :type df: pd.DataFrame
    :return: Series representing strategy returns
    :rtype: pd.Series
    """
    return df.apply(quarterly_rets, weights=weights, axis=1)


def construct_perf(df, benchmark, weights):
    """
    Construct performance DataFrame comparing the strategy and a benchmark.

    :param df: DataFrame containing returns and economic regimes
    :type df: pd.DataFrame
    :param benchmark: Series representing benchmark returns
    :type benchmark: pd.Series
    :return: DataFrame containing 'Strategy', 'Long S&P', and 'Tracking Error' columns
    :rtype: pd.DataFrame
    """
    perf = pd.DataFrame({'Strategy': get_strat_rets(df, weights), 'Long S&P': benchmark}, index=df.index)
    perf['Tracking Error'] = perf.Strategy - perf['Long S&P']
    return perf


def ir(tracking_error, n):
    """
    Calculate the Information Ratio (IR) for a given tracking error.

    :param tracking_error: Series representing the tracking error
    :type tracking_error: pd.Series
    :param n: Number of observations (e.g., number of periods)
    :type n: int
    :return: Information Ratio (IR)
    :rtype: float
    """
    return (tracking_error.mean() * (n) / (tracking_error.std() * np.sqrt(n))).round(3)