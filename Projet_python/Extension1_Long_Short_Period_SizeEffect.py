import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 20)

from DataBase import DataCompute
from SantaClausAnalysis import SantaClausBucket
from SantaClausAnalysis import statistics

print("\n-------------------------------------------------------- ADJUSTED SCR PERIOD ------------------------------------------------------")

tickers = ["^GSPC", "^IXIC", "^FTSE", "^AXJO"]
start_date = '1999-12-31'
end_date = '2022-01-01'

# DataFrames consolidés
desc_stats_global = pd.DataFrame(columns=["Ticker", "Days_type"])
test_stats_global = pd.DataFrame(columns=["Ticker"])
ols_stats_global = pd.DataFrame(columns=["Ticker"])

for ticker in tickers:
    # Ticker data initialisation
    base = DataCompute([ticker], start_date, end_date, False)
    data_dict = base.data_dict
    stats = statistics(None, None, data_dict)

    # Ticker returns and prices
    df_returns = stats.recup_df(data_dict, ticker, 'df_returns')
    df_prices = stats.recup_df(data_dict, ticker, 'df_prices')

    # Buckets Santa Claus Long Period
    bucket = SantaClausBucket(df_returns, df_prices)
    santa_claus_df = bucket.santa_claus_df
    other_df = bucket.other_df

    # Descriptive statistics analysis
    stats = statistics(santa_claus_df, other_df, None)
    st_desc = stats.desc_stats

    desc_stats_global = pd.concat([
        desc_stats_global,
        st_desc.assign(Ticker=ticker, Nature="Long Period (7d)", Days_type=st_desc.index.str.replace(ticker + " - ", ""))
    ])
    test_stats_global = pd.concat([
        test_stats_global,
        stats.test_stats.assign(Ticker=ticker, Nature="Long Period (7d)")
    ])
    ols_stats_global = pd.concat([
        ols_stats_global,
        stats.OLS_stats.assign(Ticker=ticker, Nature="Long Period (7d)")
    ])

    # Buckets Santa Claus Short Period
    bucket = SantaClausBucket(df_returns, df_prices, False)
    santa_claus_df = bucket.santa_claus_df
    other_df = bucket.other_df

    # Global analysis
    stats = statistics(santa_claus_df, other_df, None)
    st_desc = stats.desc_stats

    desc_stats_global = pd.concat([
        desc_stats_global,
        st_desc.assign(Ticker=ticker, Nature="Short Period (5d)",
                       Days_type=st_desc.index.str.replace(ticker + " - ", ""))
    ])
    test_stats_global = pd.concat([
        test_stats_global,
        stats.test_stats.assign(Ticker=ticker, Nature="Short Period (5d)")
    ])
    ols_stats_global = pd.concat([
        ols_stats_global,
        stats.OLS_stats.assign(Ticker=ticker, Nature="Short Period (5d)")
    ])
# Final DataFrames form
desc_stats_global = desc_stats_global.set_index(['Ticker', 'Nature', 'Days_type'], drop=True)
test_stats_global = test_stats_global.set_index(['Ticker', 'Nature'], drop=True)
ols_stats_global = ols_stats_global.set_index(['Ticker', 'Nature'], drop=True)

print("\n Descriptives Statistics")
print(desc_stats_global)

print("\nStatistics Tests")
print(test_stats_global)

print("\nDummy Regression Statistics")
print(ols_stats_global)

print("\n-------------------------------------------------------- RESIZED SCR ------------------------------------------------------")

tickers = [
    "^GSPC",  # S&P 500 (Large-Cap): market capitalizations typically above $10 billion, often much higher.
    "^SP400",  # S&P MidCap 400 (Mid-Cap): market capitalizations typically between $2.4 billion and $10 billion.
    "^SP600",  # S&P SmallCap 600 (Small-Cap): market capitalizations typically between $300 million and $2.4 billion.
    "^FTSE",  # FTSE 100 (Large-Cap): market capitalizations for the 100 largest companies on the London Stock Exchange.
    "^FTMC",  # FTSE 250 (Mid-Cap): market capitalizations for the 101st to 350th largest companies on the LSE.
    "^FTSC"  # FTSE SmallCap (Small-Cap): market capitalizations for smaller companies just below the FTSE 250.
]

# Data initialisation
base = DataCompute(tickers, start_date, end_date,export_data=False)
data_dict = base.data_dict
stats = statistics(None, None, data_dict)

# Merged dataframes
desc_stats_global = pd.DataFrame(columns=["Ticker", "Days_type"])
test_stats_global = pd.DataFrame(columns=["Ticker"])
ols_stats_global = pd.DataFrame(columns=["Ticker"])

for ticker in tickers:
    # Ticker initialised data
    base = DataCompute([ticker], start_date, end_date, False)
    data_dict = base.data_dict
    stats = statistics(None, None, data_dict)

    # Prices and returns ticker dataframes
    df_returns = stats.recup_df(data_dict, ticker, 'df_returns')
    df_prices = stats.recup_df(data_dict, ticker, 'df_prices')

    # Buckets Santa Claus
    bucket = SantaClausBucket(df_returns, df_prices)
    santa_claus_df = bucket.santa_claus_df
    other_df = bucket.other_df

    # Descriptive analysis
    stats = statistics(santa_claus_df, other_df, None)
    st_desc = stats.desc_stats


    desc_stats_global = pd.concat([
        desc_stats_global,
        st_desc.assign(Ticker=ticker, Days_type=st_desc.index.str.replace(ticker + " - ", ""))
    ])
    test_stats_global = pd.concat([
        test_stats_global,
        stats.test_stats.assign(Ticker=ticker)
    ])
    ols_stats_global = pd.concat([
        ols_stats_global,
        stats.OLS_stats.assign(Ticker=ticker)
    ])

# Final dataframe form
desc_stats_global = desc_stats_global.set_index(['Ticker', 'Days_type'], drop=True)
test_stats_global = test_stats_global.set_index(['Ticker'], drop=True)
ols_stats_global = ols_stats_global.set_index(['Ticker'], drop=True)

print("\n Descriptives Statistics")
print(desc_stats_global)

print("\nStatistics Tests")
print(test_stats_global)

print("\nDummy Regression Statistics")
print(ols_stats_global)

print("\n------------------------------------------------------- SCR SIZE EFFECT ------------------------------------------------------")

diff_pairs = {
    "US SMALL-MID": ("^SP600", "^SP400"),
    "US SMALL-LARGE": ("^SP600", "^GSPC"),
    "US MID-LARGE": ("^SP400", "^GSPC"),
    "UK SMALL-MID": ("^FTSC", "^FTMC"),
    "UK SMALL-LARGE": ("^FTSC", "^FTSE"),
    "UK MID-LARGE": ("^FTMC", "^FTSE")
}

# Data initialisation
base = DataCompute(tickers, start_date, end_date,export_data=False)
data_dict = base.data_dict
stats = statistics(None, None, data_dict)

# Consolidated dataframes
desc_stats_global = pd.DataFrame(columns=["Ticker", "Nature", "Days_type"])
test_stats_global = pd.DataFrame(columns=["Ticker", "Nature"])
ols_stats_global = pd.DataFrame(columns=["Ticker", "Nature"])

# Difference between two ticker return series
def calculate_diff(df_values, ticker1, ticker2, pair):
    df = df_values[ticker1] - df_values[ticker2]
    if isinstance(df, pd.DataFrame):
        df = df.rename(columns={df.columns[0]: pair})
        return df
    else:
        df.name = pair
        return pd.DataFrame(df)

# Index differences analysis
for name, (ticker1, ticker2) in diff_pairs.items():
    # Difference computation between two ticker returns
    df_diff_returns = calculate_diff(stats.split_df(data_dict, [ticker1,ticker2], 'df_returns'),
                                      ticker1, ticker2, name)

    df_prices = stats.split_df(data_dict, [ticker1,ticker2], 'df_prices')

    # Buckets Santa Claus for the difference returns
    bucket = SantaClausBucket(df_diff_returns, df_prices)
    santa_claus_df = bucket.santa_claus_df
    other_df = bucket.other_df

    # Descriptive statistics analysis
    stats = statistics(santa_claus_df, other_df, None)
    st_desc = stats.desc_stats

    desc_stats_global = pd.concat([
        desc_stats_global,
        st_desc.assign(Ticker=ticker1.replace("^","") + "-" + ticker2.replace("^",""), Nature=name, Days_type=st_desc.index.str.replace(name + " - ", ""))
    ])
    test_stats_global = pd.concat([
        test_stats_global,
        stats.test_stats.assign(Ticker=ticker1.replace("^","") + "-" + ticker2.replace("^",""), Nature=name)
    ])
    ols_stats_global = pd.concat([
        ols_stats_global,
        stats.OLS_stats.assign(Ticker=ticker1.replace("^","") + "-" + ticker2.replace("^",""), Nature=name)
    ])

# Final dataframes form
desc_stats_global = desc_stats_global.set_index(['Ticker', 'Nature', 'Days_type'], drop=True)
test_stats_global = test_stats_global.set_index(['Ticker', 'Nature'], drop=True)
ols_stats_global = ols_stats_global.set_index(['Ticker', 'Nature'], drop=True)

print("\nDescriptive Statistics")
print(desc_stats_global)

print("\nStatistics Tests")
print(test_stats_global)

print("\nDummy Regression Statistics")
print(ols_stats_global)


