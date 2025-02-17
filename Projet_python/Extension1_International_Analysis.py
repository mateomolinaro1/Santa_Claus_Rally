import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 20)

from DataBase import DataCompute
from SantaClausAnalysis import SantaClausBucket
from SantaClausAnalysis import statistics
from datetime import timedelta

# -------------------------------------------- DATASET CRAFT ----------------------------------------------
# Christian and non christian based index lists
christian_indices_list = [
    "^IXIC", "^GSPC", "^RUT", "^GSPTSE", "^MXX",
    "^BVSP", "^FTSE", "^GDAXI", "^FCHI", "^AXJO"
]

non_christian_indices_list = [
    "^N225", "^HSI", "000001.SS", "^STI", "^BSESN",
    "^JKSE", "^TWII", "^KS11"
]

# Combined tickers
tickers = christian_indices_list + non_christian_indices_list

# Specific starting dates related to the Nippani Washer and Johnson (NWJ) 2015 paper
Nippani_start_dates = {
    "^IXIC": "1971-02-04", # //
    "^GSPC": "1950-01-02", # vs initial start "1927-12-30"
    "^RUT": "1987-09-09", # //
    "^GSPTSE": "1979-06-28", # //
    "^MXX": "1991-11-07",  # //
    "^BVSP": "1993-04-26", # //
    "^FTSE": "1984-01-02", # //
    "^GDAXI": "1990-11-25", # vs initial start "1987-12-30"
    "^FCHI": "1990-02-28", # //
    "^AXJO": "1992-11-22", # //
    "^N225": "1984-01-03", # vs initial start "1965-01-05"
    "^HSI": "1987-01-01", # //
    "000001.SS": "1990-12-18", # // vs decay start "1997-07-02"
    "^STI": "1987-12-27", # //
    "^BSESN": "1997-06-30", # //
    "^JKSE": "1997-06-30", # vs initial start "1990-04-06"
    "^TWII": "1997-07-01", # //
    "^KS11": "1997-06-30" # vs initial start "1996-12-11"
}

# Yahoo Finance specific starting index dates
Initial_start_dates = {
    "^IXIC": "1971-02-04", # real index beginning
    "^GSPC": "1927-12-30", # real index beginning
    "^RUT": "1987-09-10", # real beginning in 1978
    "^GSPTSE": "1979-06-28", # real beginning in 1961
    "^MXX": "1991-11-07",
    "^BVSP": "1993-04-27", # real beginning ten years before with inflation adjustments
    "^FTSE": "1984-01-03", # real index beginning
    "^GDAXI": "1987-12-30", # real index beginning
    "^FCHI": "1990-02-28", # real beginning in 1988
    "^AXJO": "1992-11-22",  # real beginning in 1980
    "^N225": "1965-01-04", # real beginning in 1950
    "^HSI": "1986-12-31", # real beginning in 1964
    "000001.SS": "1991-12-19", # année de départ
    "^STI": "1987-12-27", # real beginning in the 1970's.
    "^BSESN": "1997-06-30", # real beginning in 1979
    "^JKSE": "1990-04-05",
    "^TWII": "1997-07-01", # real beginning in1967
    "^KS11": "1996-12-10" # real beginning in the 1980's
}

# J.Patel starting date
Patel_start_date = "1999-12-31"

# J.Patel ending date
Patel_end_date = "2022-01-01"

# Commune actual end date
Actual_end_date = "2025-01-10"

# NWJ commune end date
Nippani_end_date = "2014-01-03"

# Index category
def get_nature(ticker):
    if ticker in christian_indices_list:
        return "Christian Based"
    elif ticker in non_christian_indices_list:
        return "Non-Christian Based"
    return "Unknown"

print("----------------------------------------- J.PATEL DATASET EXTENSION -----------------------------------------")

desc_stats_global = pd.DataFrame(columns=["Ticker", "Nature", "Days_type"])
test_stats_global = pd.DataFrame(columns=["Ticker", "Nature"])
ols_stats_global = pd.DataFrame(columns=["Ticker", "Nature"])

for ticker in tickers:
    # Index data initialisation
    base = DataCompute([ticker], Patel_start_date, Patel_end_date, False)
    data_dict = base.data_dict
    stats = statistics(None, None, data_dict)

    # Ticker returns and prices
    df_returns = stats.recup_df(data_dict, ticker, 'df_returns')
    df_prices = stats.recup_df(data_dict, ticker, 'df_prices')

    # Buckets Santa Claus
    bucket = SantaClausBucket(df_returns, df_prices)
    santa_claus_df = bucket.santa_claus_df
    other_df = bucket.other_df

    # Descriptive analysis
    stats = statistics(santa_claus_df, other_df, None)
    st_desc = stats.desc_stats

    nature = get_nature(ticker)

    desc_stats_global = pd.concat([
        desc_stats_global,
        st_desc.assign(Ticker=ticker, Nature=nature, Days_type=st_desc.index.str.replace(ticker + " - ", ""))
    ])
    test_stats_global = pd.concat([
        test_stats_global,
        stats.test_stats.assign(Ticker=ticker, Nature=nature)
    ])
    ols_stats_global = pd.concat([
        ols_stats_global,
        stats.OLS_stats.assign(Ticker=ticker, Nature=nature)
    ])

# Final DataFrames form
desc_stats_global = desc_stats_global.set_index(['Ticker', 'Nature', 'Days_type'], drop=True)
test_stats_global = test_stats_global.set_index(['Ticker', 'Nature'], drop=True)
ols_stats_global = ols_stats_global.set_index(['Ticker', 'Nature'], drop=True)

print("\nDescriptive Statistics")
print(desc_stats_global)

print("\nStatistics Tests")
print(test_stats_global)

print("\nDummy Regression Statistics")
print(ols_stats_global)

print("\n----------------------------------------- NIPPANI DATASET EXTENSION -----------------------------------------")

# DataFrames initialisation
desc_stats_global = pd.DataFrame(columns=["Ticker", "Nature", "Days_type"])
test_stats_global = pd.DataFrame(columns=["Ticker", "Nature"])
ols_stats_global = pd.DataFrame(columns=["Ticker", "Nature"])

for ticker in tickers:
    start_date = Nippani_start_dates[ticker]

    # Ticker data initialisation
    base = DataCompute([ticker], start_date, Nippani_end_date, False)
    data_dict = base.data_dict
    stats = statistics(None, None, data_dict)

    # Ticker returns and prices
    df_returns = stats.recup_df(data_dict, ticker, 'df_returns')
    df_prices = stats.recup_df(data_dict, ticker, 'df_prices')

    # Buckets Santa Claus
    bucket = SantaClausBucket(df_returns, df_prices)
    santa_claus_df = bucket.santa_claus_df
    other_df = bucket.other_df

    # Descriptive analysis
    stats = statistics(santa_claus_df, other_df, None)
    st_desc = stats.desc_stats

    nature = get_nature(ticker)

    desc_stats_global = pd.concat([
        desc_stats_global,
        st_desc.assign(Ticker=ticker, Nature=nature, Days_type=st_desc.index.str.replace(ticker + " - ", ""))
    ])
    test_stats_global = pd.concat([
        test_stats_global,
        stats.test_stats.assign(Ticker=ticker, Nature=nature)
    ])
    ols_stats_global = pd.concat([
        ols_stats_global,
        stats.OLS_stats.assign(Ticker=ticker, Nature=nature)
    ])

# Final DataFrames form
desc_stats_global = desc_stats_global.set_index(['Ticker', 'Nature', 'Days_type'], drop=True)
test_stats_global = test_stats_global.set_index(['Ticker', 'Nature'], drop=True)
ols_stats_global = ols_stats_global.set_index(['Ticker', 'Nature'], drop=True)

print("\nDescriptive Statistics")
print(desc_stats_global)

print("\nStatistics Tests")
print(test_stats_global)

print("\nDummy Regression Statistics")
print(ols_stats_global)

print("\n----------------------------------------- ENTIRE DATASET EXTENSION -----------------------------------------")

# DataFrames initialisation
desc_stats_global = pd.DataFrame(columns=["Ticker", "Nature", "Days_type"])
test_stats_global = pd.DataFrame(columns=["Ticker", "Nature"])
ols_stats_global = pd.DataFrame(columns=["Ticker", "Nature"])
ols_variants_global = pd.DataFrame(columns=["Ticker", "Nature", "HC Adj."])

for ticker in tickers:
    start_date = Initial_start_dates[ticker]

    # Ticker data initialisation
    base = DataCompute([ticker], start_date, Actual_end_date,False)
    data_dict = base.data_dict
    stats = statistics(None, None, data_dict)

    # Ticker returns and prices
    df_returns = stats.recup_df(data_dict, ticker, 'df_returns')
    df_prices = stats.recup_df(data_dict, ticker, 'df_prices')

    # Buckets Santa Claus
    bucket = SantaClausBucket(df_returns, df_prices)
    santa_claus_df = bucket.santa_claus_df
    other_df = bucket.other_df

    # Descriptive analysis
    stats = statistics(santa_claus_df, other_df, None,extratest=True)
    st_desc = stats.desc_stats

    nature = get_nature(ticker)

    desc_stats_global = pd.concat([
        desc_stats_global,
        st_desc.assign(Ticker=ticker, Nature=nature, Days_type=st_desc.index.str.replace(ticker + " - ", ""))
    ])
    test_stats_global = pd.concat([
        test_stats_global,
        stats.test_stats.assign(Ticker=ticker, Nature=nature)
    ])
    ols_stats_global = pd.concat([
        ols_stats_global,
        stats.OLS_stats.assign(Ticker=ticker, Nature=nature)
    ])
    ols_variants_global = pd.concat([
        ols_variants_global,
        stats.OLS_variants_stats.assign(Ticker=ticker, Nature=nature)
    ])

# Final DataFrame form
desc_stats_global = desc_stats_global.set_index(['Ticker', 'Nature', 'Days_type'], drop=True)
test_stats_global = test_stats_global.set_index(['Ticker', 'Nature'], drop=True)
ols_stats_global = ols_stats_global.set_index(['Ticker', 'Nature'], drop=True)

print("\nDescriptive Statistics")
print(desc_stats_global)

print("\nStatistics Tests")
print(test_stats_global)

print("\nDummy Regression Statistics")
print(ols_stats_global)

print("\n------------------------------------------------------- SCR THRESHOLD LEVEL ------------------------------------------------------")

# Significant threshold test
threshold = 0.05

consolidated_results = []

for ticker, start_date in Initial_start_dates.items():
    # Data initialisation
    base = DataCompute([ticker], start_date, Actual_end_date, export_data=False)
    data_dict = base.data_dict

    # Ticker returns and prices
    df_returns = data_dict[ticker]["df_returns"]
    df_prices = data_dict[ticker]["df_prices"]

    current_start_date = pd.to_datetime(start_date)
    subset_returns = df_returns
    subset_prices = df_prices

    p_values = []
    # Year by year period sweep
    while current_start_date < pd.to_datetime(Actual_end_date):

        # Check if enough sub periods data
        if subset_returns.empty or subset_prices.empty:
            print(f"No available data after {current_start_date}.")
            break

        # Buckets Santa Claus
        bucket = SantaClausBucket(subset_returns, subset_prices)
        santa_claus_df = bucket.santa_claus_df
        other_df = bucket.other_df

        stats = statistics(santa_claus_df, other_df, None)

        # β1 p-value extraction
        p_value = stats.OLS_stats.loc[ticker, "Corrected p-val β1"]
        p_values.append(p_value)

        # Check if the p-value goes beyond the threshold
        if p_value > threshold:
            print(f"\nSCR Breaking date of {ticker} observed on {current_start_date.strftime('%Y-%m-%d')} for a {p_value:.4f} p-value level.")
            break

        # Next year end date computation
        current_end_date = current_start_date + timedelta(days=365)
        current_end_date_str = current_end_date.strftime('%Y-%m-%d')

        # Checks if the end date goes eventually beyond last end date
        if current_end_date_str > Actual_end_date:
            current_end_date_str = Actual_end_date

        subset_returns = df_returns.loc[current_end_date.strftime('%Y-%m-%d'):Actual_end_date]
        subset_prices = df_prices.loc[current_end_date.strftime('%Y-%m-%d'):Actual_end_date]

        # Change the next current start date with the computed end date
        current_start_date = current_end_date

    # Case if each sub period β1 is significant
    else:
        print(f"\nAll period remains significant for {ticker} until the end of the dataset.")

    # SCR Breaking date
    break_date = current_start_date.strftime('%Y-%m-%d')

    # Chow test
    bucket = SantaClausBucket(df_returns, df_prices)
    santa_claus_df = bucket.santa_claus_df
    other_df = bucket.other_df

    # Checks the stationary of the global return serie
    stationary_results = statistics.check_adf_stationary(df_returns)

    # Check if all columns in "ADF_Conclusion" from stationary_results are "Stationary"
    if stationary_results["ADF_Conclusion"].values[0] == "Stationary":
        # All columns are stationary, populate the results
        stats = statistics(santa_claus_df, other_df, None)
        chow_test_result = stats.chow_test(santa_claus_df, other_df, break_date)

        result = {
            "Ticker": ticker,
            "Break_Date": break_date,
            "p_value (β1)": p_values[-1] if p_values else None,
            "Chow_F_stat": chow_test_result["F-stat"].values[0] if "F-stat" in chow_test_result else "Insufficient Data",
            "Chow_p_value": chow_test_result["p-value"].values[0] if "p-value" in chow_test_result else "Insufficient Data",
            "Chow_RSS_full": chow_test_result["RSS_full"].values[0] if "RSS_full" in chow_test_result else "Insufficient Data",
            "Chow_RSS_part1": chow_test_result["RSS_part1"].values[0] if "RSS_part1" in chow_test_result else "Insufficient Data",
            "Chow_RSS_part2": chow_test_result["RSS_part2"].values[0] if "RSS_part2" in chow_test_result else "Insufficient Data",
            "Chow_n_full": chow_test_result["n_full"].values[0] if "n_full" in chow_test_result else "Insufficient Data",
            "Chow_n_part1": chow_test_result["n_part1"].values[0] if "n_part1" in chow_test_result else "Insufficient Data",
            "Chow_n_part2": chow_test_result["n_part2"].values[0] if "n_part2" in chow_test_result else "Insufficient Data",
        }
        consolidated_results.append(result)
    else:
        # If any column is not stationary, write "Non Stationary" for all fields
        result = {
            "Ticker": ticker,
            "Break_Date": break_date,
            "p_value (β1)": "Non Stationary",
            "Chow_F_stat": "Non Stationary",
            "Chow_p_value": "Non Stationary",
            "Chow_RSS_full": "Non Stationary",
            "Chow_RSS_part1": "Non Stationary",
            "Chow_RSS_part2": "Non Stationary",
            "Chow_n_full": "Non Stationary",
            "Chow_n_part1": "Non Stationary",
            "Chow_n_part2": "Non Stationary",
        }
        consolidated_results.append(result)

results_df = pd.DataFrame(consolidated_results)

results_df.set_index("Ticker", inplace=True)

print("\nSCR breaking date Chow test results")
print(results_df)
