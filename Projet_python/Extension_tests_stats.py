
import pandas as pd


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)
pd.set_option('display.max_columns',30)

from DataBase import DataCompute
from SantaClausAnalysis import SantaClausBucket
from SantaClausAnalysis import statistics


# Yahoo Finance ticker of the index on which we wish to launch the analysis.
tickers = ['^GSPC', '^IXIC']

#  Start and end dates of the indices
start_date = '1999-12-31'
end_date = '2022-01-01'

# Initialization of data for each ticker.
base = DataCompute(tickers, start_date, end_date)
df_prices = base.df_prices
df_returns = base.df_returns
data_dict = base.data_dict

#  Calculation of descriptive statistics for all tickers.
stats = statistics(None, None,data_dict,extratest = True,bootstrap =True)
stats_descriptives = stats.desc_stats
print("Statistiques descriptives des indices")
print(stats_descriptives)
print("--------------------------------------")


# Replication of the research paper. ------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Period 2000-2021---------------------------------------------------------------------------------------------------------

# Iteration on each ticker to launch the paper analysis.
for ticker in tickers:

    # Retrieval of price and return data for the ticker.
    df_returns = stats.recup_df(data_dict,ticker,'df_returns')
    df_prices = stats.recup_df(data_dict, ticker, 'df_prices')

    # Creation of buckets dividing returns into Santa Claus period and the other.
    bucket = SantaClausBucket(df_returns,df_prices)
    santa_claus_df = bucket.santa_claus_df
    other_df = bucket.other_df
    days = bucket.santa_claus_days  # To check

    # Calculation of descriptive statistics for the two separate buckets for the ticker.
    stats = statistics(santa_claus_df, other_df,None,extratest = True,bootstrap =True)
    stats_descriptives_separate = stats.desc_stats
    stats_test_separate = stats.test_stats
    stats_OLS = stats.OLS_stats

    # Display of the results
    print(f"Analysis for the ticker. {ticker}")
    print("---------------------------------------")

    print("Descriptive statistics of the separated periods.")
    print(stats_descriptives_separate)
    print("--------------------------------------")

    print("Test statistics for the separated periods.")
    print(stats_test_separate)
    print("--------------------------------------")

    print("Regression results.")
    print(stats_OLS)
    print("--------------------------------------")

    # Period 2000-2009---------------------------------------------------------------------------------------------------------
    print("Période 2000-2009")

    #  Start and end dates of the indices
    start_date = '2000-01-01'
    end_date = '2009-12-31'

    # Filtering the initial data to obtain the correct price and return dataframe according to the dates.
    df_filtered_returns = DataCompute.modif_df(df_returns, start_date, end_date)
    df_filtered_prices = DataCompute.modif_df(df_prices, start_date, end_date)

    # Creation of buckets dividing returns into Santa Claus period and the other.
    bucket = SantaClausBucket(df_filtered_returns,df_filtered_prices)
    santa_claus_df = bucket.santa_claus_df
    other_df = bucket.other_df
    days = bucket.santa_claus_days  # pour verif

    # Calculation of descriptive statistics for the two separate buckets for the ticker.
    stats = statistics(santa_claus_df, other_df,None,extratest = True,bootstrap =True)
    stats_descriptives_separate = stats.desc_stats
    stats_test_separate = stats.test_stats
    stats_OLS = stats.OLS_stats

    # Display of the results
    print("Statistiques descriptives des périodes separées")
    print(stats_descriptives_separate)
    print("--------------------------------------")

    print("Statistiques de test des périodes separées")
    print(stats_test_separate)
    print("--------------------------------------")

    print("Résultat de la regression")
    print(stats_OLS)
    print("--------------------------------------")

    # Period 2010-2021---------------------------------------------------------------------------------------------------------

    print("Période 2010-2021")

    #  Start and end dates of the indices
    start_date = '2010-01-01'
    end_date = '2021-12-31'

    # Filtering the initial data to obtain the correct price and return dataframe according to the dates.
    df_filtered_returns = DataCompute.modif_df(df_returns, start_date, end_date)
    df_filtered_prices = DataCompute.modif_df(df_prices, start_date, end_date)

    # Creation of buckets dividing returns into Santa Claus period and the other.
    bucket = SantaClausBucket(df_filtered_returns,df_filtered_prices)
    santa_claus_df = bucket.santa_claus_df
    other_df = bucket.other_df
    days = bucket.santa_claus_days  # pour verif

    # Calculation of descriptive statistics for the two separate buckets for the ticker.
    stats = statistics(santa_claus_df, other_df,None,extratest = True,bootstrap =True)
    stats_descriptives_separate = stats.desc_stats
    stats_test_separate = stats.test_stats
    stats_OLS = stats.OLS_stats

    # Display of the results
    print("Statistiques descriptives des périodes separées")
    print(stats_descriptives_separate)
    print("--------------------------------------")

    print("Statistiques de test des périodes separées")
    print(stats_test_separate)
    print("--------------------------------------")

    print("Résultat de la regression")
    print(stats_OLS)
    print("--------------------------------------")

    # Economic cycles  ---------------------------------------------------------------------------------------------------------
    print("Economic cycles")

    # Dates of the economic expansion and recession cycles from the NBER used in the paper.
    expansions_periods = [["2000-01-01", "2001-03-31"], ["2001-12-01", "2007-12-31"], ["2009-07-01", "2020-02-29"],
                          ["2020-05-01", "2021-12-31"]]
    recessions_periods = [["2001-04-01", "2001-11-30"], ["2008-01-01", "2009-06-30"], ["2020-03-01", "2020-04-30"]]

    # Creation of price and return dataframes for the expansion and recession periods.
    df_expansion_combined = pd.DataFrame()
    df_recession_combined = pd.DataFrame()
    df_prices_expansion_combined = pd.DataFrame()
    df_prices_recession_combined = pd.DataFrame()

    # Loop for each expansion period to retrieve the corresponding data.
    for period in expansions_periods:
        start_date, end_date = period
        df_filtered_expansion = DataCompute.modif_df(df_returns, start_date, end_date)
        df_expansion_combined = pd.concat([df_expansion_combined, df_filtered_expansion])

        df_filtered_prices_expansion = DataCompute.modif_df(df_prices, start_date, end_date)
        df_prices_expansion_combined = pd.concat([df_prices_expansion_combined, df_filtered_prices_expansion])

    # Loop for each recession period to retrieve the corresponding data.
    for period in recessions_periods:
        start_date, end_date = period
        df_filtered_recession = DataCompute.modif_df(df_returns, start_date, end_date)
        df_recession_combined = pd.concat([df_recession_combined, df_filtered_recession])

        df_filtered_prices_recession = DataCompute.modif_df(df_prices, start_date, end_date)
        df_prices_recession_combined = pd.concat([df_prices_recession_combined, df_filtered_prices_recession])


    # Expansion Periods ------------

    # Creation of buckets dividing returns into Santa Claus period and the other.
    bucket = SantaClausBucket(df_expansion_combined,df_prices_expansion_combined)
    santa_claus_df = bucket.santa_claus_df
    other_df = bucket.other_df
    days = bucket.santa_claus_days  # pour verif

    # Calculation of descriptive statistics for the two separate buckets for the ticker.
    stats = statistics(santa_claus_df, other_df,None,extratest = True,bootstrap =True)
    stats_descriptives_separate = stats.desc_stats
    stats_test_separate = stats.test_stats
    stats_OLS = stats.OLS_stats

    # Display of the results
    print("Statistiques descriptives des périodes separées")
    print(stats_descriptives_separate)
    print("--------------------------------------")

    print("Statistiques de test des périodes separées")
    print(stats_test_separate)
    print("--------------------------------------")

    print("Résultat de la regression")
    print(stats_OLS)
    print("--------------------------------------")

    # Recession Periods ----------------

    # Creation of buckets dividing returns into Santa Claus period and the other.
    bucket = SantaClausBucket(df_recession_combined,df_prices_recession_combined)
    santa_claus_df = bucket.santa_claus_df
    other_df = bucket.other_df
    days = bucket.santa_claus_days  # pour verif

    # Calculation of descriptive statistics for the two separate buckets for the ticker.
    stats = statistics(santa_claus_df, other_df,None,extratest = True,bootstrap =True)
    stats_descriptives_separate = stats.desc_stats
    stats_test_separate = stats.test_stats
    stats_OLS = stats.OLS_stats

    # Display of the results
    print("Statistiques descriptives des périodes separées")
    print(stats_descriptives_separate)
    print("--------------------------------------")

    print("Statistiques de test des périodes separées")
    print(stats_test_separate)
    print("--------------------------------------")

    print("Résultat de la regression")
    print(stats_OLS)
    print("--------------------------------------")




