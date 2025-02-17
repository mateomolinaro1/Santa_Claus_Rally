from DataBase import DataCompute
from SantaClausAnalysis import statistics,SantaClausBucket
from VariablesAnalysis import MacroDataFormat,MacroAnalysis
import pandas as pd
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 20)

#Exemple d'utilisation
if __name__ == "__main__":

    tickers = ['^GSPC', '^IXIC','^STOXX', '^FCHI']

    start_date = '1999-12-31'
    end_date = '2022-01-01'

    base = DataCompute(tickers, start_date, end_date)
    df_prices = base.df_prices
    df_returns = base.df_returns
    data_dict = base.data_dict

    stats = statistics(None, None, data_dict)
    stats_descriptives = stats.desc_stats
    print("Statistiques descriptives des indices")
    print(stats_descriptives)
    print("--------------------------------------")

    #US---------------------------------------------------------------------------------------------------------

    for ticker in tickers:

        print(f"Analyse pour le ticker : {ticker}")
        df_returns = stats.recup_df(data_dict, ticker, 'df_returns')
        df_prices = stats.recup_df(data_dict, ticker, 'df_prices')

        bucket = SantaClausBucket(df_returns, df_prices)
        santa_claus_df = bucket.santa_claus_df
        other_df = bucket.other_df
        days = bucket.santa_claus_days  # pour verif

        # The full range of variable additions we tested for the US does not appear in the final script because the results were not very interesting.
        # file_names = ['T10Y2Y.xlsx','EURUSD.xlsx','DTB3.xlsx','DGS10.xlsx','DGS1.xlsx','BAA10Y.xlsx','VIX.xlsx']
        # files_with_variations =['T10Y2Y.xlsx','EURUSD.xlsx','DTB3.xlsx','DGS10.xlsx','DGS1.xlsx','BAA10Y.xlsx','VIX.xlsx']

        # Extension: Analysis of the relationship between Santa Claus rally returns and the evolution of market variables. ---------------------------------------------------------------------------------------------------------

        # File path containing the data to import and differentiate for stationarity.
        file_names = ['EURUSD.xlsx', 'VIX.xlsx']
        files_with_variations = ['EURUSD.xlsx', 'VIX.xlsx']

        # Instantiate the Macro analysis class with a parameter set to us=True,
        # which is not necessary for the variables we use since they are common to both regions, but it allows differentiating the file path for the US and Europe.
        macro_data = MacroDataFormat(santa_claus_df, other_df, file_names, us=True, data_dir='data',
                                     files_with_variations=files_with_variations)

        # Dataframe containing all the loaded data.
        all_data_macro = macro_data.get_data()

        # Retrieve the Santa Claus days.
        santa_claus_period = santa_claus_df.index

        # Regression identical to the one in the paper but with the HAC correction.
        variables = ['']
        combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        regression_results = macro_analysis.perform_combined_regression()
        print(regression_results)

        # Regression with the VIX as a daily variation as a new variable (entire series without applying a dummy).
        variables = ['VIX']
        combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        regression_results = macro_analysis.perform_combined_regression()
        print(regression_results)

        # Maximum Drawdown calculated on a rolling window (1-year window, calculated daily, without applying a dummy)).
        variables = ['Maxdd']
        combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        regression_results = macro_analysis.perform_combined_regression()
        print(regression_results)

        # Regression with the EURUSD as a daily variation as a new variable (entire series without applying a dummy).
        variables = ['EURUSD']
        combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        regression_results = macro_analysis.perform_combined_regression()
        print(regression_results)

        # Regression with the addition of three variables: MaxDD, VIX, and EURUSD as entire series without applying a dummy.
        variables = ['Maxdd', 'VIX', 'EURUSD']
        combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        regression_results = macro_analysis.perform_combined_regression()
        print(regression_results)

        # Regression with the MaxDD calculated in a 1-year rolling window, multiplied by the dummy of the Santa Claus return.
        variables = ["Maxdd"]
        combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        regression_results = macro_analysis.perform_combined_regression(surplus_analysis=True,
                                                                        var_surplus=['Maxdd'],
                                                                        drop_var_surplus=True,
                                                                        onlysurplus=True)
        print(regression_results)

        # Regression with the VIX as a daily variation, multiplied by the dummy of the Santa Claus return.
        variables = ["VIX"]
        combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        regression_results = macro_analysis.perform_combined_regression(surplus_analysis=True,
                                                                        var_surplus=['VIX'],
                                                                        drop_var_surplus=True,
                                                                        onlysurplus=True)
        print(regression_results)

        # Regression with the EURUSD as a daily variation, multiplied by the dummy of the Santa Claus return.
        variables = ["EURUSD"]
        combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        regression_results = macro_analysis.perform_combined_regression(surplus_analysis=True,
                                                                        var_surplus=['EURUSD'],
                                                                        drop_var_surplus=True,
                                                                        onlysurplus=True)
        print(regression_results)

        # Regression with the addition of three variables: MaxDD, VIX, and EURUSD as entire series, each multiplied by the dummy of the Santa Claus return for each variable.
        variables = ['Maxdd', 'VIX', 'EURUSD']
        combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        regression_results = macro_analysis.perform_combined_regression(surplus_analysis=True,
                                                                        var_surplus=variables,
                                                                        drop_var_surplus=True,
                                                                        onlysurplus=True)
        print(regression_results)
