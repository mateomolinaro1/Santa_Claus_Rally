import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
import statsmodels.api as sm

import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

from DataBase import DataCompute
from SantaClausAnalysis import SantaClausBucket
from SantaClausAnalysis import statistics


from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import het_arch


class MacroDataFormat:
    """
    Class that manages the data format with macro variables.
    """

    def __init__(self,santa_claus_df, other_df,file_names, us=True, data_dir='data', files_with_variations=None):
        """
        :param file_names: List of Excel file names.
        :param us: Boolean that defines whether the data is for the United States (default True) or for Europe (False).
        :param data_dir: Folder containing the Excel files (default 'data').
        :param files_with_variations: List of file names for which we need to calculate the variations, all in the end.
        """
        self.file_names = file_names
        self.santa_claus_df = santa_claus_df
        self.other_df = other_df
        self.us = us
        self.data_dir = data_dir
        self.files_with_variations = files_with_variations
        self.data = None
        self._load_data()

    def _load_data(self):
        """
        Load the specified Excel files, assemble the data into a DataFrame with dates as the index, and then calculate the variations of the columns if necessary.
        """

        # Global DataFrame to assemble all the data.
        all_data = []

        # Iteration over each desired variable (file path).
        for file_name in self.file_names:
            file_path = os.path.join(self.data_dir, 'US' if self.us else 'EUR', file_name)
            if os.path.exists(file_path):
                print(f"Chargement de {file_name}...")

                # Load the data from the Excel file.
                df = pd.read_excel(file_path)


                # The first column is assumed to be the date column and is converted to a date format.
                df['Date'] = pd.to_datetime(df.iloc[:, 0])
                df.set_index('Date', inplace=True)

                # Extract the file name without the .xlsx extension.
                column_name = file_name.replace('.xlsx', '')

                # Add the data column to the DataFrame.
                df = df.iloc[:, [1]]  # Select only the second column (values).
                df.rename(columns={df.columns[0]: column_name}, inplace=True)

                # Fill missing values (NaN) with the previous value (forward fill).
                #df.fillna(method='ffill', inplace=True)
                df.ffill(inplace=True)

                # Add this DataFrame to the global DataFrame.
                all_data.append(df)

            else:
                print(f"Le fichier {file_name} n'existe pas dans le répertoire {self.data_dir}.")

        # Combine all the data into a single DataFrame using the dates as the index.
        self.data = pd.concat(all_data, axis=1)

        if 'DGS10' in self.data.columns and 'DGS1' in self.data.columns :
            df_us_spread = self.calculate_term_spread(self.data['DGS10'],self.data['DGS1'])
            self.data = pd.concat([self.data, df_us_spread], axis=1)


    def get_data(self):
        """
        Return the loaded and transformed data as a DataFrame.
        """
        return self.data


    # def calculate_santa_claus_mean(self,santa_claus_df):
    #
    #
    #     mean_rendements = []
    #
    #     #Ajoute une colonne YEAR avec les années correspondante à chaque ligne
    #     santa_claus_df['Year'] = santa_claus_df.index.year
    #
    #     #Ajoute une colonne YEAR avec les années correspondante à chaque ligne
    #     grouped = santa_claus_df.groupby('Year')  # Grouper par année
    #
    #     for year, group in grouped:
    #         # Récupérer les 5 derniers jours de l'année
    #         last_5_days = group.tail(5)
    #
    #         # Récupérer les 2 premiers jours de l'année
    #         first_2_days = group.head(2)
    #         period_7_days = pd.concat([last_5_days, first_2_days])
    #         period_7_days_mean = period_7_days.mean(axis=0)
    #
    #         mean_rendements.append(pd.DataFrame({
    #             **{f'{col}': period_7_days_mean[col] for col in period_7_days_mean.index}
    #         }, index=[f'mean_scar_{year}']))
    #
    #         # Concaténer les résultats pour toutes les années
    #         result_df = pd.concat(mean_rendements, axis=0)
    #
    #     result_df.drop(columns='Year', inplace=True)
    #     return result_df


    # def create_combined_returns_and_macro(self,var,santa_claus_dates,index):
    #
    #     if "Year" in self.santa_claus_df:
    #         self.santa_claus_df.drop(columns="Year", inplace= True)
    #
    #     # Santa Claus et autres rendements
    #     combined_returns = pd.concat([self.santa_claus_df, self.other_df]).sort_index()
    #
    #     # Ajoute les variations des variables macroéconomiques (T1-T3 confondus)
    #     combined_data = combined_returns[[index]].copy()
    #
    #
    #     for column in self.data_t1_t3_changes.columns:
    #         if column in var:
    #             for date in combined_data.index:
    #                 # Si la date appartient à la période de Santa Claus
    #                 if date in santa_claus_dates:
    #                     # Récupérer l'année de la date actuelle
    #                     year = date.year
    #
    #                     # Ajoute la valeur de la variable macro pour l'année correspondante
    #                     if year in self.data_t1_t3_changes.index:
    #                         combined_data.loc[date, column] = self.data_t1_t3_changes.loc[year, column]
    #
    #                     if date.month == 1:
    #                         combined_data.loc[date, column] = self.data_t1_t3_changes.loc[year-1, column]
    #
    #     return combined_data.fillna(0)


    def create_macro_dataframe_for_regress(self,index,var):
        """
        Method that will model the formatted DataFrame to be used in the regression based on the variables the user wishes to use.
        """

        # If MaxDD is requested, remove it from the variables, as the series will be calculated using a rolling window and added later.
        isMaxdd = False
        if "Maxdd" in var :
            isMaxdd = True
            var.remove("Maxdd")

        # Possibility to work on the rolled VIX, a method that proved to be ineffective and therefore absent from the final script.
        isVixRolling = False
        if "VixRoll" in var:
            isVixRolling = True
            VIX_Data = self.data['VIX']
            var.remove("VixRoll")

        # For each requested variable, format the DataFrame for classical regression with the dummy variable (as in the paper).
        if var !=[""] :
            self.data_returns = self.data[var].dropna()
            self.data_returns = self.data_returns.pct_change()
            self.data_returns = self.data_returns.iloc[1:]
            self.data_returns.sort_index(inplace=True)

        if "Year" in self.santa_claus_df:
            self.santa_claus_df.drop(columns="Year", inplace=True)

        # To retrieve the series of returns without date distinction.
        combined_returns = pd.concat([self.santa_claus_df, self.other_df]).sort_index()


        # Add the variations of the variables requested by the user.
        combined_data = combined_returns[[index]].copy()
        if var !=[""]:
            combined_data[var] = self.data_returns[var].reindex(combined_data.index)
            combined_data[var] = combined_data[var].replace([np.inf, -np.inf], np.nan)  # Remplacer infinies par NaN
            combined_data[var] = combined_data[var].fillna(0)  # Remplacer NaN par 0

        # Creation of the Max Drawdown series, calculated on a rolling window of 252 trading days.
        if isMaxdd:
            max_drawdown = self.calculate_max_drawdown(combined_data[index], window=252)
            combined_data['Maxdd'] = max_drawdown
            combined_data = combined_data.dropna() #perte des 20 premières données
            var.append('Maxdd')

        # Creation of the Vix Rolling series, calculated on a rolling window of 252 trading days, but not used in the final script..
        if isVixRolling:
            Vix_Rolling = self.calculate_vix_evolution(VIX_Data, window=252)
            combined_data['VixRoll'] = Vix_Rolling
            combined_data = combined_data.dropna()
            var.append('VixRoll')

        #  Creation of graphs for each series in order to observe the stationarity of the series used for regression.
        #  The plots are stored in the 'Variable_graph' folder within the project.
        for column in combined_data.columns:
            self.plot_and_save_variable_graph(combined_data[column], column, 'Variable_graph')

        return combined_data

    def calculate_max_drawdown(self, df, window=20):
        """
        Method for calculating max drawdown using a sliding window.
        """

        cumulative_returns = (1 + df).cumprod()

        # Initialization of the Max Drawdown series
        max_drawdown = pd.Series(index=df.index, dtype=float)

        # Calculation of drawdown using a sliding window
        for i in range(window, len(df)):
            window_data = cumulative_returns.iloc[i - window:i]
            peak = window_data.max()
            trough = window_data.min()
            drawdown = (trough - peak) / peak
            max_drawdown.iloc[i] = drawdown

        return max_drawdown

    def calculate_vix_evolution(self, df_vix, window=20):
        """
        Method for calculating the rolled VIX
        """

        # Initialization of the VIX evolution series
        vix_evolution = pd.Series(index=df_vix.index, dtype=float)

        # Calculation of the VIX evolution using a sliding window
        for i in range(window, len(df_vix)):
            window_data = df_vix.iloc[i - window:i]
            start_value = window_data.iloc[0]
            end_value = window_data.iloc[-1]
            evolution = (end_value - start_value) / start_value
            vix_evolution.iloc[i] = evolution

        return vix_evolution

    def plot_and_save_variable_graph(self, variable_data, variable_name, output_dir):
        """
        Method to plot and save a file of time series graphs
        """

        # Create the folder if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(variable_data.index, variable_data, label=variable_name, color='b')
        plt.title(f'Évolution de {variable_name} au fil du temps')
        plt.xlabel('Date')
        plt.ylabel(f'{variable_name}')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Save the plot
        output_path = os.path.join(output_dir, f"{variable_name}.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def calculate_term_spread(self,df_10yr, df_1yr):
        """
        Method used for research but not included in the final script:
        calculates the term spread by subtracting the 1-year US rate from the 10-year US rate.
        """

        df_10yr.index = pd.to_datetime(df_10yr.index)
        df_1yr.index = pd.to_datetime(df_1yr.index)

        merged_df = pd.merge(df_10yr, df_1yr, left_index=True, right_index=True, how='inner')
        merged_df['US Term Spread'] = merged_df['DGS10'] - merged_df['DGS1']

        return merged_df[['US Term Spread']]

    def plot_vix_evolution_by_year(self):
        """
        Method used in the research but not implemented in the final script:
        plot a graph for each year showing the evolution of the VIX and calculate the average VIX level for each month of the year, as well as the annual average level.

        """

        vix_file = os.path.join('Data', 'US', 'VIX.xlsx')
        vix_data = pd.read_excel(vix_file, index_col='Obs_Date', parse_dates=True)
        vix_data.sort_index(inplace=True)

        # Create a DataFrame for monthly averages (years in rows, months in columns).
        years = vix_data.index.year.unique()
        monthly_vix_mean = pd.DataFrame(index=years, columns=range(1, 13))

        for year in years:
            year_data = vix_data[vix_data.index.year == year]
            year_data['Month'] = year_data.index.month
            monthly_data = year_data.groupby('Month')['VIX'].mean()
            monthly_vix_mean.loc[year] = monthly_data.reindex(range(1, 13)).fillna(
                0).values

        # Calculate the annual average (average of the rows).
        monthly_vix_mean['mean'] = monthly_vix_mean.mean(axis=1)

        # Plot the data year by year (one graph per year).
        output_dir = os.path.join(os.getcwd(), 'plot_descriptive')
        os.makedirs(output_dir, exist_ok=True)  # Create the folder if it does not exist.

        for year in years:
            plt.figure(figsize=(10, 6))
            year_data = vix_data[vix_data.index.year == year]
            plt.plot(year_data.index, year_data['VIX'])
            plt.title(f'Évolution du VIX en {year}')
            plt.xlabel('Date')
            plt.ylabel('VIX')
            plt.savefig(os.path.join(output_dir, f'vix_{year}.png'))
            plt.close()

        # Export an Excel file with a summary of the average levels per month.
        monthly_vix_mean.to_excel(os.path.join(output_dir, 'vix_monthly_mean.xlsx'))



class MacroAnalysis:
    """
    Class for analyzing the returns during the Santa Claus rally period with market variables requested by the user.
    """

    def __init__(self,combined_data,santa_claus_period,variables):

        self.combined_data = combined_data
        self.santa_claus_period = santa_claus_period
        self.variables = variables
        self.output_dir = "Stationnarity graphs"
        os.makedirs(self.output_dir, exist_ok=True)

    def perform_combined_regression(self, surplus_analysis=False, var_surplus=None, drop_var_surplus=None,
                                onlysurplus=False):
        """
        Method for flexible regression based on user preferences:
        takes the combined dataframe of variables and creates a dummy for the Santa Claus return period.
        The parameters allow for adding an "extra" variable to be applied with the dummy,
        and for deciding whether or not to keep the entire variable throughout the entire duration of the indices.

        :param surplus_analysis: If true, add to the explanatory variable matrix only the variables requested by the user for the Santa Claus days (multiplied by the dummy = 1 on those days).
        :param var_surplus: List of variables we wish to add to the explanatory variable matrix, multiplied by the dummy.
        :param drop_var_surplus: If we wish to remove a variable from the list of explanatory variables, allow adding an extra variable (with the dummy) and removing the entire series, keeping only the series with the dummy.
        :param onlysurplus: If we want to remove the simple dummy representing the excess return during the Santa Claus period.
        """



        # Identify the Santa Claus days in the data
        self.combined_data['Santa_Claus'] = self.combined_data.index.isin(self.santa_claus_period).astype(int)

        # Matrix of explanatory variables
        X = pd.DataFrame(index=self.combined_data.index)

        # Add the macro variables to the Santa Claus dates
        for var in self.variables:
            X['Santa_Claus'] = self.combined_data['Santa_Claus']
            if var != "": X[var] = self.combined_data[var] # Add the time series of the variables to the explanatory variable matrix
            if surplus_analysis and var in var_surplus: X[f'Santa_clauss_{var}'] = X['Santa_Claus'] * X[var] # If surplus analysis was requested, add a new variable corresponding to the requested variable multiplied by the Santa Claus day dummy.
            if drop_var_surplus and var in var_surplus:  X.drop(var, axis=1, inplace=True) # If we wish to remove a variable from the list of explanatory variables

        if onlysurplus: X.drop('Santa_Claus', axis=1, inplace=True) #If we want to remove the simple dummy representing the excess return during the Santa Claus period.

        # The dependent variable
        y = self.combined_data.iloc[:, 0]

        # Replace NaN values with 0 in the explanatory variables to avoid errors in the regression.Normally, there should be none since we used the fill-forward method in the MacroDataFormat class.
        X.fillna(0, inplace=True)

        # Add a constant to the regression to capture the intercept term.
        X = sm.add_constant(X, has_constant='add')
        model = sm.OLS(y, X)
        results_base = model.fit()
        residues_base = results_base.resid #Retrieve the residuals.

        # Apply the heteroscedasticity and autocorrelation tests to determine if the regression needs to be rerun with HAC (Heteroskedasticity and Autocorrelation Consistent) correction.

        # Ljung-Box test for autocorrelation
        lb_test = acorr_ljungbox(residues_base, lags=[1], return_df=True)
        ljung_box_stat = lb_test["lb_stat"].iloc[0]
        ljung_box_pvalue = lb_test["lb_pvalue"].iloc[0]

        # ARCH LM test for heteroscedasticity
        arch_lm_stat, arch_lm_pvalue, _, _ = het_arch(residues_base)

        # Display the results
        test_result_base = {'Ljung-Box Stat': round(ljung_box_stat, 3),
                       'Ljung-Box p-value': round(ljung_box_pvalue, 3),
                       'ARCH LM Stat': round(arch_lm_stat, 3),
                       'ARCH LM p-value': round(arch_lm_pvalue, 3)}

        test_result_base = pd.DataFrame(test_result_base, index=['Test'])
        print(test_result_base)

        # Rerun the regression with the necessary corrections if needed.
        if ljung_box_pvalue < 0.05 and arch_lm_pvalue < 0.05:
            print("--------------- Utilisation de la correction HAC------------------")
            model = sm.OLS(y, X)
            results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 1})
        elif ljung_box_pvalue > 0.05 and arch_lm_pvalue < 0.05:
            print("--------------- Utilisation de la correction White ------------------")
            model = sm.OLS(y, X)
            results = model.fit(cov_type='HC0')
        elif ljung_box_pvalue < 0.05 and arch_lm_pvalue > 0.05:
            print("--------------- Utilisation de la correction de HAC------------------")
            model = sm.OLS(y, X)
            results = model.fit(cov_type='HAC')
        else:
            print("--------------- Aucunne correction appliquée------------------")
            model = sm.OLS(y, X)
            results = model.fit()

        return results.summary()

    def plot_and_check_stationarity(self,df_column):
        """
        Plot the time series of a column in a DataFrame and perform a stationarity test (ADF).
        """

        column_name = df_column.name

        # Plot the time series.
        plt.figure(figsize=(10, 6))
        plt.plot(df_column, label="Time series")
        plt.title(f"Quarterly variations of  {column_name}")
        plt.xlabel("Days")
        plt.ylabel("Values")
        plt.legend()
        plt.grid(True)

        plt.xticks(rotation=45)

        # Saving the cumulative chart
        output_path = os.path.join(self.output_dir, f"{column_name}_Stationarity_graphs.png")
        plt.savefig(output_path)
        plt.close()

        # Perform the stationarity test (Augmented Dickey-Fuller).
        result = adfuller(df_column)

        print("---------------------------------------------------------------------")
        # Display the results of the ADF test
        print(f"Résultat du test ADF pour {column_name} :")
        print(f"Statistique ADF: {result[0]}")
        print(f"p-value: {result[1]}")
        print(f"Nombre de retard(s): {result[2]}")
        print(f"Nombre d'observations utilisées pour le calcul du test: {result[3]}")
        print(f"Critères de valeurs (1%, 5%, 10%): {result[4]}")

        # Interpretation of the p-value
        if result[1] <= 0.05:
            print("La série est stationnaire (p-value <= 0.05).")
        else:
            print("La série n'est pas stationnaire (p-value > 0.05).")

        print('Test de stationnarité terminé')
        print("---------------------------------------------------------------------")




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

        #US --------------------------------------------------------------------------------------------------------------------------------------------------------
        # file_names = ['T10Y2Y.xlsx','EURUSD.xlsx','DTB3.xlsx','DGS10.xlsx','DGS1.xlsx','BAA10Y.xlsx','VIX.xlsx']
        # files_with_variations =['T10Y2Y.xlsx','EURUSD.xlsx','DTB3.xlsx','DGS10.xlsx','DGS1.xlsx','BAA10Y.xlsx','VIX.xlsx']

        file_names = ['EURUSD.xlsx','VIX.xlsx']
        files_with_variations =['EURUSD.xlsx','VIX.xlsx']

        # Créer l'instance de la classe avec un paramètre US à True
        macro_data = MacroDataFormat(santa_claus_df,other_df,file_names, us=True, data_dir='data', files_with_variations=files_with_variations)

        # Dataframe contenant toutes les données chargées
        all_data_macro = macro_data.get_data()

        #Récupère les jours du santa clauss
        santa_claus_period = santa_claus_df.index

        variables = ['']
        combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        regression_results = macro_analysis.perform_combined_regression()
        print(regression_results)


        variables = ['VIX']
        combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        regression_results  = macro_analysis.perform_combined_regression()
        print(regression_results)


        variables = ['Maxdd']
        combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        regression_results  = macro_analysis.perform_combined_regression()
        print(regression_results)


        variables = ['EURUSD']
        combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        regression_results  = macro_analysis.perform_combined_regression()
        print(regression_results)

        # Regression initial
        variables = ['Maxdd','VIX','EURUSD']
        combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        regression_results  = macro_analysis.perform_combined_regression()
        print(regression_results)




        variables = ["Maxdd"]
        combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        regression_results  = macro_analysis.perform_combined_regression(surplus_analysis=True, var_surplus=['Maxdd'],
                                                                        drop_var_surplus=True, onlysurplus=True)
        print(regression_results)


        # Regression avec le VIX Roll
        variables = ["VIX"]
        combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        regression_results  = macro_analysis.perform_combined_regression(surplus_analysis=True, var_surplus=['VIX'],
                                                                        drop_var_surplus=True, onlysurplus=True)
        print(regression_results)


        # Regression avec le VIX Roll
        variables = ["EURUSD"]
        combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        regression_results  = macro_analysis.perform_combined_regression(surplus_analysis=True, var_surplus=['EURUSD'],
                                                                        drop_var_surplus=True, onlysurplus=True)
        print(regression_results)


        # Regression initial
        variables = ['Maxdd', 'VIX', 'EURUSD']
        combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        regression_results  = macro_analysis.perform_combined_regression(surplus_analysis=True, var_surplus=variables,
                                                                        drop_var_surplus=True, onlysurplus=True)
        print(regression_results)




#
# #
# # # EUROPE
# if __name__ == "__main__":
#     #tickers = ['^FCHI','^STOXX','^GDAXI','^FTSE'] #Europe
#     tickers = ['^FCHI', '^STOXX']
#     start_date = '1999-12-31'
#     end_date = '2022-01-01'
#
#     base = DataCompute(tickers, start_date, end_date)
#     df_prices = base.df_prices
#     df_returns = base.df_returns
#     data_dict = base.data_dict
#
#     stats = statistics(None, None, data_dict)
#     stats_descriptives = stats.desc_stats
#     print("Statistiques descriptives des indices")
#     print(stats_descriptives)
#     print("--------------------------------------")
#
#     for ticker in tickers:
#
#         print(f"Analyse pour le ticker : {ticker}")
#         df_returns = stats.recup_df(data_dict, ticker, 'df_returns')
#         df_prices = stats.recup_df(data_dict, ticker, 'df_prices')
#
#         bucket = SantaClausBucket(df_returns, df_prices)
#         santa_claus_df = bucket.santa_claus_df
#         other_df = bucket.other_df
#         days = bucket.santa_claus_days  # pour verif
#
#         #EUROPE ------------------------------------------------------------------------------------------------------------------------------------
#         file_names = ['3MGB.xlsx', 'EURUSD.xlsx', 'T10Y1Y.xlsx', 'T10Y2Y.xlsx','VIX.xlsx']
#         files_with_variations = ['3MGB.xlsx', 'EURUSD.xlsx', 'T10Y1Y.xlsx', 'T10Y2Y.xlsx','VIX.xlsx']
#
#         # Créer l'instance de la classe avec un paramètre US à True
#         macro_data = MacroDataFormat(santa_claus_df, other_df, file_names, us=False, data_dir='data',
#                                      files_with_variations=files_with_variations)
#
#         # Dataframe contenant toutes les données chargées
#         all_data_macro = macro_data.get_data()
#
#         # Récupère les jours du santa clauss
#         santa_claus_period = santa_claus_df.index
#
#         #Regression avec le taux de change EURUSD
#         variables = ['EURUSD']
#         combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
#         macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
#         regression_results = macro_analysis.perform_combined_regression()
#         print(regression_results)

        # # Modélisation du dataframe avec les rendements daily et les variations macro aux jours du santa clauss
        # variables = ['T10Y2Y']
        # combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        # macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        # # #Régréssion avec les rendements daily et les variables macroéconomiques
        # regression_results = macro_analysis.perform_combined_regression()
        # print(regression_results)
        #
        # variables = ['3MGB']
        # combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        # macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        # regression_results = macro_analysis.perform_combined_regression()
        # print(regression_results)
        #
        # variables = ['EURUSD', '3MGB']
        # combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        # macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        # regression_results = macro_analysis.perform_combined_regression()
        # print(regression_results)
        #
        # variables = ['T10Y1Y']
        # combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        # macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        # regression_results = macro_analysis.perform_combined_regression()
        # print(regression_results)
        #
        # variables = ['T10Y1Y', 'EURUSD', '3MGB']
        # combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        # macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        # regression_results = macro_analysis.perform_combined_regression()
        # print(regression_results)
        #
        # variables = ['T10Y2Y', '3MGB']
        # combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        # macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        # regression_results = macro_analysis.perform_combined_regression()
        #
        # variables = ['VIX', 'EURUSD']
        # combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        # macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        # regression_results = macro_analysis.perform_combined_regression()
        # print(regression_results)

        #   Regression avec le VIX et le Vix Roll en variable de surplus du rendement santa claus
        # variables = ['VIX', 'VixRoll']
        # combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        # macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        # regression_results = macro_analysis.perform_combined_regression(surplus_analysis=True, var_surplus='VixRoll',
        #                                                                 drop_var_surplus=True)
        # print(regression_results)

        # Regression avec le VIX sans surplus du rendement santa claus
        # variables = ['VIX']
        # combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        # macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        # regression_results = macro_analysis.perform_combined_regression(surplus_analysis=False, var_surplus='VIX')
        # print(regression_results)

        # # Regression avec le MAXDD en surplus uniquement du rendement santaclauss
        # # variables = ["Maxdd"]
        # # combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        # # macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        # # regression_results = macro_analysis.perform_combined_regression(surplus_analysis=True, var_surplus='Maxdd',
        # #                                                                 drop_var_surplus=True)
        # print(regression_results)
        #
        # # # Regression avce le MAXDD
        # # variables = ["Maxdd"]
        # # combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        # # macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        # # regression_results = macro_analysis.perform_combined_regression(surplus_analysis=False, var_surplus='Maxdd',
        # #                                                                 drop_var_surplus=False)
        # print(regression_results)
        #
        # # Regression avce le MAXDD
        # variables = ["Maxdd"]
        # combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        # macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        # regression_results = macro_analysis.perform_combined_regression()
        # print(regression_results)
        #
        # # # Regression avec le VIX Roll
        # # variables = ["VixRoll"]
        # # combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        # # macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        # # regression_results = macro_analysis.perform_combined_regression()
        # # print(regression_results)
        #
        # # Regression initial
        # variables = ['Maxdd','VIX','EURUSD']
        # combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        # macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        # regression_results = macro_analysis.perform_combined_regression()
        # print(regression_results)
        #
        # # Regression initial
        # variables = [""]
        # combined_data_SP500 = macro_data.create_macro_dataframe_for_regress(ticker, variables)
        # macro_analysis = MacroAnalysis(combined_data_SP500, santa_claus_period, variables)
        # regression_results = macro_analysis.perform_combined_regression()
        # print(regression_results)


#
#
#
#         #Exemple de toutes les variables utilisables
#         macro_vars = ['Advance Retail Sales US', 'CORE CPI US', 'CPI ALL items US', 'EFFR',
#                       'Industrial Production US', 'PCE_Etat_Unis', 'Personal Saving Rate US', 'T10Y2Y',
#                       'Unemployment Rate US', 'University of Michigan Consumer Sentiment', 'M2REAL US']
#
#         macro_daily_vars =['T10Y2Y','EURUSD','DTB3']
#
#
#
#
#
