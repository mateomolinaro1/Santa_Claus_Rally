
import pandas as pd
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 20)
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from statsmodels.tsa.stattools import adfuller

from DataBase import DataCompute
from SantaClausAnalysis import statistics,SantaClausBucket



class EmpiricalAnalysis:
    """
    Empirical study class:
    allows for empirical analysis on multiple variables to try to determine a relationship between the variables and the excess return of the Santa Claus rally.
    """

    def __init__(self, santa_claus_df, other_df,df_prices):
        self.santa_claus_df = santa_claus_df
        self.other_df = other_df
        self.output_dir = "graphs"
        os.makedirs(self.output_dir, exist_ok=True)
        self.df_prices = df_prices

    def analyze_annual_performance(self):
        """
        A method that will calculate the annual performance for each year of the index as well as the performance
        during the Santa Claus rally period, to display a summary histogram and create a DataFrame classifying the years by Santa Claus return.
        """

        results ={}
        for column in self.santa_claus_df.columns:
            annual_performance = []
            column_data = self.other_df[column].dropna()
            years = column_data.index.year.unique()
            years = list(range(years.min(), years.max() + 1))

            for year in years:
                # Current year's date.
                year_start = pd.Timestamp(f'{year}-01-01')
                year_end = pd.Timestamp(f'{year}-12-31')

                # Total annual return.
                yearly_data = self.other_df[(self.other_df.index >= year_start) & (self.other_df.index <= year_end)]
                annual_return = (1 + yearly_data[column]).prod() - 1

                # Santa Claus return.
                santa_data = self.santa_claus_df[self.santa_claus_df.index.year.isin([year, year + 1])]
                santa_data = santa_data[(santa_data.index >= pd.Timestamp(f'{year}-02-01')) & (santa_data.index <= pd.Timestamp(f'{year+1}-02-01'))]  # Take the days corresponding to the current year starting from February and the following year before February.
                santa_return = (1 + santa_data[column]).prod() - 1

                # Add the calculated data to a list.
                annual_performance.append((year, annual_return, santa_return))

            # Sort the results by Santa Claus return (in descending order
            annual_performance_df = pd.DataFrame(annual_performance, columns=["Year", "Annual return", "Santa Claus return"])
            annual_performance_df = annual_performance_df.sort_values(by="Santa Claus return", ascending=False)

            results[column] = annual_performance_df

            # Display the results as a histogram
            years, annual_returns, santa_returns = zip(*annual_performance)
            plt.figure(figsize=(12, 6))
            bar_width = 0.8

            for i, year in enumerate(years):
                annual_ret = annual_returns[i] * 100
                santa_ret = santa_returns[i] * 100

                # Bar positioning
                plt.bar(year, annual_ret, color='blue', width=bar_width, label='Annual return' if i == 0 else "")
                if annual_ret >= 0 and santa_ret >= 0:
                    plt.bar(year, santa_ret, bottom=annual_ret, color='orange', width=bar_width,
                            label='Santa Claus Days' if i == 0 else "")

                elif annual_ret < 0 and santa_ret >= 0:
                    plt.bar(year, santa_ret, bottom=0, color='orange', width=bar_width, hatch='//',
                            label='Santa Claus Days' if i == 0 else "")

                elif annual_ret < 0 and santa_ret < 0:
                    plt.bar(year, santa_ret, bottom=annual_ret, color='orange', width=bar_width,
                            label='Santa Claus Days (Négatif)' if i == 0 else "")

                elif annual_ret >= 0 and santa_ret < 0:
                    plt.bar(year, santa_ret, bottom=0, color='orange', width=bar_width, hatch='//',
                            label='Santa Claus Days (Négatif)' if i == 0 else "")

                # Adding annotations.
                if annual_ret != 0:
                    plt.text(year, annual_ret+2, f"{annual_ret:.1f}%", ha='center', va='center', color='black',
                             fontsize=9, fontweight='bold')

                if santa_ret != 0:
                    plt.text(year, annual_ret-1, f"{santa_ret:.1f}%", ha='center',  va='bottom' if santa_ret < 0 else 'top', color='Orange',
                             fontsize=9, fontweight='bold')

            # Adjusting the axes
            plt.xticks(years)
            plt.xlabel("Year")
            plt.ylabel("Return (%)")
            plt.title(f"Annual Return and Santa Claus Days Return - {column}")
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Saving the chart in the "graphs" folder of the project
            output_path = os.path.join(self.output_dir, f"graph_empirique_annual_performance_{column}.png")
            plt.savefig(output_path)
            plt.close()

        return results


    def analyze_risk_vs_return(self, risk_function):
        """
        Method useful for research but not used in the final script :
        A method that will compare the excess return of the average return during the Santa Claus rally to the annual return,
        with another variable of the user's choice based on the available functions.
        """

        results = []

        # Colors to differentiate the indices.
        colors = plt.cm.tab10.colors
        color_mapping = {}

        for idx, column in enumerate(self.santa_claus_df.columns):
            color_mapping[column] = colors[idx % len(colors)]

            # Results for the individual chart.
            individual_results = []

            for year in self.other_df.index.year.unique():
                # Return and risk for Santa Claus.
                santa_data = self.santa_claus_df[self.santa_claus_df.index.year.isin([year, year + 1])]
                santa_data = santa_data[(santa_data.index >= pd.Timestamp(f'{year}-02-01')) & (
                            santa_data.index <= pd.Timestamp(f'{year + 1}-02-01'))]
                mean_santa_return = santa_data[column].mean()

                # Return and risk for the full year.
                year_start = pd.Timestamp(f'{year}-01-01')
                year_end = pd.Timestamp(f'{year}-12-31')
                yearly_data = self.other_df[(self.other_df.index >= year_start) & (self.other_df.index <= year_end)]
                annual_return = (1 + yearly_data[column]).prod() - 1
                mean_annual_return = yearly_data[column].mean()

                annuel_excess_returns = mean_santa_return -mean_annual_return
                annual_risk, risk_name, sens = risk_function(yearly_data[column])

                # Store the results.
                if annual_risk !=0:
                    results.append((year, column, annuel_excess_returns, annual_return, annual_risk))
                    individual_results.append((annuel_excess_returns, annual_return, annual_risk))

            # Individual chart for the index
            santa_returns, annual_return, annual_risks = zip(*individual_results)
            df_annual_risk = pd.DataFrame(annual_risks, columns=[risk_name])

            #Stationarity check -> not stationary, no construction.
            self.plot_and_check_stationarity(df_annual_risk[risk_name])

            plt.figure(figsize=(12, 6))
            plt.scatter(santa_returns, [abs(r) if sens == 'inverse' else r for r in annual_risks],
                        color=color_mapping[column],
                        label=f"{column} ")

            try :
                # Adding a regression line, but it has little econometric meaning because the data is not stationary.
                santa_returns_np = np.array(santa_returns)
                annual_risks_np = np.array([abs(r) if sens == 'inverse' else r for r in annual_risks])
                if len(santa_returns_np) > 1:
                    regression_result = linregress(santa_returns_np, annual_risks_np)
                    slope = regression_result.slope
                    intercept = regression_result.intercept
                    r_value = regression_result.rvalue
                    p_value = regression_result.pvalue
                    std_err = regression_result.stderr

                    reg_line = slope * santa_returns_np + intercept
                    plt.plot(santa_returns_np, reg_line, color='red', linestyle='--', label='Regression Line')

            except:
                pass

            plt.axhline(0, color='black', linewidth=0.8,
                        linestyle='--')
            plt.axvline(0, color='black', linewidth=0.8,
                        linestyle='--')

            plt.title(f"{risk_name} vs Santa Claus Excess returns - {column}")
            plt.xlabel("Santa Claus excess returns")
            plt.ylabel(f"{risk_name}{' (-)' if sens == 'inverse' else ''}")
            plt.legend()
            plt.grid()

            # Saving the chart.
            output_path = os.path.join(self.output_dir, f"graph_empirique_{risk_name}_vs_return_{column}.png")
            plt.savefig(output_path)
            plt.close()

        df_result = pd.DataFrame({
            "Santa_Returns": santa_returns,
            "Annual_Risk": annual_risks
        })

        return  df_result

    def plot_combined_risk_vs_return(self, dict_result, risk_name, sens):
        """
        Method to display the histogram of all indices together, Santa Claus return excess compared to a variable per year for all indices at the same time.
        """

        plt.figure(figsize=(12, 6))
        colors = plt.cm.tab10.colors

        all_santa_returns = []
        all_annual_risks = []

        for i, (ticker, data) in enumerate(dict_result.items()):
            santa_returns = data.get("Santa_Returns", [])
            annual_risks = data.get("Annual_Risk", [])

            risks = [abs(risk) if sens == "inverse" else risk for risk in annual_risks]

            plt.scatter(
                santa_returns,
                risks,
                color=colors[i % len(colors)],
                label=ticker
            )

        plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
        plt.axvline(0, color='black', linestyle='--', linewidth=0.8)

        plt.title(f"{risk_name} vs Santa Claus Excess Returns (Combined Tickers)")
        plt.xlabel("Santa Claus Excess Returns")
        plt.ylabel(f"{risk_name}{' (-)' if sens == 'inverse' else ''}")
        plt.legend(title="Tickers", loc="best")
        plt.grid()

        output_path = os.path.join(self.output_dir, f"combined_{risk_name}_vs_return_with_regression.png")
        plt.savefig(output_path)
        plt.close()

    def plot_and_check_stationarity(self, df_column):
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


# Function to calculate max_dd
def max_drawdown(df):
    cumulative = (1 + df).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    result = drawdown.min()
    return result,"max_drawdown","inverse"

# Function to calculate std
def standard_deviation(df):
    result = df.std()
    return result, "Standard Deviation", "normal"

# Function to calculate annual returns
def annual_returns(df):
    result = (1 + df).prod() - 1
    return result, "Annual returns", "normal"

# Function to calculate cumulative exchange returns
def Cumulative_Exchange_Return(df):
    file_path = os.path.join("Data", "US", "EURUSD.xlsx")
    exchange_data = pd.read_excel(file_path, index_col=0, parse_dates=True)
    exchange_data = exchange_data.sort_index().dropna()

    common_dates = df.index.intersection(exchange_data.index)
    filtered_data = exchange_data.loc[common_dates]

    daily_returns = filtered_data.pct_change(fill_method=None).dropna()

    cumulative_return = (1 + daily_returns).prod() - 1

    return cumulative_return.iloc[0], "EURUSD Annual Returns", "normal"

# Function to calculate exchange rate volatility
def Exchange_Rate_Volatility(df):
    file_path = os.path.join("Data", "US", "EURUSD.xlsx")
    exchange_data = pd.read_excel(file_path, index_col=0, parse_dates=True)
    exchange_data = exchange_data.sort_index()

    common_dates = df.index.intersection(exchange_data.index)
    filtered_data = exchange_data.loc[common_dates]

    daily_returns = filtered_data.pct_change(fill_method=None).dropna()

    volatility = daily_returns.std()

    return volatility.iloc[0], "EURUSD Daily Volatility", "normal"

# Function to calculate mVIX mean
def VIX_mean(df):
    file_path = os.path.join("Data", "US", "VIX.xlsx")
    exchange_data = pd.read_excel(file_path, index_col=0, parse_dates=True)
    exchange_data = exchange_data.sort_index()

    common_dates = df.index.intersection(exchange_data.index)
    filtered_data = exchange_data.loc[common_dates]

    if not filtered_data.empty :
       mean_VIX = filtered_data.mean().values[0]
    else:
       mean_VIX =0

    return mean_VIX, "Average Annual VIX", "normal"



# If needed, launch the generation of empirical charts.
if __name__ == "__main__":

    tickers = ['^GSPC', '^IXIC','^FCHI','^STOXX']

    start_date = '1999-12-31'
    end_date = '2025-01-06'

    base = DataCompute(tickers, start_date, end_date)
    df_prices = base.df_prices
    df_returns = base.df_returns
    data_dict = base.data_dict

    stats = statistics(None, None, data_dict)
    stats_descriptives = stats.desc_stats
    print("Descriptive statistics of the indices")
    print(stats_descriptives)
    print("--------------------------------------")

    dict_vix_results = {}

    for ticker in tickers:
        df_returns = stats.recup_df(data_dict, ticker, 'df_returns')
        df_prices = stats.recup_df(data_dict, ticker, 'df_prices')

        bucket = SantaClausBucket(df_returns, df_prices)
        santa_claus_df = bucket.santa_claus_df
        other_df = bucket.other_df
        days = bucket.santa_claus_days

        # Instantiation of the class
        analysis = EmpiricalAnalysis(santa_claus_df, other_df,df_prices)

        # Create the annual performance analysis charts and the Santa Claus performance chart, and return the results table.
        table_result = analysis.analyze_annual_performance()




        #The sequence was not used in the final script but was used for the research. --------------------------------------
        #file path for  EURUSD data
        file_path = os.path.join("Data", "US", "EURUSD.xlsx")

        #Extract data and compute dataframe
        exchange_data = pd.read_excel(file_path, index_col=0, parse_dates=True)
        exchange_data = exchange_data.sort_index().dropna()

        # Creation of charts and storage of empirical analysis results for the VIX as a variable compared to the excess return of Santa Claus.
        dict_vix_results[ticker] = analysis.analyze_risk_vs_return(VIX_mean)
        analysis.analyze_risk_vs_return(VIX_mean)

        # Creation of independent comparison charts for each ticker.
        analysis.analyze_risk_vs_return(Cumulative_Exchange_Return)
        analysis.analyze_risk_vs_return(Exchange_Rate_Volatility)
        analysis.analyze_risk_vs_return(max_drawdown)
        analysis.analyze_risk_vs_return(standard_deviation)
        analysis.analyze_risk_vs_return(annual_returns)

    # Creation of the comparison chart of excess return to the VIX for all indices combined.
    if dict_vix_results:
        analysis.plot_combined_risk_vs_return(dict_vix_results, "VIX_mean", "normal")