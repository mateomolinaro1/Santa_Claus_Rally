import math

import pandas as pd
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from scipy.stats import f
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm

from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.ar_model import AutoReg




class SantaClausBucket():
    """
    A class that will handle the creation of buckets, one for Santa Claus' day data, and one for other days.
    """
    def __init__(self,df_returns,df_prices,long_period: bool = True):
        #Classic Analysis
        self.santa_claus_days = self.search_santa_claus_days(df_returns,df_prices,long_period=long_period)
        self.santa_claus_df = self.compute_santa_claus_df(df_returns)
        self.other_df = self.compute_other_df(df_returns)

        # December Analysis
        self.december_days = self.search_santa_claus_days(df_returns,df_prices, 21, 0)
        self.december_df = self.compute_santa_claus_df(df_returns, santa_claus_or_december="december")
        self.december_other_df = self.compute_other_df(df_returns, santa_claus_or_december="december")

    def compute_santa_claus_df(self,df,santa_claus_or_december:str="santa_claus"):
        """
        A method that retrieves the data for Santa Claus' days based on the date passed as arguments.
        """

        # Checking inputs
        if santa_claus_or_december not in ["santa_claus", "december"]:
            raise ValueError("Wrong name of period entered. Supported: 'santa_claus' or 'december'.")

        if santa_claus_or_december == "santa_claus":
            santa_claus_df = df[df.index.isin(self.santa_claus_days)]
            santa_claus_df.index = pd.to_datetime(santa_claus_df.index)
        else:
            santa_claus_df = df[df.index.isin(self.december_days)]
            santa_claus_df.index = pd.to_datetime(santa_claus_df.index)

        return santa_claus_df

    def compute_other_df(self,df,santa_claus_or_december:str="santa_claus"):
        """
        A method that retrieves the data for other days based on the date passed as arguments.
        """
        # Checking inputs
        if santa_claus_or_december not in ["santa_claus", "december"]:
            raise ValueError("Wrong name of period entered. Supported: 'santa_claus' or 'december'.")

        if santa_claus_or_december == "santa_claus":

            other_df = df[~df.index.isin(self.santa_claus_days)]
            other_df.index = pd.to_datetime(other_df.index)
        else :
            other_df = df[~df.index.isin(self.december_days)]
            other_df.index = pd.to_datetime(other_df.index)


        return  other_df


    def search_santa_claus_days(self,df_returns,df_prices,n_december:int=5, n_january:int=2,long_period: bool = True):
        """
        A method that retrieves the days of the Santa Claus Rally for all years.
        """
        santa_claus_days = []
        df_prices.index = pd.to_datetime(df_prices.index)
        years = df_prices.index.year.unique()
        years = list(range(years.min(), years.max() + 1))

        for year in years:
            if year == years[0]:
                if long_period:
                    first_days_of_next_year = self.get_first_weekdays(year - 1, 2,df_returns)
                    santa_claus_days.extend(first_days_of_next_year)

            last_days_of_year = self.get_last_weekdays(year, n_december,df_returns) # A method that retrieves the last n days of the year.

            santa_claus_days.extend(last_days_of_year)

            if long_period:
                first_days_of_next_year = self.get_first_weekdays(year, n_january,df_returns)
                santa_claus_days.extend(first_days_of_next_year)

        return tuple(santa_claus_days)



    def is_weekday(self, date):
        """
        A method that checks if the day is a weekday (Monday = 0, Friday = 4).
        """

        if date.weekday() < 5:
            return True
        else:
            return False

    def get_last_weekdays(self,year, n,df):
        """
        A method that retrieves the last n days of the year.
        """
        days = []
        date = pd.Timestamp(f'{year}-12-31')
        count =0

        # Retrieve the business days, starting from December 31st.
        while len(days) < n:
            if self.is_weekday(date) and date.strftime('%Y-%m-%d')  in df.index:
                days.append(date.strftime('%Y-%m-%d'))
            date -= pd.Timedelta(days=1)
            count += 1
            if count == 150: break

        return days[::-1]  # Return the days in the correct order.


    def get_first_weekdays(self,year, n,df):
        """
        A method that retrieves the first n days of the year.
        """
        days = []
        date = pd.Timestamp(f'{year + 1}-01-02')
        count =0
        # Retrieve the business days starting from January 1st.
        if pd.Timestamp(df.index[0]) <= pd.Timestamp(f'{year + 1}-01-06'):
            while len(days) < n:
                if self.is_weekday(date) and date.strftime('%Y-%m-%d')  in df.index:
                    days.append(date.strftime('%Y-%m-%d'))
                date += pd.Timedelta(days=1)
                count +=1
                if count == 150: break

        return days



class statistics():

    def __init__(self,santa_claus_df,other_df,data_dict=None,december_df=None, december_other_df=None,extratest:bool=False,bootstrap:bool = False):
        """
        A class that contains all the methods for calculating descriptive statistics and econometric analysis.
        """

        if december_df is None :
            # Calling the method for calculating descriptive statistics with formatting, one for display replication and another for reusing the data.
            self.desc_stats, self.desc_stats_numeric = self.compute_descriptive_statistics(santa_claus_df,other_df,data_dict)

            if not data_dict :
                # Calling the statistical test method with formatting, one for display replication and another for reusing the data.
                self.test_stats, self.test_stats_numeric = self.compute_Santa_Claus_test(santa_claus_df,other_df)

                # Calling the OLS regression method with formatting, one for display replication and another for reusing the data.
                self.OLS_stats,self.OLS_stats_numeric = self.OLS_regression(santa_claus_df,other_df)


        if extratest:

            if bootstrap:

                #Calling the regression methods with the bootstrapping method, with formatting, one for display replication and another for reusing the data.
                self.bootstrap_stats,self.bootstrap_stats_numeric = self.bootstrap(santa_claus_df, other_df, n_iterations=1000, alpha=0.05)


            # Calling the regression methods with or HAC correction, with formatting, one for display replication and another for reusing the data.
            self.OLS_variants_stats,self.OLS_variants_numeric = self.OLS_variants(santa_claus_df, other_df)




        # December Analysis, repeat the process with december_df instead santa_claus_df
        if december_df is not None:
            self.dec_desc_stats,  self.dec_desc_stats_numeric = self.compute_descriptive_statistics(santa_claus_df = december_df, other_df = december_other_df)
            self.dec_test_stats,self.dec_test_stats_numeric = self.compute_Santa_Claus_test(santa_claus_df = december_df, other_df = december_other_df)
            self.dec_OLS_stats, self.dec_OLS_stats_numeric = self.OLS_regression(santa_claus_df = december_df, other_df = december_other_df)


            if extratest:

                if bootstrap:
                    self.dec_bootstrap_stats, self.dec_bootstrap_stats_numeric = self.bootstrap(december_df, december_other_df, n_iterations=1000, alpha=0.05)

                self.dec_OLS_variants_stats,self.dec_OLS_variants_stats_numeric  = self.OLS_variants(december_df, december_other_df)



    def recup_df(self,data_dict,ticker,df_name):
        """
        Method to retrieve a DataFrame from the dictionary containing data for all the tickers.
        """
        df = data_dict[ticker].get(df_name)
        return df



    def split_df(self,data_dict,tickers,df_name):
        """
        Method of retrieving dataframes from a dictionary to merge them.
        """
        df_list = []
        for ticker in tickers:
            df = data_dict[ticker].get(df_name)
            if df is not None:
                df = df.rename(columns={df.columns[0]: ticker})
                df_list.append(df)

        df_global = pd.concat(df_list, axis=1)
        return df_global


    def compute_stats(self,df):
        """
        A method that handles the function calls calculating the stats to group them in a dictionary.
        """
        stats = {
            "mean": self.compute_daily_average(df),
            "std": self.compute_standard_deviation(df),
            "min": self.compute_min(df),
            "max": self.compute_max(df),
            "skewness": self.compute_skewness(df),
            "kurtosis": self.compute_kurtosis(df),
            "range": self.compute_range(df),
            "observations": self.nb_observations(df)
        }

        return stats

    def compute_statistics_table(self,stats):
        """
        A method that models the display of the statistics table to replicate the paper.
        """
        formatted_stats = {}

        for key, value in stats.items():
            if key != "observations" and key != "skewness" and key != "kurtosis":
                if isinstance(value, pd.Series):
                    formatted_stats[key] = value.apply(lambda x: f"{x * 100:.4f}%")
                else:
                    formatted_stats[key] = f"{value * 100:.4f}%"
            else:
                formatted_stats[key] = value

        return formatted_stats


    def compute_descriptive_statistics(self,santa_claus_df=None,other_df=None,data_dict=None,numeric:bool=False):
        """
        Method for calculating descriptive statistics of DataFrames passed as arguments.
        """

        # Calculation for each ticker if requested.
        if data_dict is not None:
            formatted_stats = {}
            for ticker in data_dict:
                df = self.recup_df(data_dict,ticker,"df_returns")
                stats = self.compute_stats(df)
                formatted_stats[ticker] = self.compute_statistics_table(stats)

            # formatted_stats in DataFrame
            formatted_stats_df = pd.DataFrame(formatted_stats).T
            formatted_stats_df.index = data_dict.keys()

        # Calculation for the DataFrames if requested.
        elif other_df is not None and santa_claus_df is not None :
            df = pd.concat([santa_claus_df, other_df], axis=1)
            santa_claus_columns = [f"{col} - Santa Claus Rally Days" for col in santa_claus_df.columns]
            other_columns = [f"{col} - Remaining Days" for col in other_df.columns]
            df.columns = santa_claus_columns + other_columns
            stats = self.compute_stats(df)

            formatted_stats = self.compute_statistics_table(stats)

        elif santa_claus_df is not None and other_df is None:
            df = santa_claus_df
            stats = self.compute_stats(df)
            formatted_stats = self.compute_statistics_table(stats)


        return pd.DataFrame(formatted_stats),pd.DataFrame(stats)



    def compute_Santa_Claus_test(self, santa_claus_df, other_df,numeric:bool=False):
        """
        A method that will perform the statistical tests presented in the paper and manage the display to be as faithful as possible to the paper.
        """
        if other_df is None: return "None","None"
        results = {}

        # To be able to use it on a DataFrame containing all the indices (one index per DataFrame in the final script).
        for column in santa_claus_df.columns:
            santa_col = santa_claus_df[column].dropna()
            other_col = other_df[column].dropna()

            # Calling the calculation methods.
            mean_scr = self.compute_daily_average(santa_col)
            mean_other = self.compute_daily_average(other_col)
            obs_scr = self.nb_observations(santa_col)
            obs_other = self.nb_observations(other_col)

            # Calling the test methods.
            t_stat, t_p_value, df = self.T_test(santa_col, other_col)
            u_stat, u_p_value, z_value = self.Mann_Whitney_test(santa_col, other_col)

            results[column] = {
                "Santa Claus Rally Days - Mean": mean_scr,
                "Santa Claus Rally Days - Observations": obs_scr,
                "Remaining Days - Mean": mean_other,
                "Remaining Days - Observations": obs_other,
                "T-test Value": t_stat,
                "T-test Significance": t_p_value,
                "T-test Degrees of Freedom": df,
                "Mann-Whitney U Z": z_value,
                "Mann-Whitney U Significance": u_p_value
            }

        # Formatting the display table of results.
        formatted_stats = {}
        for key, values in results.items():
            formatted_values = {}
            for stat_name, stat_value in values.items():
                if "Mean" in stat_name:
                    formatted_values[stat_name] = f"{(stat_value * 100):.4f}%"
                elif "Significance" in stat_name or "T-test" in stat_name or "Mann-Whitney" in stat_name:
                    formatted_values[stat_name] = f"{stat_value:.3f}"
                else:
                    formatted_values[stat_name] = stat_value

            formatted_stats[key] = formatted_values

        result_df = pd.DataFrame(formatted_stats).T
        result_numeric = pd.DataFrame(results).T


        return result_df,result_numeric




    def compute_daily_average(self,df):
        return df.mean()

    def compute_standard_deviation (self, df):
        return df.std()

    def compute_min(self, df):
        return df.min()

    def compute_max(self, df):
        return df.max()

    def compute_kurtosis(self,df):
        return df.kurtosis()

    def compute_skewness(self,df):
        return df.skew()

    def compute_range(self,df):
        result = df.max() - df.min()
        return result

    def nb_observations(self,df):
        return df.count()

    @staticmethod
    def check_adf_stationary(df):
        """
        Checks the stationary for each column of a returns DataFrame (df)
        using ADF tests. The conclusion about stationary is included in the result.
        """
        results = {}

        for column in df.columns:
            series = df[column].dropna()

            # ADF test
            adf_result = adfuller(series)
            adf_p_value = adf_result[1]
            adf_conclusion = "Stationary" if adf_p_value < 0.05 else "Non stationary"

            results[column] = {
                'ADF_p_value': round(adf_p_value, 3),
                'ADF_Conclusion': adf_conclusion,
            }

        result_df = pd.DataFrame(results).T
        result_df.columns = ['ADF_p_value', 'ADF_Conclusion']

        return result_df

    def T_test(self,santa_claus_df,other_df):

        # if self.compute_standard_deviation(santa_claus_df) != self.compute_standard_deviation(other_df):
        #   equal_var = False
        # else:
        #   equal_var = True

        t_stat, p_value = ttest_ind(santa_claus_df, other_df, equal_var=True)
        n_scr = len(santa_claus_df)
        n_autres = len(other_df)
        df = n_scr + n_autres - 2
        return t_stat,p_value,df

    def Mann_Whitney_test(self,santa_claus_df,other_df):
        u_stat, p_value = mannwhitneyu(santa_claus_df, other_df, alternative='two-sided')

        n_scr = len(santa_claus_df)
        n_autres = len(other_df)
        mean_u = n_scr * n_autres / 2
        std_u = ((n_scr * n_autres * (n_scr + n_autres + 1)) / 12) ** 0.5
        z_value = (u_stat - mean_u) / std_u
        return u_stat,p_value,z_value


    @staticmethod
    def chow_test(santa_claus_df, other_df, split_date: str):
        """
        Performs a Chow test for multiple indices based on a given date.

        Parameters:
        - santa_claus_df: DataFrame containing returns for the Santa Claus Rally periods.
        - other_df: DataFrame containing returns for the other periods.
        - split_date: Split date to perform the Chow test (format 'YYYY-MM-DD').

        Returns:
        - A dictionary where each key is an index, and each value is a dictionary containing
          the results of the Chow test (F-stat, p-value, RSS).
        """
        if other_df is None or santa_claus_df is None:
            raise ValueError("The DataFrames santa_claus_df and other_df cannot be None.")

        santa_claus_df = santa_claus_df.copy()
        other_df = other_df.copy()
        santa_claus_df['SantaClausRally'] = 1
        other_df['SantaClausRally'] = 0

        combined_df = pd.concat([santa_claus_df, other_df], ignore_index=False)
        combined_df = combined_df.sort_values(by='Date')

        results = {}

        for column in santa_claus_df.columns:
            if column == 'SantaClausRally':
                continue

            temp_df = combined_df[['SantaClausRally', column]].dropna()

            X = temp_df[['SantaClausRally']].values
            X = np.hstack([np.ones((X.shape[0], 1)), X])

            Y = temp_df[column].values

            split_index = combined_df.index.normalize() <= split_date
            X_part1 = X[split_index]
            Y_part1 = Y[split_index]
            X_part2 = X[~split_index]
            Y_part2 = Y[~split_index]

            if len(Y_part1) == 0 or len(Y_part2) == 0:
                results[column] = {
                    'Error': "Insufficient data in one of the periods."
                }
                continue

            # Total regression
            beta_full = np.linalg.inv(X.T @ X) @ (X.T @ Y)
            residuals_full = Y - X @ beta_full
            RSS_full = np.sum(residuals_full ** 2)

            # First period regression
            beta_part1 = np.linalg.inv(X_part1.T @ X_part1) @ (X_part1.T @ Y_part1)
            residuals_part1 = Y_part1 - X_part1 @ beta_part1
            RSS_part1 = np.sum(residuals_part1 ** 2)

            # Second period regression
            beta_part2 = np.linalg.inv(X_part2.T @ X_part2) @ (X_part2.T @ Y_part2)
            residuals_part2 = Y_part2 - X_part2 @ beta_part2
            RSS_part2 = np.sum(residuals_part2 ** 2)

            # Chow test parameters
            n_full = len(Y)
            n_part1 = len(Y_part1)
            n_part2 = len(Y_part2)
            k = X.shape[1]  # X number of parameters (2 here : intercept + dummy)

            # F stat Chow test
            numerator = (RSS_full - (RSS_part1 + RSS_part2)) / k
            denominator = (RSS_part1 + RSS_part2) / (n_full - 2 * k)
            F_stat = numerator / denominator

            # Related p-value
            p_value = 1 - f.cdf(F_stat, k, n_full - 2 * k)

            results[column] = {
                'F-stat': round(F_stat, 3),
                'p-value': round(p_value, 3),
                'RSS_full': round(RSS_full, 2),
                'RSS_part1': round(RSS_part1, 2),
                'RSS_part2': round(RSS_part2, 2),
                'n_full': n_full,
                'n_part1': n_part1,
                'n_part2': n_part2,
            }

        return pd.DataFrame(results).T

    def OLS_regression2(self, santa_claus_df, other_df, numeric: bool = False):
        if other_df is None:
            return "None","None"

        results = {}
        results_numeric = {}

        santa_claus_df = santa_claus_df.copy()
        other_df = other_df.copy()

        santa_claus_df.loc[:, 'SantaClausRally'] = 1  # 1 for Santa Claus Rally
        other_df.loc[:, 'SantaClausRally'] = 0  # 0 for others days

        combined_df = pd.concat([santa_claus_df, other_df], ignore_index=True)

        for column in santa_claus_df.columns:
            if column == 'SantaClausRally':
                continue

            temp_df = combined_df[['SantaClausRally', column]].dropna()

            X = temp_df[['SantaClausRally']].values
            X = np.hstack([np.ones((X.shape[0], 1)), X])

            y = temp_df[column].values

            # Estimation of OLS (Ordinary Least Squares) coefficients.
            X_transpose = X.T
            beta = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)

            beta_0 = beta[0]
            beta_1 = beta[1]

            # Residuals
            residuals = y - X.dot(beta)
            n = len(y)
            k = X.shape[1]
            vare = np.sum(residuals ** 2) / (n - k)

            # Variance-covariance matrix.
            varbet = vare * np.linalg.inv(X_transpose.dot(X))
            std_err = np.sqrt(np.diag(varbet))

            # t-statistics and p-values.
            t_stat = beta / std_err
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - k))

            # R-squared and F-statistic.
            ss_total = np.sum((y - np.mean(y)) ** 2)
            ss_residual = np.sum(residuals ** 2)
            r_squared = 1 - (ss_residual / ss_total)

            f_value = (r_squared / (k - 1)) / ((1 - r_squared) / (n - k))

            # Ljung-Box test for autocorrelation.
            lb_test = acorr_ljungbox(residuals, lags=[1], return_df=True)
            ljung_box_stat = lb_test["lb_stat"].iloc[0]
            ljung_box_pvalue = lb_test["lb_pvalue"].iloc[0]

            # Jarque-Bera test for the normality of residuals.
            jarque_bera_stat, jarque_bera_pvalue = stats.jarque_bera(residuals)

            # ARCH LM test for heteroscedasticity.
            arch_lm_stat, arch_lm_pvalue, _, _ = het_arch(residuals)

            # Storing the results
            results[column] = {
                'β0 (Mean of non-rally days)': f"{round(beta_0 * 100, 5)}%",
                'β1 (Difference between rally and non-rally days)': f"{round(beta_1 * 100, 5)}%",
                'Standard Error (β0)': f"{round(std_err[0] * 100, 4)}%",
                'Standard Error (β1)': f"{round(std_err[1] * 100, 4)}%",
                't-stat (β0)': round(t_stat[0], 3),
                't-stat (β1)': round(t_stat[1], 3),
                'p-value (β0)': round(p_value[0], 3),
                'p-value (β1)': round(p_value[1], 3),
                'R-squared': round(r_squared, 4),
                'F-value': round(f_value, 3),
                'Ljung-Box Stat': round(ljung_box_stat, 3),
                'Ljung-Box p-value': round(ljung_box_pvalue, 3),
                'Jarque-Bera Stat': round(jarque_bera_stat, 3),
                'Jarque-Bera p-value': round(jarque_bera_pvalue, 3),
                'ARCH LM Stat': round(arch_lm_stat, 3),
                'ARCH LM p-value': round(arch_lm_pvalue, 3)
            }

            results_numeric[column] = {
                'β0 (Mean of non-rally days)': beta_0,
                'β1 (Difference between rally and non-rally days)': beta_1,
                'Standard Error (β0)': std_err[0],
                'Standard Error (β1)': std_err[1],
                't-stat (β0)': t_stat[0],
                't-stat (β1)': t_stat[1],
                'p-value (β0)': p_value[0],
                'p-value (β1)': p_value[1],
                'R-squared': r_squared,
                'F-value': f_value,
                'Ljung-Box Stat': ljung_box_stat,
                'Ljung-Box p-value': ljung_box_pvalue,
                'Jarque-Bera Stat': jarque_bera_stat,
                'Jarque-Bera p-value': jarque_bera_pvalue,
                'ARCH LM Stat': arch_lm_stat,
                'ARCH LM p-value': arch_lm_pvalue
            }

        # Converting the results into a DataFrame.
        results_df = pd.DataFrame(results).T

        return results_df,pd.DataFrame(results_numeric).T



    def OLS_regression(self, santa_claus_df, other_df, numeric: bool = False):
        if other_df is None:
            return "None","None"

        results = {}
        results_numeric = {}

        santa_claus_df = santa_claus_df.copy()
        other_df = other_df.copy()

        santa_claus_df['SantaClausRally'] = 1  # 1 for Santa Claus Rally
        other_df['SantaClausRally'] = 0  # 0 for others days

        combined_df = pd.concat([santa_claus_df, other_df], ignore_index=True)

        for column in santa_claus_df.columns:
            if column == 'SantaClausRally':
                continue

            temp_df = combined_df[['SantaClausRally', column]].dropna()

            X = temp_df[['SantaClausRally']].values
            X = np.hstack([np.ones((X.shape[0], 1)), X])

            y = temp_df[column].values

            # Classic regression (OLS)
            X_transpose = X.T
            beta = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)

            beta_0 = beta[0]
            beta_1 = beta[1]

            residuals = y - X.dot(beta)
            n = len(y)
            k = X.shape[1]
            vare = np.sum(residuals ** 2) / (n - k)

            # variance-covariance matrix
            varbet = vare * np.linalg.inv(X_transpose.dot(X))
            std_err = np.sqrt(np.diag(varbet))

            # Statistics t and p-values
            t_stat = beta / std_err
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - k))

            ss_total = np.sum((y - np.mean(y)) ** 2)
            ss_residual = np.sum((y - X.dot(beta)) ** 2)
            r_squared = 1 - (ss_residual / ss_total)

            f_value = (r_squared / (k - 1)) / ((1 - r_squared) / (n - k))

            # Diagnostic tests
            lb_test = acorr_ljungbox(residuals, lags=[1], return_df=True)
            ljung_box_stat = lb_test["lb_stat"].iloc[0]
            ljung_box_pvalue = lb_test["lb_pvalue"].iloc[0]

            jarque_bera_stat, jarque_bera_pvalue = stats.jarque_bera(residuals)
            arch_lm_stat, arch_lm_pvalue, _, _ = het_arch(residuals)

            ## Engle's test for heterosckedasticity
            optimal_lag, aic_results = statistics.find_optimal_lag(residuals, max_lag=1)
            arch_test_result = sm.stats.diagnostic.het_arch(residuals, nlags=optimal_lag)
            arch_test_pval = arch_test_result[1]

            # OLS appropriated correction method
            _,ols_variants_results = self.OLS_variants(santa_claus_df, other_df, numeric=True)

            ## Assigning HAC, White or no correction based on above tests
            if ljung_box_pvalue < 0.05 and arch_test_pval < 0.05:
                correction = "HAC"  # 2
                correction_numeric = 2
            if ljung_box_pvalue > 0.05 and arch_test_pval < 0.05:
                correction = "White"  # 1
                correction_numeric = 1
            if ljung_box_pvalue > 0.05 and arch_test_pval > 0.05:
                correction = "no correction"  # 0
                correction_numeric = 0
            if ljung_box_pvalue < 0.05 and arch_test_pval > 0.05:
                correction = "HAC"
                correction_numeric = 2


            if ljung_box_pvalue < 0.05:  # Significant autocorrélation
                selected_method = "HAC"
                corrected_p_value = ols_variants_results.loc[f'{column} (HAC, maxlags=1)', 'p-value (β1)']
                corrected_p_value_constant = ols_variants_results.loc[f'{column} (HAC, maxlags=1)', 'p-value (β0)']
            elif arch_lm_pvalue < 0.05:  # Significant heteroskedasticity
                selected_method = "HC0"
                corrected_p_value = ols_variants_results.loc[f'{column} (HC0)', 'p-value (β1)']
                corrected_p_value_constant = ols_variants_results.loc[f'{column} (HC3)', 'p-value (β0)']
            else:
                selected_method = 'OLS classique'
                corrected_p_value = ols_variants_results.loc[f'{column} (OLS classique)', 'p-value (β1)']
                corrected_p_value_constant = ols_variants_results.loc[f'{column} (OLS classique)', 'p-value (β0)']

            results[column] = {
                'β0 (Mean of non-rally days)': f"{round(beta_0 * 100, 5)}%",
                'β1 (Difference between rally and non-rally days)': f"{round(beta_1 * 100, 5)}%",
                'Standard Error (β0)': f"{round(std_err[0] * 100, 4)}%",
                'Standard Error (β1)': f"{round(std_err[1] * 100, 4)}%",
                't-stat (β0)': round(t_stat[0], 3),
                't-stat (β1)': round(t_stat[1], 3),
                'p-value (β0)': round(p_value[0], 3),
                'p-value (β1)': round(p_value[1], 3),
                'Corrected p-value β0': round(corrected_p_value_constant, 3),
                'Corrected p-val β1': round(corrected_p_value, 3),
                'F-value': round(f_value, 3),
                'Correction Method': selected_method,
                'LJB p-value': round(ljung_box_pvalue, 3),
                'Jarque-Bera p-value': round(jarque_bera_pvalue, 3),
                'Arch_test p-value': round(arch_lm_pvalue, 3),
            }

            results_numeric[column] = {
                'β0 (Mean of non-rally days)': beta_0,
                'β1 (Difference between rally and non-rally days)': beta_1,
                'Standard Error (β0)': std_err[0],
                'Standard Error (β1)': std_err[1],
                't-stat (β0)': t_stat[0],
                't-stat (β1)': t_stat[1],
                'p-value (β0)': p_value[0],
                'p-value (β1)': p_value[1],
                'Corrected p-value β0': round(corrected_p_value_constant, 3),
                'Corrected p-val β1': round(corrected_p_value, 3),
                'F-value': f_value,
                'Ljung-Box Stat': ljung_box_stat,
                'Ljung-Box p-value': ljung_box_pvalue,
                'Jarque-Bera Stat': jarque_bera_stat,
                'Jarque-Bera p-value': jarque_bera_pvalue,
                'ARCH LM Stat': arch_lm_stat,
                'ARCH LM p-value': arch_lm_pvalue,
                'correction': correction_numeric
            }

        results_df = pd.DataFrame(results).T

        return results_df, pd.DataFrame(results_numeric).T

    # Dynamically determine the optimal lag based on AIC
    @staticmethod
    def find_optimal_lag(residuals, max_lag=5):
        aic_values = []
        lags = range(1, max_lag + 1)

        for lag in lags:
            try:
                model = AutoReg(residuals, lags=lag, old_names=False)
                fitted_model = model.fit()
                aic_values.append(fitted_model.aic)
            except ValueError:
                aic_values.append(np.nan)

        aic_df = pd.DataFrame({'Lag': lags, 'AIC': aic_values})
        aic_df = aic_df.dropna()
        optimal_lag = aic_df.loc[aic_df['AIC'].idxmin(), 'Lag']

        return optimal_lag, aic_df


    def OLS_variants(self, santa_claus_df, other_df, numeric: bool = False):
        if other_df is None:
            return "None","None"

        results = {}
        results_numeric = {}

        santa_claus_df = santa_claus_df.copy()
        other_df = other_df.copy()

        santa_claus_df['SantaClausRally'] = 1
        other_df['SantaClausRally'] = 0

        combined_df = pd.concat([santa_claus_df, other_df], ignore_index=True)

        for column in santa_claus_df.columns:
            if column == 'SantaClausRally':
                continue

            temp_df = combined_df[['SantaClausRally', column]].dropna()
            X = temp_df[['SantaClausRally']].values
            X = np.hstack([np.ones((X.shape[0], 1)), X])  # Adding the constant
            y = temp_df[column].values

            model = sm.OLS(y, X)

            # Classic OLS regression (non-robust).
            results_model_classic = model.fit()  # Fit the model without robust corrections.
            p_values_classic = results_model_classic.pvalues

            # Storing the results of the classic OLS regression.
            results[f'{column} (OLS classique)'] = {
                'p-value (β0)': round(p_values_classic[0], 3),
                'p-value (β1)': round(p_values_classic[1], 3)
            }

            # For the numerical results of the classic OLS regression.
            results_numeric[f'{column} (OLS classique)'] = {
                'p-value (β0)': p_values_classic[0],
                'p-value (β1)': p_values_classic[1]
            }

            # List of HC correction types to test.
            hc_types = ['HC0', 'HC1', 'HC2', 'HC3']

            # Calculating p-values for each HC correction type.
            for hc_type in hc_types:
                # Calculating results with the robust variance-covariance correction.
                results_model_robust = results_model_classic.get_robustcov_results(cov_type=hc_type)

                # Extracting the p-values.
                p_values = results_model_robust.pvalues

                # Storing the results for each HC type.
                results[f'{column} ({hc_type})'] = {
                    'p-value (β0)': round(p_values[0], 3),
                    'p-value (β1)': round(p_values[1], 3)
                }

                # For the numerical results of each HC type.
                results_numeric[f'{column} ({hc_type})'] = {
                    'p-value (β0)': p_values[0],
                    'p-value (β1)': p_values[1]
                }

            # Determining the optimal number of maxlags for HAC (Heteroscedasticity and Autocorrelation Consistent) estimation.
            maxlags_candidates = range(1, 21)  # Test lags from 1 to 20, adjust if necessary.
            aic_values = []
            bic_values = []

            #  Calculate AIC and BIC for each value of maxlags.
            for maxlags in maxlags_candidates:
                try:
                    results_model_hac = results_model_classic.get_robustcov_results(cov_type='HAC', maxlags=maxlags)
                    aic_values.append(results_model_hac.aic)
                    bic_values.append(results_model_hac.bic)
                except:
                    # In case of an error (e.g., if the sample size is too small for this maxlags), ignore this iteration.
                    aic_values.append(np.nan)
                    bic_values.append(np.nan)

            # Select the optimal maxlags based on the AIC criterion (here, AIC).
            best_maxlags = maxlags_candidates[np.argmin(aic_values)]

            # Final calculation of HAC results with the optimal maxlags.
            results_model_hac_optimal = results_model_classic.get_robustcov_results(cov_type='HAC',
                                                                                    maxlags=best_maxlags)

            # Extracting HAC p-values.
            p_values_hac = results_model_hac_optimal.pvalues

            # Storing the HAC results.
            results[f'{column} (HAC, maxlags={best_maxlags})'] = {
                'p-value (β0)': round(p_values_hac[0], 3),
                'p-value (β1)': round(p_values_hac[1], 3)
            }

            # For the numerical HAC results.
            results_numeric[f'{column} (HAC, maxlags={best_maxlags})'] = {
                'p-value (β0)': p_values_hac[0],
                'p-value (β1)': p_values_hac[1]
            }

        # Converting the results into a DataFrame.
        results_df = pd.DataFrame(results).T

        # Return the results as a numerical or standard DataFrame.
        return results_df,pd.DataFrame(results_numeric).T


    def bootstrap(self, santa_claus_df, other_df, n_iterations=1000, alpha=0.05, numeric: bool = False):
        if other_df is None:
            return "None","None"

        results = {}
        results_numeric = {}

        santa_claus_df = santa_claus_df.copy()
        other_df = other_df.copy()

        santa_claus_df.loc[:, 'SantaClausRally'] = 1
        other_df.loc[:, 'SantaClausRally'] = 0

        combined_df = pd.concat([santa_claus_df, other_df], ignore_index=True)

        for column in santa_claus_df.columns:
            if column == 'SantaClausRally':
                continue

            temp_df = combined_df[['SantaClausRally', column]].dropna()
            X = temp_df[['SantaClausRally']].values
            X = np.hstack([np.ones((X.shape[0], 1)), X])  # Adding the constant
            y = temp_df[column].values

            n = len(y)
            beta_samples = []

            for _ in range(n_iterations):
                # Resampling
                indices = np.random.choice(range(n), size=n, replace=True)
                X_sample = X[indices]
                y_sample = y[indices]

                beta = np.linalg.inv(X_sample.T.dot(X_sample)).dot(X_sample.T).dot(y_sample)
                beta_samples.append(beta)

            beta_samples = np.array(beta_samples)
            beta_means = np.mean(beta_samples, axis=0)
            ci = np.percentile(beta_samples, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)

            results[column] = {
                'β0 (Mean of non-rally days)': f"{round(beta_means[0] * 100, 5)}%",
                'CI (β0)': (round(ci[0, 0] * 100, 5), round(ci[1, 0] * 100, 5)),
                'β1 (Difference between rally and non-rally days)': f"{round(beta_means[1] * 100, 5)}%",
                'CI (β1)': (round(ci[0, 1] * 100, 5), round(ci[1, 1] * 100, 5))
            }

            results_numeric[column] = {
                'β0 (Mean of non-rally days)': f"{beta_means[0] * 100}%",
                'CI (β0)': (round(ci[0, 0] * 100, 5), ci[1, 0] * 100),
                'β1 (Difference between rally and non-rally days)': f"{beta_means[1] * 100}%",
                'CI (β1)': (round(ci[0, 1] * 100, 5), ci[1, 1] * 100)
            }

        # Converting the results into a DataFrame.
        results_df = pd.DataFrame(results).T


        return results_df,pd.DataFrame(results_numeric).T

