import pandas as pd
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.ar_model import AutoReg

class SantaClausFactorAnalysis:
    def __init__(self):
        factor_database = FactorsDataBase()
        self.factors = factor_database.factors_formatted
        self.common_earliest_date_factors = self.get_common_earliest_date_factors()
        self.trimmed_factors = self.trim_factors()
        self.start_end_indices_factors = self.get_year_start_end_indices_factors()
        self.start_end_indices_trimmed_factors = self.get_year_start_end_indices_trimmed_factors()

        # Santa Claus Analysis
        # Results
        self.results_overall = self.santa_claus_factor_analysis_overall()
        self.santa_claus_days = self.days(december_or_santa_claus="santa_claus")
        self.results_sub_periods = self.santa_claus_factor_analysis_sub_periods()

        # Correction
        self.p_values_extra_overall = self.get_extra_tests_data_overall(self.results_overall) # for the overall period
        self.correction_time_series = self.get_data_santa_claus_factor_analysis_sub_periods(self.results_sub_periods,3, "correction")

        # P-values and proportion
        # P-values corrected overall
        self.p_values_time_series_corrected_overall = self.get_p_values_corrected_overall(december_or_santa_claus="santa_claus")

        # P-values corrected sub periods (time series)
        self.p_values_time_series = self.get_data_santa_claus_factor_analysis_sub_periods(self.results_sub_periods)
        self.p_values_time_series_extra = self.get_extra_tests_data(self.results_sub_periods)
        self.p_values_time_series_extra_HCO = self.get_extra_tests_data(self.results_sub_periods)["p_values_time_series_HCO"]
        self.p_values_time_series_extra_HAC = self.get_extra_tests_data(self.results_sub_periods)["p_values_time_series_HAC"]
        self.p_values_time_series_corrected = self.get_p_values_corrected(self.results_sub_periods, december_or_santa_claus="santa_claus")
        # Tests
        self.p_values_mann_whitney = self.get_data_santa_claus_factor_analysis_sub_periods(self.results_sub_periods, 2,"Mann-Whitney U Significance")
        self.p_values_t_test = self.get_data_santa_claus_factor_analysis_sub_periods(self.results_sub_periods, 2,"T-test Significance")
        # Proportion
        self.p_values_proportion_pos_stat = self.get_santa_claus_factor_analysis_sub_periods_proportion(self.p_values_time_series)
        self.p_values_proportion_pos = self.get_santa_claus_factor_analysis_sub_periods_proportion(self.p_values_time_series, stat_significance=False)

        self.p_values_proportion_HCO_pos_stat = self.get_santa_claus_factor_analysis_sub_periods_proportion(self.p_values_time_series_extra_HCO)
        self.p_values_proportion_HCO_pos = self.get_santa_claus_factor_analysis_sub_periods_proportion(self.p_values_time_series_extra_HCO, stat_significance=False)

        self.p_values_proportion_HAC_pos_stat = self.get_santa_claus_factor_analysis_sub_periods_proportion(self.p_values_time_series_extra_HAC)
        self.p_values_proportion_HAC_pos = self.get_santa_claus_factor_analysis_sub_periods_proportion(self.p_values_time_series_extra_HAC, stat_significance=False)

        self.p_values_proportion_corrected_pos_stat = self.get_santa_claus_factor_analysis_sub_periods_proportion(self.p_values_time_series_corrected)
        self.p_values_proportion_corrected_pos = self.get_santa_claus_factor_analysis_sub_periods_proportion(self.p_values_time_series_corrected, stat_significance=False)

        self.p_values_mann_whitney_proportion_pos_stat = self.get_santa_claus_factor_analysis_sub_periods_proportion(self.p_values_mann_whitney)
        self.p_values_mann_whitney_proportion_pos = self.get_santa_claus_factor_analysis_sub_periods_proportion(self.p_values_mann_whitney, stat_significance=False)

        self.p_values_t_test_proportion_pos_stat = self.get_santa_claus_factor_analysis_sub_periods_proportion(self.p_values_t_test)
        self.p_values_t_test_proportion_pos = self.get_santa_claus_factor_analysis_sub_periods_proportion(self.p_values_t_test, stat_significance=False)


        self.relative_ret_diff = self.get_data_santa_claus_factor_analysis_sub_periods(self.results_sub_periods,3, "no importance", True)

        # Mean excess returns
        self.mean_excess_ret = self.compute_mean_returns_across_years(self.results_sub_periods)

        # FINAL RESULTS
        self.consolidated_proportion_pos = self.get_consolidated_proportion(pos_or_pos_stat="pos", december_or_santa_claus="santa_claus")
        self.consolidated_proportion_pos_stat = self.get_consolidated_proportion(pos_or_pos_stat="pos_stat", december_or_santa_claus="santa_claus")


        # December Analysis
        # Results
        self.results_overall_december = self.santa_claus_factor_analysis_overall(december_or_santa_claus="december")
        self.dec_days = self.days(december_or_santa_claus="december")
        self.results_sub_periods_december = self.santa_claus_factor_analysis_sub_periods(december_or_santa_claus="december")

        # Correction
        self.p_values_extra_overall_december = self.get_extra_tests_data_overall(self.results_overall_december)  # for the overall period
        self.correction_time_series_december = self.get_data_santa_claus_factor_analysis_sub_periods(self.results_sub_periods_december, 3,"correction")

        # P-values and proportion
        # P-values corrected overall
        self.p_values_time_series_corrected_overall_december = self.get_p_values_corrected_overall(december_or_santa_claus="december")

        # P-values corrected sub periods (time series)
        self.p_values_time_series_december = self.get_data_santa_claus_factor_analysis_sub_periods(self.results_sub_periods_december)
        self.p_values_time_series_december_extra = self.get_extra_tests_data(self.results_sub_periods_december)
        self.p_values_time_series_december_extra_HCO = self.get_extra_tests_data(self.results_sub_periods_december)["p_values_time_series_HCO"]
        self.p_values_time_series_december_extra_HAC = self.get_extra_tests_data(self.results_sub_periods_december)["p_values_time_series_HAC"]
        self.p_values_time_series_december_corrected = self.get_p_values_corrected(self.results_sub_periods_december, december_or_santa_claus="december")
        # Tests
        self.p_values_mann_whitney_december = self.get_data_santa_claus_factor_analysis_sub_periods(self.results_sub_periods_december, 2, "Mann-Whitney U Significance")
        self.p_values_t_test_dec = self.get_data_santa_claus_factor_analysis_sub_periods(self.results_sub_periods_december, 2, "T-test Significance")
        # Proportion
        self.p_values_proportion_pos_stat_dec = self.get_santa_claus_factor_analysis_sub_periods_proportion(self.p_values_time_series_december)
        self.p_values_proportion_pos_dec = self.get_santa_claus_factor_analysis_sub_periods_proportion(self.p_values_time_series_december, stat_significance=False)

        self.p_values_proportion_HCO_pos_stat_dec = self.get_santa_claus_factor_analysis_sub_periods_proportion(self.p_values_time_series_december_extra_HCO)
        self.p_values_proportion_HCO_pos_dec = self.get_santa_claus_factor_analysis_sub_periods_proportion(self.p_values_time_series_december_extra_HCO, stat_significance=False)

        self.p_values_proportion_HAC_pos_stat_dec = self.get_santa_claus_factor_analysis_sub_periods_proportion(self.p_values_time_series_december_extra_HAC)
        self.p_values_proportion_HAC_pos_dec = self.get_santa_claus_factor_analysis_sub_periods_proportion(self.p_values_time_series_december_extra_HAC, stat_significance=False)

        self.p_values_proportion_corrected_pos_stat_dec = self.get_santa_claus_factor_analysis_sub_periods_proportion(self.p_values_time_series_december_corrected)
        self.p_values_proportion_corrected_pos_dec = self.get_santa_claus_factor_analysis_sub_periods_proportion(self.p_values_time_series_december_corrected, stat_significance=False)

        self.p_values_mann_whitney_proportion_pos_stat_dec = self.get_santa_claus_factor_analysis_sub_periods_proportion(self.p_values_mann_whitney_december)
        self.p_values_mann_whitney_proportion_pos_dec = self.get_santa_claus_factor_analysis_sub_periods_proportion(self.p_values_mann_whitney_december, stat_significance=False)

        self.p_values_t_test_proportion_pos_stat_dec = self.get_santa_claus_factor_analysis_sub_periods_proportion(self.p_values_t_test_dec)
        self.p_values_t_test_proportion_pos_dec = self.get_santa_claus_factor_analysis_sub_periods_proportion(self.p_values_t_test_dec, stat_significance=False)


        self.relative_ret_diff_dec = self.get_data_santa_claus_factor_analysis_sub_periods(self.results_sub_periods_december, 3,"no importance", True)

        # Mean excess returns
        self.mean_excess_ret_dec = self.compute_mean_returns_across_years(self.results_sub_periods_december)

        # FINAL RESULTS
        self.consolidated_proportion_pos_dec = self.get_consolidated_proportion(pos_or_pos_stat="pos", december_or_santa_claus="december")
        self.consolidated_proportion_pos_stat_dec = self.get_consolidated_proportion(pos_or_pos_stat="pos_stat",december_or_santa_claus="december")


    def days(self, december_or_santa_claus:str="december"):
        for key in self.factors.keys():
            df_returns_factors = self.factors[key]
            bucket_factors = SantaClausBucket(df_returns_factors)
            if december_or_santa_claus == 'december':
                return bucket_factors.december_days
            elif december_or_santa_claus == 'santa_claus':
                return bucket_factors.santa_claus_days
            else:
                raise TypeError(
                    "Wrong value entered for december_or_santa_claus. 'december' or 'santa_claus' accepted.")

    def santa_claus_factor_analysis_overall(self, december_or_santa_claus:str="santa_claus"):
        # Checking inputs
        if december_or_santa_claus not in ["santa_claus", "december"]:
            raise ValueError("Wrong name of period entered. Supported: 'santa_claus' or 'december'.")

        ## Santa Claus Factor Analysis - Overall years
        results = {factor: [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()] for factor in
                   self.factors.keys()}  # to store results

        # Iterating through each factor
        for key in self.factors.keys():
            df_returns_factors = self.factors[key]
            bucket_factors = SantaClausBucket(df_returns_factors)

            # Santa Claus Analysis
            if december_or_santa_claus == "santa_claus":
                santa_claus_df_factors = bucket_factors.santa_claus_df
                other_df_factors = bucket_factors.other_df

                # Descriptive Statistics
                stats_factors = statistics(df_returns_factors, None)
                results[key][0] = stats_factors.desc_stats_numeric

                # Statistical Tests
                stats_factors_test = statistics(santa_claus_df_factors, other_df_factors)
                results[key][1] = stats_factors_test.desc_stats_numeric
                results[key][2] = stats_factors_test.test_stats_numeric

                # OLS Regression
                results[key][3] = stats_factors_test.OLS_stats_numeric

                # Extra tests
                # Bootstrap
                stats_factors_test = statistics(santa_claus_df_factors, other_df_factors, extratest=True)
                # results[key][4] = stats_factors_test.bootstrap_stats_numeric

                # OLS Variant
                results[key][5] = stats_factors_test.OLS_variants_numeric

            # December Analysis
            elif december_or_santa_claus == "december":
                santa_claus_df_factors = bucket_factors.santa_claus_df
                other_df_factors = bucket_factors.other_df

                dec_df_factors = bucket_factors.december_df
                dec_other_df_factors = bucket_factors.december_other_df

                # Descriptive Statistics
                stats_factors = statistics(santa_claus_df_factors, None, dec_df_factors, None)
                results[key][0] = stats_factors.dec_desc_stats_numeric

                # Statistical Tests
                stats_factors_test = statistics(santa_claus_df_factors, other_df_factors, dec_df_factors, dec_other_df_factors)
                results[key][1] = stats_factors_test.dec_desc_stats_numeric
                results[key][2] = stats_factors_test.dec_test_stats_numeric

                # OLS Regression
                results[key][3] = stats_factors_test.dec_OLS_stats_numeric

                # Extra tests
                # Bootstrap
                stats_factors_test = statistics(santa_claus_df_factors, other_df_factors, dec_df_factors, dec_other_df_factors, extratest=True)
                # results[key][4] = stats_factors_test.dec_bootstrap_stats_numeric

                # OLS Variant
                results[key][5] = stats_factors_test.dec_OLS_variants_numeric

        return results

    def get_year_start_end_indices_factors(self):
        return Utilities.get_year_start_end_indices_from_dict(self.factors)

    def get_year_start_end_indices_trimmed_factors(self):
        return Utilities.get_year_start_end_indices_from_dict(self.trimmed_factors)

    def get_common_earliest_date_factors(self):
        # Getting all datetime indices of factors and storing them into a list
        all_datetime_indices_factors = []
        for factor in self.factors.keys():
            all_datetime_indices_factors.append(self.factors[factor].index)

        # Once all datetime indices of factors are stored, we can call get_common_earliest_date()
        common_earliest_dates_factors = Utilities.get_common_earliest_date(all_datetime_indices_factors)

        return common_earliest_dates_factors

    def trim_factors(self):
        # To store data
        trimmed_factors = {factor: pd.DataFrame for factor in self.factors.keys()}
        for factor in self.factors.keys():
            trimmed_factors[factor] = self.factors[factor].loc[self.common_earliest_date_factors:,:]

        return trimmed_factors


    def santa_claus_factor_analysis_sub_periods(self, december_or_santa_claus:str="santa_claus"):
        # Checking inputs
        if december_or_santa_claus not in ["santa_claus", "december"]:
            raise ValueError("Wrong name of period entered. Supported: 'santa_claus' or 'december'.")

        ## Santa Claus Factor Analysis - Sub periods
        # Getting start and end dates/indices of each sub periods
        start_end_indices_trimmed_factors = self.start_end_indices_trimmed_factors

        # To store results
        results_sub_periods = {factor: {
                                        str(self.trimmed_factors[factor].index[end_date]): [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()] # str mandatory because some issues when putting a date as key in a dict
                                        for start_date, end_date in start_end_indices_trimmed_factors[factor]
                                        }
                                for factor in self.trimmed_factors.keys()
                                }

        # Iterating through each factor
        for key in self.trimmed_factors.keys():
            # Iterating through each sub periods
            for i in range(len(start_end_indices_trimmed_factors[key])):
                start_idx = start_end_indices_trimmed_factors[key][i][0]
                end_idx = start_end_indices_trimmed_factors[key][i][1]

                df_returns_factors = self.trimmed_factors[key].iloc[start_idx:end_idx+1,:] # +1 because .iloc excludes the outer bound
                bucket_factors = SantaClausBucket(df_returns_factors)

                if december_or_santa_claus == "santa_claus":
                    santa_claus_df_factors = bucket_factors.santa_claus_df
                    other_df_factors = bucket_factors.other_df

                    # Descriptive Statistics
                    stats_factors = statistics(df_returns_factors, None)

                    # To store results
                    end_date = str(self.trimmed_factors[key].index[end_idx])
                    results_sub_periods[key][end_date][0] = stats_factors.desc_stats_numeric

                    # Statistical Tests
                    stats_factors_test = statistics(santa_claus_df_factors, other_df_factors)
                    results_sub_periods[key][end_date][1] = stats_factors_test.desc_stats_numeric
                    results_sub_periods[key][end_date][2] = stats_factors_test.test_stats_numeric

                    # OLS Regression
                    results_sub_periods[key][end_date][3] = stats_factors_test.OLS_stats_numeric

                    # Extra tests
                    # Bootstrap
                    stats_factors_test = statistics(santa_claus_df_factors, other_df_factors, extratest=True)
                    # results_sub_periods[key][end_date][4] = stats_factors_test.bootstrap_stats_numeric

                    # OLS Variant
                    results_sub_periods[key][end_date][5] = stats_factors_test.OLS_variants_numeric


                elif december_or_santa_claus == "december":
                    santa_claus_df_factors = bucket_factors.santa_claus_df
                    other_df_factors = bucket_factors.other_df

                    dec_df_factors = bucket_factors.december_df
                    dec_other_df_factors = bucket_factors.december_other_df

                    # Descriptive Statistics
                    stats_factors = statistics(santa_claus_df_factors, None, dec_df_factors, None)

                    # To store results
                    end_date = str(self.trimmed_factors[key].index[end_idx])
                    results_sub_periods[key][end_date][0] = stats_factors.dec_desc_stats_numeric

                    # Statistical Tests
                    stats_factors_test = statistics(santa_claus_df_factors, other_df_factors, dec_df_factors, dec_other_df_factors)
                    results_sub_periods[key][end_date][1] = stats_factors_test.dec_desc_stats_numeric
                    results_sub_periods[key][end_date][2] = stats_factors_test.dec_test_stats_numeric

                    # OLS Regression
                    results_sub_periods[key][end_date][3] = stats_factors_test.dec_OLS_stats_numeric

                    # Extra tests
                    # Bootstrap
                    stats_factors_test = statistics(santa_claus_df_factors, other_df_factors, dec_df_factors,
                                                    dec_other_df_factors, extratest=True)
                    # results_sub_periods[key][end_date][4] = stats_factors_test.dec_bootstrap_stats_numeric

                    # OLS Variant
                    results_sub_periods[key][end_date][5] = stats_factors_test.dec_OLS_variants_numeric

        return results_sub_periods

    def get_data_santa_claus_factor_analysis_sub_periods(self, results_sub_periods, number_df:int=3, col_name:str="p-value (β1)", relative_ret_diff:bool=False):
        # To store results
        p_values_time_series = {factor: pd.DataFrame() for factor in self.trimmed_factors.keys()}

        # Iterating through each factor
        for factor in self.trimmed_factors.keys():
            x_str = list( results_sub_periods[factor].keys() )
            x_datetime = [pd.to_datetime(date_str) for date_str in x_str]
            # Iterating through each date
            for i, date in enumerate(x_str):
                # We want to retrieve time series of p-values for all countries for each factor
                # The Santa Claus rally is present if both p-value of beta1 is statistically significant and beta1 is positive
                # Hence, it's why we're multiplying by the sign of beta1
                if relative_ret_diff:

                    ret_diff = results_sub_periods[factor][date][number_df].loc[:, "β1 (Difference between rally and non-rally days)"]
                    mean_non_rally = results_sub_periods[factor][date][number_df].loc[:, "β0 (Mean of non-rally days)"]
                    ret_diff_relative = ret_diff / mean_non_rally
                    dict_for_df = {x_datetime[i]: ret_diff_relative}

                    # First date to store
                    if i == 0:
                        p_values_time_series[factor] = pd.DataFrame(dict_for_df).T
                    else:
                        # For other dates, we concat
                        p_values_time_series[factor] = pd.concat(
                            [p_values_time_series[factor], pd.DataFrame(dict_for_df).T], axis=0)

                else:

                    sign_beta_one = np.sign(results_sub_periods[factor][date][3].loc[:,"β1 (Difference between rally and non-rally days)"])
                    p_values = results_sub_periods[factor][date][number_df].loc[:,col_name] * sign_beta_one
                    dict_for_df = {x_datetime[i]: p_values}

                    # First date to store
                    if i==0:
                        p_values_time_series[factor] = pd.DataFrame(dict_for_df).T
                    else:
                        # For other dates, we concat
                        p_values_time_series[factor] = pd.concat([p_values_time_series[factor], pd.DataFrame(dict_for_df).T], axis=0)

        return p_values_time_series

    def get_extra_tests_data(self, results_sub_periods):

        p_values_time_series_HCO = {factor: pd.DataFrame() for factor in self.trimmed_factors.keys()}
        p_values_time_series_HC1 = {factor: pd.DataFrame() for factor in self.trimmed_factors.keys()}
        p_values_time_series_HC2 = {factor: pd.DataFrame() for factor in self.trimmed_factors.keys()}
        p_values_time_series_HC3 = {factor: pd.DataFrame() for factor in self.trimmed_factors.keys()}
        p_values_time_series_HAC = {factor: pd.DataFrame() for factor in self.trimmed_factors.keys()}
        p_values_time_series_extra = {"p_values_time_series_HCO": p_values_time_series_HCO,
                                      "p_values_time_series_HC1": p_values_time_series_HC1,
                                      "p_values_time_series_HC2": p_values_time_series_HC2,
                                      "p_values_time_series_HC3": p_values_time_series_HC3,
                                      "p_values_time_series_HAC": p_values_time_series_HAC}

        # Iterating through each p-values extra
        # start-line to retrieve lines of p-values
        start_line = 0
        for p_value_extra in p_values_time_series_extra.keys():
            start_line += 1
            # Iterating through each factor
            for factor in self.trimmed_factors.keys():
                x_str = list(results_sub_periods[factor].keys())
                x_datetime = [pd.to_datetime(date_str) for date_str in x_str]
                # Iterating through each date
                for i, date in enumerate(x_str):
                    # We want to retrieve time series of p-values for all countries for each factor
                    # The Santa Claus rally is present if both p-value of beta1 is statistically significant and beta1 is positive
                    # Hence, it's why we're multiplying by the sign of beta1
                    sign_beta_one = np.sign(results_sub_periods[factor][date][3].loc[:, "β1 (Difference between rally and non-rally days)"])
                    sign_beta_one.index = results_sub_periods[factor][date][5].iloc[start_line::6]["p-value (β1)"].index

                    # For p_values_time_series_extra
                    p_values = results_sub_periods[factor][date][5].iloc[start_line::6]["p-value (β1)"] * sign_beta_one
                    dict_for_df = {x_datetime[i]: p_values}

                    # First date to store
                    if i == 0:
                        p_values_time_series_extra[p_value_extra][factor] = pd.DataFrame(dict_for_df).T
                    else:
                        # For other dates, we concat
                        p_values_time_series_extra[p_value_extra][factor] = pd.concat([p_values_time_series_extra[p_value_extra][factor], pd.DataFrame(dict_for_df).T], axis=0)

        return p_values_time_series_extra

    def get_extra_tests_data_overall(self, results_overall):

        p_values_time_series_HCO = {factor: pd.DataFrame() for factor in self.trimmed_factors.keys()}
        p_values_time_series_HC1 = {factor: pd.DataFrame() for factor in self.trimmed_factors.keys()}
        p_values_time_series_HC2 = {factor: pd.DataFrame() for factor in self.trimmed_factors.keys()}
        p_values_time_series_HC3 = {factor: pd.DataFrame() for factor in self.trimmed_factors.keys()}
        p_values_time_series_HAC = {factor: pd.DataFrame() for factor in self.trimmed_factors.keys()}
        p_values_time_series_extra = {"p_values_time_series_HCO": p_values_time_series_HCO,
                                      "p_values_time_series_HC1": p_values_time_series_HC1,
                                      "p_values_time_series_HC2": p_values_time_series_HC2,
                                      "p_values_time_series_HC3": p_values_time_series_HC3,
                                      "p_values_time_series_HAC": p_values_time_series_HAC}

        # Iterating through each p-values extra
        # start-line to retrieve lines of p-values
        start_line = 0
        for p_value_extra in p_values_time_series_extra.keys():
            start_line += 1
            # Iterating through each factor
            for factor in self.trimmed_factors.keys():

                # We want to retrieve time series of p-values for all countries for each factor
                # The Santa Claus rally is present if both p-value of beta1 is statistically significant and beta1 is positive
                # Hence, it's why we're multiplying by the sign of beta1
                sign_beta_one = np.sign(results_overall[factor][3]["β1 (Difference between rally and non-rally days)"])
                sign_beta_one.index = results_overall[factor][5].iloc[start_line::6]["p-value (β1)"].index

                # For p_values_time_series_extra
                p_values = results_overall[factor][5].iloc[start_line::6]["p-value (β1)"] * sign_beta_one

                # To store
                p_values_time_series_extra[p_value_extra][factor] = pd.DataFrame(p_values)
                p_values_time_series_extra[p_value_extra][factor].rename(columns={0:f"{p_value_extra}"}, inplace=True)

        return p_values_time_series_extra

    def get_p_values_corrected(self, results_sub_periods, december_or_santa_claus:str):
        # To store results
        p_values_time_series_corrected = {factor: pd.DataFrame(data=np.nan, index=self.p_values_time_series[factor].index, columns=[self.p_values_time_series[factor].columns]) for factor in self.trimmed_factors.keys()}

        # Iterating through each factor
        for factor in self.trimmed_factors.keys():
            x_str = list(results_sub_periods[factor].keys())
            x_datetime = [pd.to_datetime(date_str) for date_str in x_str]
            # Iterating through each date
            for _, date in enumerate(x_str):
                # We want to retrieve time series of p-values for all countries for each factor
                # The Santa Claus rally is present if both p-value of beta1 is statistically significant and beta1 is positive
                # Hence, it's why we're multiplying by the sign of beta1
                if december_or_santa_claus=="santa_claus":

                    correction = np.abs(self.correction_time_series[factor].loc[date])

                    # Iterating through each value of the df... (long, maybe a vectorized way)
                    for i in range(0,correction.shape[0]):
                        current_correction = correction.iloc[i]
                        if current_correction == 0:
                            corresponding_pval = self.p_values_time_series[factor].loc[date].iloc[i]
                        if current_correction == 1:
                            corresponding_pval = self.p_values_time_series_extra["p_values_time_series_HCO"][factor].loc[date].iloc[i]
                        if current_correction == 2:
                            corresponding_pval = self.p_values_time_series_extra["p_values_time_series_HAC"][factor].loc[date].iloc[i]

                        index_for_iloc = p_values_time_series_corrected[factor].index.get_loc(date)
                        p_values_time_series_corrected[factor].iloc[index_for_iloc, i] = corresponding_pval

                elif december_or_santa_claus == "december":

                    correction = np.abs(self.correction_time_series_december[factor].loc[date])

                    # Iterating through each value of the df... (long, maybe a vectorized way)
                    for i in range(0, correction.shape[0]):
                        current_correction = correction.iloc[i]
                        if current_correction == 0:
                            corresponding_pval = self.p_values_time_series_december[factor].loc[date].iloc[i]
                        if current_correction == 1:
                            corresponding_pval = \
                            self.p_values_time_series_december_extra["p_values_time_series_HCO"][factor].loc[date].iloc[i]
                        if current_correction == 2:
                            corresponding_pval = \
                            self.p_values_time_series_december_extra["p_values_time_series_HAC"][factor].loc[date].iloc[i]

                        index_for_iloc = p_values_time_series_corrected[factor].index.get_loc(date)
                        p_values_time_series_corrected[factor].iloc[index_for_iloc, i] = corresponding_pval
                else:
                    raise ValueError("Wrong string entered for december_or_santa_claus. Should be 'december' or 'santa_claus'.")

        return p_values_time_series_corrected

    def get_p_values_corrected_overall(self, december_or_santa_claus:str):
        # To store results
        p_values_corrected_overall = {factor: pd.DataFrame(data=np.nan, index=self.results_overall["MKT"][3].index, columns=["p-val corrected overall_period"]) for factor in self.trimmed_factors.keys()}

        # Iterating through each factor
        for factor in self.trimmed_factors.keys():

            # We want to retrieve p-values for all countries for each factor
            # The Santa Claus rally is present if both p-value of beta1 is statistically significant and beta1 is positive
            # Hence, it's why we're multiplying by the sign of beta1
            if december_or_santa_claus=="santa_claus":

                correction = np.abs(self.results_overall[factor][3]["correction"])
                sign = np.sign(self.results_overall[factor][3]["β1 (Difference between rally and non-rally days)"])

                # Iterating through each value of the df... (long, maybe a vectorized way)
                for i in range(0,correction.shape[0]):
                    current_correction = correction.iloc[i]
                    if current_correction == 0:
                        corresponding_pval = self.results_overall[factor][3]["p-value (β1)"].iloc[i]
                    if current_correction == 1:
                        corresponding_pval = self.p_values_extra_overall["p_values_time_series_HCO"][factor].iloc[i]
                    if current_correction == 2:
                        corresponding_pval = self.p_values_extra_overall["p_values_time_series_HAC"][factor].iloc[i]

                    p_values_corrected_overall[factor].iloc[i] = corresponding_pval * sign.iloc[i]

            elif december_or_santa_claus == "december":

                correction = np.abs(self.results_overall_december[factor][3]["correction"])
                sign = np.sign(self.results_overall_december[factor][3]["β1 (Difference between rally and non-rally days)"])

                # Iterating through each value of the df... (long, maybe a vectorized way)
                for i in range(0, correction.shape[0]):
                    current_correction = correction.iloc[i]
                    if current_correction == 0:
                        corresponding_pval = self.results_overall_december[factor][3]["p-value (β1)"].iloc[i]
                    if current_correction == 1:
                        corresponding_pval = self.p_values_extra_overall_december["p_values_time_series_HCO"][factor].iloc[i]
                    if current_correction == 2:
                        corresponding_pval = self.p_values_extra_overall_december["p_values_time_series_HAC"][factor].iloc[i]

                    p_values_corrected_overall[factor].iloc[i] = corresponding_pval * sign.iloc[i]
            else:
                raise ValueError("Wrong string entered for december_or_santa_claus. Should be 'december' or 'santa_claus'.")

        return p_values_corrected_overall

    def get_santa_claus_factor_analysis_sub_periods_proportion(self, df, alpha:float=0.05, sign:str="positive", stat_significance:bool=True):
        # To store results
        prop = {factor: pd.DataFrame() for factor in self.trimmed_factors}
        for factor in self.trimmed_factors:

            if stat_significance:
                if sign == "both":
                    flg_significance = np.abs(df[factor])<=alpha
                elif sign == "positive":
                    condition_positive = df[factor] > 0
                    flg_significance_stat = np.abs(df[factor]) <= alpha
                    flg_significance = condition_positive & flg_significance_stat
                elif sign == "negative":
                    condition_positive = df[factor] < 0
                    flg_significance_stat = np.abs(df[factor]) <= alpha
                    flg_significance = condition_positive & flg_significance_stat
                else:
                    raise ValueError("sign must be among 'both', 'positive', 'negative'")

                prop_df = pd.DataFrame(flg_significance.sum(axis=0) / flg_significance.shape[0], columns=[f"Proportion of Santa Claus Rally {sign} for stat_sign:{stat_significance} (alpha={alpha})"])
                prop[factor] = prop_df

            else:
                if sign == "both":
                    raise ValueError(" When stat_significance is set to False, sign cannot be 'both'. Must be either 'positive' or 'negative'.")
                elif sign == "positive":
                    flg_significance = df[factor] > 0
                elif sign == "negative":
                    flg_significance = df[factor] < 0
                else:
                    raise ValueError("sign must be among 'positive', 'negative'")

                prop_df = pd.DataFrame(flg_significance.sum(axis=0)/flg_significance.shape[0], columns=[f"Proportion of Santa Claus Rally {sign} for stat_sign:{stat_significance} (alpha={alpha})"])
                prop[factor] = prop_df

        return prop

    def get_consolidated_proportion(self, pos_or_pos_stat: str, december_or_santa_claus: str):
        prop_concat_dict = {factor: pd.DataFrame() for factor in self.trimmed_factors.keys()}
        for factor in self.trimmed_factors.keys():
            if pos_or_pos_stat == "pos":
                if december_or_santa_claus == "santa_claus":
                    current_concat = pd.concat(
                        [self.p_values_proportion_corrected_pos[factor].reset_index(drop=True),
                         self.p_values_proportion_pos[factor].reset_index(drop=True),
                         self.p_values_proportion_HCO_pos[factor].reset_index(drop=True),
                         self.p_values_proportion_HAC_pos[factor].reset_index(drop=True),
                         self.p_values_mann_whitney_proportion_pos[factor].reset_index(drop=True),
                         self.p_values_t_test_proportion_pos[factor].reset_index(drop=True),
                         ],
                        axis=1)
                    current_concat.columns = ["SCR corrected", "SCR OLS", "SCR HCO", "SCR HAC", "SCR M-W", "SCR T-Test"]
                    current_concat.index = self.correction_time_series[factor].columns
                    prop_concat_dict[factor] = current_concat

                elif december_or_santa_claus == "december":
                    current_concat = pd.concat([self.p_values_proportion_corrected_pos_dec[
                                                    factor].reset_index(drop=True),
                                                self.p_values_proportion_pos_dec[
                                                    factor].reset_index(drop=True),
                                                self.p_values_proportion_HCO_pos_dec[
                                                    factor].reset_index(drop=True),
                                                self.p_values_proportion_HAC_pos_dec[
                                                    factor].reset_index(drop=True),
                                                self.p_values_mann_whitney_proportion_pos_dec[
                                                    factor].reset_index(drop=True),
                                                self.p_values_t_test_proportion_pos_dec[
                                                    factor].reset_index(drop=True),
                                                ],
                                               axis=1)
                    current_concat.columns = ["SCR corrected", "SCR OLS", "SCR HCO", "SCR HAC", "SCR M-W",
                                              "SCR T-Test"]
                    current_concat.index = self.correction_time_series[factor].columns
                    prop_concat_dict[factor] = current_concat
                else:
                    raise ValueError("december_or_santa_claus wrong. Must be either 'december' or 'santa_claus'.")

            elif pos_or_pos_stat == "pos_stat":
                if december_or_santa_claus == "santa_claus":
                    current_concat = pd.concat([self.p_values_proportion_corrected_pos_stat[
                                                    factor].reset_index(drop=True),
                                                self.p_values_proportion_pos_stat[
                                                    factor].reset_index(drop=True),
                                                self.p_values_proportion_HCO_pos_stat[
                                                    factor].reset_index(drop=True),
                                                self.p_values_proportion_HAC_pos_stat[
                                                    factor].reset_index(drop=True),
                                                self.p_values_mann_whitney_proportion_pos_stat[
                                                    factor].reset_index(drop=True),
                                                self.p_values_t_test_proportion_pos_stat[
                                                    factor].reset_index(drop=True),
                                                ],
                                               axis=1)
                    current_concat.columns = ["SCR corrected", "SCR OLS", "SCR HCO", "SCR HAC", "SCR M-W", "SCR T-Test"]
                    current_concat.index = self.correction_time_series[factor].columns
                    prop_concat_dict[factor] = current_concat

                elif december_or_santa_claus == "december":
                    current_concat = pd.concat([self.p_values_proportion_corrected_pos_stat_dec[
                                                    factor].reset_index(drop=True),
                                                self.p_values_proportion_pos_stat_dec[
                                                    factor].reset_index(drop=True),
                                                self.p_values_proportion_HCO_pos_stat_dec[
                                                    factor].reset_index(drop=True),
                                                self.p_values_proportion_HAC_pos_stat_dec[
                                                    factor].reset_index(drop=True),
                                                self.p_values_mann_whitney_proportion_pos_stat_dec[
                                                    factor].reset_index(drop=True),
                                                self.p_values_t_test_proportion_pos_stat_dec[
                                                    factor].reset_index(drop=True),
                                                ],
                                               axis=1)
                    current_concat.columns = ["SCR corrected", "SCR OLS", "SCR HCO", "SCR HAC", "SCR M-W",
                                              "SCR T-Test"]
                    current_concat.index = self.correction_time_series[factor].columns
                    prop_concat_dict[factor] = current_concat
                else:
                    raise ValueError("december_or_santa_claus wrong. Must be either 'december' or 'santa_claus'.")

            else:
                raise ValueError("pos_or_pos_stat must be either 'pos' or 'pos_stat'.")

        return prop_concat_dict

    def compute_mean_returns_across_years(self, results_sub_periods):
        # To store results
        mean_excess_returns_time_series = {
            factor: pd.DataFrame(data=np.nan, index=self.p_values_time_series[factor].index,
                                 columns=[self.p_values_time_series[factor].columns]) for factor in
            self.trimmed_factors.keys()}
        for factor in self.trimmed_factors.keys():
            x_str = list(results_sub_periods[factor].keys())
            x_datetime = [pd.to_datetime(date_str) for date_str in x_str]
            # Iterating through each date
            for i, date in enumerate(x_str):
                current_ret = results_sub_periods[factor][date][3].loc[:,"β1 (Difference between rally and non-rally days)"]
                dict_for_df = {x_datetime[i]: current_ret}

                # First date to store
                if i == 0:
                    mean_excess_returns_time_series[factor] = pd.DataFrame(dict_for_df).T
                else:
                    # For other dates, we concat
                    mean_excess_returns_time_series[factor] = pd.concat(
                        [mean_excess_returns_time_series[factor], pd.DataFrame(dict_for_df).T], axis=0)

        return mean_excess_returns_time_series

class Utilities:

    @staticmethod
    def get_year_start_end_indices(datetime_index):
        """
        Get the start and end indices of each year in a given DatetimeIndex.

        Parameters:
        ----------
        datetime_index : pd.DatetimeIndex
            The datetime index from which to extract the start and end indices
            of each year.

        Returns:
        -------
        list of tuples
            A list of tuples where each tuple contains the start and end indices
            for a year in the format (start_index, end_index).

        Example:
        -------
        date_range = pd.date_range(start="2018-01-01", end="2023-12-31", freq="D")
        get_year_start_end_indices(date_range)
        [(0, 364), (365, 729), (730, 1095), (1096, 1460), (1461, 1825), (1826, 2189)]
        """
        # Ensure the index is sorted
        datetime_index = datetime_index.sort_values()

        # Extract unique years from the index
        years = datetime_index.year.unique()
        start_end_indices = []

        for year in years:
            # Find the start index for the year
            start_idx = datetime_index.get_loc(datetime_index[datetime_index.year == year][0])
            # Find the end index for the year
            end_idx = datetime_index.get_loc(datetime_index[datetime_index.year == year][-1])
            start_end_indices.append((start_idx, end_idx))

        return start_end_indices

    @staticmethod
    def get_year_start_end_indices_from_dict(factors_dict):
        """
        Compute year start and end indices for a given factors dictionary.

        Parameters:
        ----------
        factors_dict : dict
            Dictionary where keys are factor names and values are DataFrames with datetime indices.

        Returns:
        -------
        dict
            A dictionary where each key corresponds to a factor, and the value is the list of year
            start and end indices.
        """
        return {
            factor: Utilities.get_year_start_end_indices(factor_data.index)
            for factor, factor_data in factors_dict.items()
        }

    @staticmethod
    def get_common_earliest_date(datetime_indices:list):
        """
        Find the earliest common datetime from a list of datetime indices.

        Parameters:
        ----------
        datetime_indices : list of pd.DatetimeIndex
            A list of DatetimeIndex objects to find the common earliest date.

        Returns:
        -------
        pd.Timestamp or None
            The earliest datetime that all DatetimeIndex objects have in common.
            Returns None if there is no common date.

        Example:
        -------
        index1 = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        index2 = pd.date_range("2023-06-01", "2023-12-31", freq="D")
        index3 = pd.date_range("2023-07-01", "2023-12-31", freq="D")
        get_common_earliest_date([index1, index2, index3])
        Timestamp('2023-07-01 00:00:00')
        """

        if not datetime_indices:
            return None

        # Start with the first DatetimeIndex
        common_dates = datetime_indices[0]

        # Iteratively find the intersection with subsequent indices
        for index in datetime_indices[1:]:
            common_dates = common_dates.intersection(index)

        # Return the earliest common datetime, or None if no common dates
        return common_dates.min() if not common_dates.empty else None


class SantaClausBucket():
    def __init__(self,df):
        # Santa Claus Analysis
        self.santa_claus_days = self.search_santa_claus_days(df)
        self.santa_claus_df = self.compute_santa_claus_df(df)
        self.other_df = self.compute_other_df(df)

        # December Analysis
        self.december_days = self.search_santa_claus_days(df, 21, 0)
        self.december_df = self.compute_santa_claus_df(df, santa_claus_or_december="december")
        self.december_other_df = self.compute_other_df(df, santa_claus_or_december="december")

    def compute_santa_claus_df(self,df, santa_claus_or_december:str="santa_claus"):
        # Checking inputs
        if santa_claus_or_december not in ["santa_claus", "december"]:
            raise ValueError("Wrong name of period entered. Supported: 'santa_claus' or 'december'.")

        if santa_claus_or_december == "santa_claus":
            santa_claus_days = pd.to_datetime(self.santa_claus_days)
        else:
            santa_claus_days = pd.to_datetime(self.december_days)

        return df[df.index.isin(santa_claus_days)]

    def compute_other_df(self,df, santa_claus_or_december:str="santa_claus"):
        # Checking inputs
        if santa_claus_or_december not in ["santa_claus", "december"]:
            raise ValueError("Wrong name of period entered. Supported: 'santa_claus' or 'december'.")

        if santa_claus_or_december == "santa_claus":
            santa_claus_days = pd.to_datetime(self.santa_claus_days)
        else:
            santa_claus_days = pd.to_datetime(self.december_days)

        return  df[~df.index.isin(santa_claus_days)]


    # Méthode qui récupère les jours du Santa Claus Rally pour toutes les années
    def search_santa_claus_days(self,df, n_december:int=5, n_january:int=2):
        santa_claus_days = []
        df.index = pd.to_datetime(df.index)
        years = df.index.year.unique()
        years = list(range(years.min(), years.max() + 1)) #Plus long mais nécessaire pour le decoupage par période économique

        for year in years:
            if year == years[0]:
                first_days_of_next_year = self.get_first_weekdays(year - 1, 2,df)
                santa_claus_days.extend(first_days_of_next_year)

            last_days_of_year = self.get_last_weekdays(year, n_december,df)
            first_days_of_next_year = self.get_first_weekdays(year, n_january,df)

            santa_claus_days.extend(last_days_of_year)
            santa_claus_days.extend(first_days_of_next_year)

        return tuple(santa_claus_days)

    # Méthode qui controle si le jour est un jour de la semaine (lundi =0, vendredi =4) et si ce n'est pas le 25/12 ou 01/01
    def is_weekday(self,date,year):
        if date.weekday() < 5 and date != pd.Timestamp(f'{year}-12-25') and date != pd.Timestamp(f'{year}-01-01'):
            return True
        else:
            return False

    #Méthode qui va récupérer les n derniers jours de l'année
    def get_last_weekdays(self,year, n,df):
        days = []
        date = pd.Timestamp(f'{year}-12-31')
        count =0

        # Récupére les jours ouvrés, en commençant par le 31 décembre
        while len(days) < n:
            if self.is_weekday(date,year) and date in df.index:
                days.append(date.strftime('%Y-%m-%d'))
            date -= pd.Timedelta(days=1)
            count += 1
            if count == 150: break

        return days[::-1]  # retourne les jours dans l'ordre correct

    # Méthode qui va récupérer les n premiers jours de l'année
    def get_first_weekdays(self,year, n,df):
        days = []
        date = pd.Timestamp(f'{year + 1}-01-02')
        count =0
        # Récupére les jours ouvrés en commençant par le 1er janvier
        while len(days) < n:
            if self.is_weekday(date,year) and date in df.index:
                days.append(date.strftime('%Y-%m-%d'))
            date += pd.Timedelta(days=1)
            count +=1
            if count == 150: break

        return days



class statistics():

    def __init__(self,santa_claus_df,other_df, december_df=None, december_other_df=None, extratest:bool=False):
        # Santa Claus Analysis
        self.desc_stats = self.compute_descriptive_statistics(santa_claus_df,other_df)
        self.desc_stats_numeric = self.compute_descriptive_statistics(santa_claus_df, other_df, numeric=True)

        self.test_stats = self.compute_Santa_Claus_test(santa_claus_df,other_df)
        self.test_stats_numeric = self.compute_Santa_Claus_test(santa_claus_df, other_df, numeric=True)

        self.OLS_stats = self.OLS_regression(santa_claus_df,other_df)
        self.OLS_stats_numeric = self.OLS_regression(santa_claus_df, other_df, numeric=True)

        if extratest:

            # self.bootstrap_stats = self.bootstrap(santa_claus_df, other_df, n_iterations=1000, alpha=0.05)
            # self.bootstrap_stats_numeric = self.bootstrap(santa_claus_df, other_df, n_iterations=1000, alpha=0.05,
            #                                               numeric=True)
            self.OLS_variants_stats = self.OLS_variants(santa_claus_df, other_df)
            self.OLS_variants_numeric = self.OLS_variants(santa_claus_df, other_df, numeric=True)

        # December Analysis
        if december_df is not None: # and december_other_df is not None:
            self.dec_desc_stats = self.compute_descriptive_statistics(december_df, december_other_df)
            self.dec_desc_stats_numeric = self.compute_descriptive_statistics(december_df, december_other_df, numeric=True)

            self.dec_test_stats = self.compute_Santa_Claus_test(december_df, december_other_df)
            self.dec_test_stats_numeric = self.compute_Santa_Claus_test(december_df, december_other_df, numeric=True)

            self.dec_OLS_stats = self.OLS_regression(december_df, december_other_df)
            self.dec_OLS_stats_numeric = self.OLS_regression(december_df, december_other_df, numeric=True)

            if extratest:

                # self.dec_bootstrap_stats = self.bootstrap(december_df, december_other_df, n_iterations=1000, alpha=0.05)
                # self.dec_bootstrap_stats_numeric = self.bootstrap(december_df, december_other_df, n_iterations=1000,
                #                                                   alpha=0.05, numeric=True)

                self.dec_OLS_variants = self.OLS_variants(december_df, december_other_df)
                self.dec_OLS_variants_numeric = self.OLS_variants(december_df, december_other_df, numeric=True)

    def compute_descriptive_statistics(self,santa_claus_df,other_df=None, numeric:bool=False):

        if other_df is not None :
            df = pd.concat([santa_claus_df, other_df], axis=1)
            santa_claus_columns = [f"{col} - Santa Claus Rally Days" for col in santa_claus_df.columns]
            other_columns = [f"{col} - Remaining Days" for col in other_df.columns]
            df.columns = santa_claus_columns + other_columns
        else:
            df = santa_claus_df

        stats = {
            "mean": self.compute_daily_average(df),
            "std": self.compute_standard_deviation(df),
            "min": self.compute_min(df),
            "max": self.compute_max(df),
            "range": self.compute_range(df),
            "observations": self.nb_observations(df)
        }


        formatted_stats = {}
        for key, values in stats.items():
            if key != "observations":
                formatted_stats[key] = (values * 100).map("{:.4f}%".format)
            else:
                formatted_stats[key] = values

        if numeric:
            return pd.DataFrame(stats)
        else:
            return pd.DataFrame(formatted_stats)

    def compute_Santa_Claus_test(self, santa_claus_df, other_df, numeric:bool=False):
        if other_df is None: return "None"
        results = {}

        for column in santa_claus_df.columns:
            santa_col = santa_claus_df[column].dropna()
            other_col = other_df[column].dropna()

            mean_scr = self.compute_daily_average(santa_col)
            mean_other = self.compute_daily_average(other_col)
            obs_scr = self.nb_observations(santa_col)
            obs_other = self.nb_observations(other_col)

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

        if numeric:
            return pd.DataFrame(results).T
        else:
            return result_df


    def compute_daily_average(self,df):
        return df.mean()

    def compute_standard_deviation (self, df):
        return df.std()

    def compute_min(self, df):
        return df.min()

    def compute_max(self, df):
        return df.max()

    def compute_range(self,df):
        result = df.max() - df.min()
        return result

    def nb_observations(self,df):
        return df.count()

    def T_test(self,santa_claus_df,other_df):
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

    def OLS_regression(self,santa_claus_df,other_df, numeric:bool=False):
        if other_df is None: return "None"

        results = {}
        results_numeric = {}

        santa_claus_df = santa_claus_df.copy()
        other_df = other_df.copy()

        santa_claus_df.loc[:,'SantaClausRally'] = 1  # 1 pour Santa Claus Rally
        other_df.loc[:,'SantaClausRally'] = 0  # 0 pour autres jours

        combined_df = pd.concat([santa_claus_df, other_df], ignore_index=True)

        for column in santa_claus_df.columns:
            if column == 'SantaClausRally':
                continue

            temp_df = combined_df[['SantaClausRally', column]].dropna()

            X = temp_df[['SantaClausRally']].values
            X = np.hstack([np.ones((X.shape[0], 1)), X])

            y = temp_df[column].values

            X_transpose = X.T
            beta = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)

            beta_0 = beta[0]
            beta_1 = beta[1]

            residuals = y - X.dot(beta)

            ## test for autocorrelation (ljung box)
            test_result_lb = acorr_ljungbox(residuals, lags=[1], return_df=True)
            lb_pval = test_result_lb['lb_pvalue'].iloc[0]

            ## Engle's test for heterosckedasticity
            optimal_lag, aic_results = statistics.find_optimal_lag(residuals, max_lag=1)
            arch_test_result = sm.stats.diagnostic.het_arch(residuals, nlags=optimal_lag)
            arch_test_pval = arch_test_result[1]

            ## Assigning HAC, White or no correction based on above tests
            if lb_pval<0.05 and arch_test_pval<0.05:
                correction = "HAC" # 2
                correction_numeric = 2
            if lb_pval>0.05 and arch_test_pval<0.05:
                correction = "White" # 1
                correction_numeric = 1
            if lb_pval>0.05 and arch_test_pval>0.05:
                correction = "no correction" # 0
                correction_numeric = 0
            if lb_pval<0.05 and arch_test_pval>0.05:
                correction = "HAC"
                correction_numeric = 2

            n = len(y)
            k = X.shape[1]
            vare = np.sum(residuals ** 2) / (n - k)

            varbet = vare * np.linalg.inv(X_transpose.dot(X))
            std_err = np.sqrt(np.diag(varbet))

            t_stat = beta / std_err
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - k))

            ss_total = np.sum((y - np.mean(y)) ** 2)
            ss_residual = np.sum((y - X.dot(beta)) ** 2)
            r_squared = 1 - (ss_residual / ss_total)

            f_value = (r_squared / (k - 1)) / ((1 - r_squared) / (n - k))

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
                'lb p-value': round(lb_pval, 3),
                'arch_test p-value': round(arch_test_pval, 3),
                'correction': correction
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
                'lb p-value': lb_pval,
                'arch_test p-value': arch_test_pval,
                'correction': correction_numeric
            }

        results_df = pd.DataFrame(results).T

        if numeric:
            return pd.DataFrame(results_numeric).T
        else:
            return results_df

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
            return "None"

        results = {}
        results_numeric = {}

        # Assurez-vous que nous travaillons sur une copie propre de chaque DataFrame
        santa_claus_df = santa_claus_df.copy()
        other_df = other_df.copy()

        # Ajout de la colonne 'SantaClausRally' sans déclencher l'avertissement
        santa_claus_df['SantaClausRally'] = 1  # 1 pour Santa Claus Rally
        other_df['SantaClausRally'] = 0  # 0 pour autres jours

        combined_df = pd.concat([santa_claus_df, other_df], ignore_index=True)

        # Itération sur les colonnes de santa_claus_df (sauf 'SantaClausRally')
        for column in santa_claus_df.columns:
            if column == 'SantaClausRally':
                continue

            temp_df = combined_df[['SantaClausRally', column]].dropna()
            X = temp_df[['SantaClausRally']].values
            X = np.hstack([np.ones((X.shape[0], 1)), X])  # Ajout de la constante
            y = temp_df[column].values

            # Utilisation de statsmodels pour une gestion plus avancée
            model = sm.OLS(y, X)

            # 1. Régression OLS classique (non robuste)
            results_model_classic = model.fit()  # Ajuster le modèle sans corrections robustes
            p_values_classic = results_model_classic.pvalues

            # Stockage des résultats de l'OLS classique
            results[f'{column} (OLS classique)'] = {
                'p-value (β0)': round(p_values_classic[0], 3),
                'p-value (β1)': round(p_values_classic[1], 3)
            }

            # Pour les résultats numériques de l'OLS classique
            results_numeric[f'{column} (OLS classique)'] = {
                'p-value (β0)': p_values_classic[0],
                'p-value (β1)': p_values_classic[1]
            }

            # Liste des types de correction HC à tester
            hc_types = ['HC0', 'HC1', 'HC2', 'HC3']

            # 2. Calcul des p-values pour chaque type de correction (HC)
            for hc_type in hc_types:
                # Calcul des résultats avec la correction de variance-covariance robuste
                results_model_robust = results_model_classic.get_robustcov_results(cov_type=hc_type)

                # Extraction des p-values
                p_values = results_model_robust.pvalues

                # Stockage des résultats pour chaque type HC
                results[f'{column} ({hc_type})'] = {
                    'p-value (β0)': round(p_values[0], 3),
                    'p-value (β1)': round(p_values[1], 3)
                }

                # Pour les résultats numériques de chaque type HC
                results_numeric[f'{column} ({hc_type})'] = {
                    'p-value (β0)': p_values[0],
                    'p-value (β1)': p_values[1]
                }

            # # 3. Détermination du nombre optimal de maxlags pour HAC
            # maxlags_candidates = range(1, 3)  # Tester des lags de 1 à 20, ajustez si nécessaire
            # aic_values = []
            # bic_values = []
            #
            # # Calculer AIC et BIC pour chaque valeur de maxlags
            # for maxlags in maxlags_candidates:
            #     try:
            #         results_model_hac = results_model_classic.get_robustcov_results(cov_type='HAC', maxlags=maxlags)
            #         aic_values.append(results_model_hac.aic)
            #         bic_values.append(results_model_hac.bic)
            #     except:
            #         # En cas d'erreur (par exemple si la taille de l'échantillon est trop petite pour ce maxlags), ignorer cette itération
            #         aic_values.append(np.nan)
            #         bic_values.append(np.nan)
            #
            # # Sélectionner le maxlags optimal en fonction du critère AIC ou BIC (ici AIC)
            # best_maxlags = maxlags_candidates[np.argmin(aic_values)]  # ou np.argmin(bic_values) pour BIC
            best_maxlags = 1


            # Calcul final des résultats HAC avec le maxlags optimal
            results_model_hac_optimal = results_model_classic.get_robustcov_results(cov_type='HAC',
                                                                                    maxlags=best_maxlags)

            # Extraction des p-values HAC
            p_values_hac = results_model_hac_optimal.pvalues

            # Stockage des résultats HAC
            results[f'{column} (HAC, maxlags={best_maxlags})'] = {
                'p-value (β0)': round(p_values_hac[0], 3),
                'p-value (β1)': round(p_values_hac[1], 3)
            }

            # Pour les résultats numériques HAC
            results_numeric[f'{column} (HAC, maxlags={best_maxlags})'] = {
                'p-value (β0)': p_values_hac[0],
                'p-value (β1)': p_values_hac[1]
            }

        # Conversion des résultats en DataFrame
        results_df = pd.DataFrame(results).T

        # Retourner les résultats sous forme de DataFrame numérique ou standard
        if numeric:
            return pd.DataFrame(results_numeric).T
        else:
            return results_df

    def bootstrap(self, santa_claus_df, other_df, n_iterations=1000, alpha=0.05, numeric: bool = False):
        if other_df is None:
            return "None"

        results = {}
        results_numeric = {}
        santa_claus_df.loc[:, 'SantaClausRally'] = 1
        other_df.loc[:, 'SantaClausRally'] = 0

        combined_df = pd.concat([santa_claus_df, other_df], ignore_index=True)

        for column in santa_claus_df.columns:
            if column == 'SantaClausRally':
                continue

            temp_df = combined_df[['SantaClausRally', column]].dropna()
            X = temp_df[['SantaClausRally']].values
            X = np.hstack([np.ones((X.shape[0], 1)), X])  # Ajout de la constante
            y = temp_df[column].values

            n = len(y)
            beta_samples = []

            for _ in range(n_iterations):
                # Rééchantillonnage
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

        # Conversion des résultats en DataFrame
        results_df = pd.DataFrame(results).T

        if numeric:
            return pd.DataFrame(results_numeric).T
        else:
            return results_df

class FactorsDataBase:
    def __init__(self, start_date:str = "1998-01-01", end_date:str="2023-12-31", file_name:str=r"Data\Betting Against Beta Equity Factors Daily.xlsx",
                 sheet_names=None, starting_line_data:int = 18):

        self.start_date= start_date
        self.end_date = end_date

        if sheet_names is None:
            sheet_names = ["MKT", "SMB", "HML FF", "HML Devil", "UMD"]

        self.sheet_names = sheet_names
        self.file_name = file_name
        self.starting_line_data = starting_line_data
        self.factors_non_formatted = self.load_data()
        self.factors_formatted = self.format_data()

    def load_data(self):
        factors_dict = {factor: pd.DataFrame() for factor in self.sheet_names}
        for sheet_name in self.sheet_names:
            factors_dict[sheet_name] = pd.read_excel(self.file_name, sheet_name=sheet_name, header=self.starting_line_data)

        return factors_dict

    def format_data(self):
        factors_formatted = {factor: pd.DataFrame for factor in self.sheet_names}
        for sheet_name in self.sheet_names:
            df_current = self.factors_non_formatted[sheet_name]
            # Set index as datetime
            df_current.set_index(pd.to_datetime(df_current.iloc[:,0], format="%m/%d/%Y"), inplace=True)
            # Drop "DATE" column
            df_current.drop(columns="DATE", inplace=True)
            # Aligning all factors to the first all non-missing values
            first_valid_index = df_current.dropna(how="any").index[0]
            df_current = df_current.loc[first_valid_index:]
            # Deleting 1997 and 2024 because not full years
            start_date = pd.to_datetime(self.start_date)
            end_date = pd.to_datetime(self.end_date)
            df_current = df_current.loc[start_date:end_date,:]

            factors_formatted[sheet_name] = df_current

        return factors_formatted