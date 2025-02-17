import pandas as pd
from Analysefactorielle import SantaClausFactorAnalysis

### Extension equity factors
## Santa Claus and December Factor Analysis - Overall years
pd.set_option("display.max_columns", 30)
print("---------------------------------------------------------------------------------------------------------------")
print("-------------------------- Santa Claus and December Factor Analysis - Overall years -------------------------- ")
print("---------------------------------------------------------------------------------------------------------------")
santa_claus_factor_analysis = SantaClausFactorAnalysis() # 8min30 to run to get results. All we need are stored as attributes.

# Here I will print evidence of the santa claus effect for the overall period, for each factor
print("--------------------------------------------")
print("------------ SRC - Overall years ---------- ")
print("--------------------------------------------")
for factor in santa_claus_factor_analysis.factors.keys():
    print(f"SRC for the overall period and for factor:{factor}")
    print((santa_claus_factor_analysis.p_values_time_series_corrected_overall[factor]<0.05).T)
    prop=(santa_claus_factor_analysis.p_values_time_series_corrected_overall[factor]<0.05).sum()/santa_claus_factor_analysis.p_values_time_series_corrected_overall[factor].shape[0]
    print(f"proportion of countries exhibiting significance (both positive and negative) of the SCR: {prop}")
    print(f"mean daily return (%) difference between SCR days and non-SCR days: {santa_claus_factor_analysis.results_overall[factor][3]["β1 (Difference between rally and non-rally days)"].mean()*100}")
    print(
        f"median daily return (%) difference between SCR days and non-SCR days: {santa_claus_factor_analysis.results_overall[factor][3]["β1 (Difference between rally and non-rally days)"].median() * 100}")
    print(
        f"min daily return (%) difference between SCR days and non-SCR days: {santa_claus_factor_analysis.results_overall[factor][3]["β1 (Difference between rally and non-rally days)"].min() * 100}")
    print(
        f"max daily return (%) difference between SCR days and non-SCR days: {santa_claus_factor_analysis.results_overall[factor][3]["β1 (Difference between rally and non-rally days)"].max() * 100}")
    print(" \n ")

# Here I will print evidence of the december effect for the overall period, for each factor
print("--------------------------------------------")
print("------------ DR - Overall years ----------- ")
print("--------------------------------------------")

for factor in santa_claus_factor_analysis.factors.keys():
    print(f"DR for the overall period and for factor:{factor}")
    print((santa_claus_factor_analysis.p_values_time_series_corrected_overall_december[factor]<0.05).T)
    prop=(santa_claus_factor_analysis.p_values_time_series_corrected_overall_december[factor]<0.05).sum()/santa_claus_factor_analysis.p_values_time_series_corrected_overall_december[factor].shape[0]
    print(f"proportion of countries exhibiting significance (both positive and negative) of the DR: {prop}")
    print(f"mean daily return (%) difference between DR days and non-DR days: {santa_claus_factor_analysis.results_overall_december[factor][3]["β1 (Difference between rally and non-rally days)"].mean()*100}")
    print(
        f"median daily return (%) difference between DR days and non-DR days: {santa_claus_factor_analysis.results_overall_december[factor][3]["β1 (Difference between rally and non-rally days)"].median() * 100}")
    print(
        f"min daily return (%) difference between DR days and non-DR days: {santa_claus_factor_analysis.results_overall_december[factor][3]["β1 (Difference between rally and non-rally days)"].min() * 100}")
    print(
        f"max daily return (%) difference between DR days and non-DR days: {santa_claus_factor_analysis.results_overall_december[factor][3]["β1 (Difference between rally and non-rally days)"].max() * 100}")
    print(" \n ")

## Santa Claus and December Factor Analysis - Sub periods (year by year)
print("---------------------------------------------------------------------------------------------------------------")
print("------------------- Santa Claus and December Factor Analysis - Sub periods (year by year) -------------------- ")
print("---------------------------------------------------------------------------------------------------------------")

print("----------------------------")
print("---- Santa Claus Rally ---- ")
print("----------------------------")

# Mean excess returns
for factor in santa_claus_factor_analysis.factors.keys():
    print(f"mean excess return (%) for factor: {factor} across years and by country")
    print(santa_claus_factor_analysis.mean_excess_ret[factor].mean()*100)

    print(f"mean excess return (%) for factor: {factor} across years and across countries")
    print(santa_claus_factor_analysis.mean_excess_ret[factor].mean().mean() * 100)
    print(f"median: {santa_claus_factor_analysis.mean_excess_ret[factor].mean().median() * 100}")
    print(
        f"country with the lowest mean excess return (%) for factor: {factor} across years is {santa_claus_factor_analysis.mean_excess_ret[factor].mean().idxmin()}")
    print(f"min: {santa_claus_factor_analysis.mean_excess_ret[factor].mean().min() * 100}")
    print(
        f"country with the highest mean excess return (%) for factor: {factor} across years is {santa_claus_factor_analysis.mean_excess_ret[factor].mean().idxmax()}")
    print(santa_claus_factor_analysis.mean_excess_ret[factor].mean().max() * 100)

# Santa Claus - Proportion of positive year
print("--------------------------------------------")
print("---- SRC - Proportion of positive year ---- ")
print("--------------------------------------------")
for factor in santa_claus_factor_analysis.factors.keys():
    avg = santa_claus_factor_analysis.consolidated_proportion_pos[factor]["SCR corrected"].mean()
    med = santa_claus_factor_analysis.consolidated_proportion_pos[factor]["SCR corrected"].median()
    mini = santa_claus_factor_analysis.consolidated_proportion_pos[factor]["SCR corrected"].min()
    maxi = santa_claus_factor_analysis.consolidated_proportion_pos[factor]["SCR corrected"].max()
    print(f"Proportion of years with positive beta 1 (mean daily SCR - mean daily non-SCR) across countries and for factor:{factor}")
    print(f"average proportion of positive years across countries: {avg}")
    print(f"med proportion of positive years across countries: {med}")
    print(f"min proportion of positive years across countries: {mini}")
    print(f"max proportion of positive years across countries: {maxi}")
    print(santa_claus_factor_analysis.consolidated_proportion_pos[factor]["SCR corrected"].sort_values(ascending=False))

# Proportion of corrected positive significant years
print("-------------------------------------------------------------------")
print("---- SRC - Proportion of corrected positive significant years ---- ")
print("-------------------------------------------------------------------")
for factor in santa_claus_factor_analysis.factors.keys():
    avg = santa_claus_factor_analysis.consolidated_proportion_pos_stat[factor]["SCR corrected"].mean()
    med = santa_claus_factor_analysis.consolidated_proportion_pos_stat[factor]["SCR corrected"].median()
    mini = santa_claus_factor_analysis.consolidated_proportion_pos_stat[factor]["SCR corrected"].min()
    maxi = santa_claus_factor_analysis.consolidated_proportion_pos_stat[factor]["SCR corrected"].max()
    print(f"Proportion of years with significant (corrected) positive beta 1 (mean daily SCR - mean daily non-SCR) across countries and for factor:{factor}")
    print(f"average proportion of positive significant (corrected) years across countries: {avg}")
    print(f"med proportion of positive significant (corrected) years across countries: {med}")
    print(f"min proportion of positive significant (corrected) years across countries: {mini}")
    print(f"max proportion of positive significant (corrected) years across countries: {maxi}")
    print(santa_claus_factor_analysis.consolidated_proportion_pos_stat[factor]["SCR corrected"].sort_values(ascending=False))

# Difference of proportion between positive and positive significant years
print("---------------------------------------------------------------")
print("---- SRC - Difference of proportion btw pos and sign. pos ---- ")
print("---------------------------------------------------------------")
for factor in santa_claus_factor_analysis.factors.keys():
    avg = santa_claus_factor_analysis.consolidated_proportion_pos[factor]["SCR corrected"].mean() - santa_claus_factor_analysis.consolidated_proportion_pos_stat[factor]["SCR corrected"].mean()
    med = santa_claus_factor_analysis.consolidated_proportion_pos[factor]["SCR corrected"].median() - santa_claus_factor_analysis.consolidated_proportion_pos_stat[factor]["SCR corrected"].median()
    mini = santa_claus_factor_analysis.consolidated_proportion_pos[factor]["SCR corrected"].min() - santa_claus_factor_analysis.consolidated_proportion_pos_stat[factor]["SCR corrected"].min()
    maxi = santa_claus_factor_analysis.consolidated_proportion_pos[factor]["SCR corrected"].max() - santa_claus_factor_analysis.consolidated_proportion_pos_stat[factor]["SCR corrected"].max()
    print(f" Difference of proportion between positive and positive significant (corrected) years for factor:{factor}")
    print(f"average proportion difference between positive and positive significant (corrected) years across countries: {avg}")
    print(f"med proportion difference between positive and positive significant (corrected) years across countries: {med}")
    print(f"min proportion difference between positive and positive significant (corrected) years across countries: {mini}")
    print(f"max proportion difference between positive and positive significant (corrected) years across countries: {maxi}")
    print((santa_claus_factor_analysis.consolidated_proportion_pos[factor]["SCR corrected"] - santa_claus_factor_analysis.consolidated_proportion_pos_stat[factor]["SCR corrected"]).sort_values(ascending=False))


print("----------------------------")
print("------ December Rally ----- ")
print("----------------------------")

# Mean excess returns
for factor in santa_claus_factor_analysis.factors.keys():
    print(f"mean excess return (%) for factor: {factor} across years and by country")
    print(santa_claus_factor_analysis.mean_excess_ret_dec[factor].mean()*100)

    print(f"mean excess return (%) for factor: {factor} across years and across countries")
    print(santa_claus_factor_analysis.mean_excess_ret_dec[factor].mean().mean() * 100)
    print(f"median: {santa_claus_factor_analysis.mean_excess_ret_dec[factor].mean().median() * 100}")
    print(
        f"country with the lowest mean excess return (%) for factor: {factor} across years is {santa_claus_factor_analysis.mean_excess_ret_dec[factor].mean().idxmin()}")
    print(f"min: {santa_claus_factor_analysis.mean_excess_ret_dec[factor].mean().min() * 100}")
    print(
        f"country with the highest mean excess return (%) for factor: {factor} across years is {santa_claus_factor_analysis.mean_excess_ret_dec[factor].mean().idxmax()}")
    print(santa_claus_factor_analysis.mean_excess_ret_dec[factor].mean().max() * 100)

# December - Proportion of positive year
print("--------------------------------------------")
print("----- DR - Proportion of positive year ---- ")
print("--------------------------------------------")
for factor in santa_claus_factor_analysis.factors.keys():
    avg = santa_claus_factor_analysis.consolidated_proportion_pos_dec[factor]["SCR corrected"].mean()
    med = santa_claus_factor_analysis.consolidated_proportion_pos_dec[factor]["SCR corrected"].median()
    mini = santa_claus_factor_analysis.consolidated_proportion_pos_dec[factor]["SCR corrected"].min()
    maxi = santa_claus_factor_analysis.consolidated_proportion_pos_dec[factor]["SCR corrected"].max()
    print(f"Proportion of years with positive beta 1 (mean daily DR - mean daily non-DR) across countries and for factor:{factor}")
    print(f"average proportion of positive years across countries: {avg}")
    print(f"med proportion of positive years across countries: {med}")
    print(f"min proportion of positive years across countries: {mini}")
    print(f"max proportion of positive years across countries: {maxi}")
    print(santa_claus_factor_analysis.consolidated_proportion_pos_dec[factor]["SCR corrected"].sort_values(ascending=False))

# Proportion of corrected positive significant years
print("-------------------------------------------------------------------")
print("---- DR - Proportion of corrected positive significant years ---- ")
print("-------------------------------------------------------------------")
for factor in santa_claus_factor_analysis.factors.keys():
    avg = santa_claus_factor_analysis.consolidated_proportion_pos_stat_dec[factor]["SCR corrected"].mean()
    med = santa_claus_factor_analysis.consolidated_proportion_pos_stat_dec[factor]["SCR corrected"].median()
    mini = santa_claus_factor_analysis.consolidated_proportion_pos_stat_dec[factor]["SCR corrected"].min()
    maxi = santa_claus_factor_analysis.consolidated_proportion_pos_stat_dec[factor]["SCR corrected"].max()
    print(f"Proportion of years with significant (corrected) positive beta 1 (mean daily DR - mean daily non-DR) across countries and for factor:{factor}")
    print(f"average proportion of positive significant (corrected) years across countries: {avg}")
    print(f"med proportion of positive significant (corrected) years across countries: {med}")
    print(f"min proportion of positive significant (corrected) years across countries: {mini}")
    print(f"max proportion of positive significant (corrected) years across countries: {maxi}")
    print(santa_claus_factor_analysis.consolidated_proportion_pos_stat_dec[factor]["SCR corrected"].sort_values(ascending=False))

# Difference of proportion between positive and positive significant years
print("---------------------------------------------------------------")
print("---- DR - Difference of proportion btw pos and sign. pos ----- ")
print("---------------------------------------------------------------")

for factor in santa_claus_factor_analysis.factors.keys():
    avg = santa_claus_factor_analysis.consolidated_proportion_pos_dec[factor]["SCR corrected"].mean() - santa_claus_factor_analysis.consolidated_proportion_pos_stat_dec[factor]["SCR corrected"].mean()
    med = santa_claus_factor_analysis.consolidated_proportion_pos_dec[factor]["SCR corrected"].median() - santa_claus_factor_analysis.consolidated_proportion_pos_stat_dec[factor]["SCR corrected"].median()
    mini = santa_claus_factor_analysis.consolidated_proportion_pos_dec[factor]["SCR corrected"].min() - santa_claus_factor_analysis.consolidated_proportion_pos_stat_dec[factor]["SCR corrected"].min()
    maxi = santa_claus_factor_analysis.consolidated_proportion_pos_dec[factor]["SCR corrected"].max() - santa_claus_factor_analysis.consolidated_proportion_pos_stat_dec[factor]["SCR corrected"].max()
    print(f" Difference of proportion between positive and positive significant (corrected) years for factor:{factor}")
    print(f"average proportion difference between positive and positive significant (corrected) years across countries: {avg}")
    print(f"med proportion difference between positive and positive significant (corrected) years across countries: {med}")
    print(f"min proportion difference between positive and positive significant (corrected) years across countries: {mini}")
    print(f"max proportion difference between positive and positive significant (corrected) years across countries: {maxi}")
    print((santa_claus_factor_analysis.consolidated_proportion_pos_dec[factor]["SCR corrected"] - santa_claus_factor_analysis.consolidated_proportion_pos_stat_dec[factor]["SCR corrected"]).sort_values(ascending=False))
