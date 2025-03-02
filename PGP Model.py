import time
start_time = time.time()
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from pypfopt import risk_models
from pypfopt import EfficientFrontier
from pypfopt import expected_returns
import matplotlib.pyplot as plt
import seaborn as sns

df_1 = pd.read_excel("Carbon Efficient By Location.xlsx")
df_1.set_index('Date', inplace=True)

log_returns = np.log(df_1 / df_1.shift(1))
log_returns = log_returns.dropna()

cov_matrix = risk_models.CovarianceShrinkage(df_1, frequency=52).ledoit_wolf()
#cov_matrix = risk_models.sample_cov(df_1, frequency=52)
exp_returns = expected_returns.mean_historical_return(df_1, compounding=True, frequency=52, log_returns=True)
n_assets = len(exp_returns)
ef = EfficientFrontier(exp_returns, cov_matrix, weight_bounds=(0, 1))
ef.min_volatility()
weights_min_vol2 = ef.clean_weights().values()
weights_min_vol = np.array(list(weights_min_vol2))
ExpRet_Vol_SR= ef.portfolio_performance() ## Returns the portfolio performance: expected returns, volatility and Sharpe Ratio
V_star = ExpRet_Vol_SR[1]

# Returns Statistics
stdev_returns = log_returns.std()
skew_returns = log_returns.skew()
kurt_returns = log_returns.kurt()

jarque_bera_results = {}
# Loop through each column and perform the Jarque-Bera test
for column in log_returns.columns:
        jb_value, p_value = stats.jarque_bera(log_returns[column].dropna())
        jarque_bera_results[column] = {'Jarque-Bera Value': jb_value, 'p-value': p_value}

JBTest_df = pd.DataFrame(jarque_bera_results).T

returns_stats_df = pd.DataFrame({
    'Index Name': log_returns.columns,
    'Mean Returns': exp_returns.values,
    'Standard Deviation': stdev_returns.values,
    'Skewness': skew_returns.values,
    'Kurtosis': kurt_returns.values,
})

returns_stats_plusJBTest_df = pd.concat([returns_stats_df.set_index('Index Name'), JBTest_df], axis=1).reset_index()
returns_stats_plusJBTest_df.rename(columns={'index': 'Index Name'}, inplace=True)

# Display the DataFrame
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
#print(returns_stats_plusJBTest_df)
#returns_stats_plusJBTest_df.to_excel('Returns Stats.xlsx', index=False, engine='openpyxl')



## Calculate R_star

def optimize_for_R_star(exp_returns):
    n_assets = len(exp_returns)
    def objective_function(x):
        return -np.dot(exp_returns, x)
    constraints = ({'type': 'eq', 'fun': lambda x: 1- np.sum(x)})
    bounds = [(0, 1) for asset in range(n_assets)]
    initial_weights =  np.random.dirichlet(np.ones(5), size=1)[0]
    solution = minimize(objective_function, initial_weights, method= 'SLSQP', bounds=bounds, constraints=constraints)
    return solution.x, -solution.fun

optimized_weights, optimized_return = optimize_for_R_star(exp_returns)
weights_max_return = optimized_weights


## Calculate S_star

def optimize_for_S_star(log_returns):
    n_assets = len(log_returns.columns)
    n_obs = len(log_returns)
    co_skew_matrix = np.zeros((n_assets, n_assets, n_assets))

    for i in range(n_assets):
        for j in range(n_assets):
            for k in range(n_assets):
                co_skew_matrix[i, j, k] = (
                                                  np.sum(
                                                      (log_returns.iloc[:, i] - log_returns.iloc[:, i].mean()) *
                                                      (log_returns.iloc[:, j] - log_returns.iloc[:, j].mean()) *
                                                      (log_returns.iloc[:, k] - log_returns.iloc[:, k].mean())
                                                  ) / n_obs
                                          ) / (log_returns.iloc[:, i].std() * log_returns.iloc[:,
                                                                              j].std() * log_returns.iloc[:, k].std())

    def negative_portfolio_skewness(weights):
        portfolio_skewness = 0.0
        for i in range(n_assets):
            for j in range(n_assets):
                for k in range(n_assets):
                    portfolio_skewness += (
                            weights[i] * weights[j] * weights[k] * co_skew_matrix[i, j, k]
                    )
        return -portfolio_skewness

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = [(0, 1) for asset in range(n_assets)]
    initial_weights =  np.random.dirichlet(np.ones(5), size=1)[0]
    solution = minimize(negative_portfolio_skewness, initial_weights, method='SLSQP', bounds=bounds,
                        constraints=constraints)

    return solution.x, -negative_portfolio_skewness(solution.x), co_skew_matrix

optimal_weights, optimal_skewness, co_skew_matrix = optimize_for_S_star(log_returns)
weights_max_skew = optimal_weights

## Calculate K_star

def optimize_for_K_star(log_returns):
    n_assets = len(log_returns.columns)
    n_obs = len(log_returns)
    co_kurt_matrix = np.zeros((n_assets, n_assets, n_assets, n_assets))

    for i in range(n_assets):
        for j in range(n_assets):
            for k in range(n_assets):
                for l in range(n_assets):
                    co_kurt_matrix[i, j, k, l] = (
                                                         np.sum(
                                                             (log_returns.iloc[:, i] - log_returns.iloc[:, i].mean()) *
                                                             (log_returns.iloc[:, j] - log_returns.iloc[:, j].mean()) *
                                                             (log_returns.iloc[:, k] - log_returns.iloc[:, k].mean()) *
                                                             (log_returns.iloc[:, l] - log_returns.iloc[:, l].mean())
                                                         ) / n_obs
                                                 ) / (log_returns.iloc[:, i].std() * log_returns.iloc[:,
                                                                                     j].std() * log_returns.iloc[:,
                                                                                                k].std() * log_returns.iloc[
                                                                                                           :, l].std())

    def portfolio_kurtosis(weights):
        portfolio_kurt = 0.0
        for i in range(n_assets):
            for j in range(n_assets):
                for k in range(n_assets):
                    for l in range(n_assets):
                        portfolio_kurt += (
                                weights[i] * weights[j] * weights[k] * weights[l] * co_kurt_matrix[i, j, k, l]
                        )
        return portfolio_kurt

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = [(0, 1) for asset in range(n_assets)]
    initial_weights =  np.random.dirichlet(np.ones(5), size=1)[0]
    solution = minimize(portfolio_kurtosis, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    return solution.x, portfolio_kurtosis(solution.x),co_kurt_matrix
optimal_weights, optimal_kurtosis, co_kurt_matrix = optimize_for_K_star(log_returns)
weights_min_kurt = optimal_weights


print(weights_max_return)
print(weights_min_vol)
print(weights_max_skew)
print(weights_min_kurt)


# Asset names
assets = ["United States", "Europe", "Asia Pacific", "China", "Latin America"]

# Ensure weights are in a list format
weights_max_return_list = list(weights_max_return)
weights_min_vol_list = list(weights_min_vol)
weights_max_skew_list = list(weights_max_skew)
weights_min_kurt_list = list(weights_min_kurt)


df_2 = pd.DataFrame({
    'Assets': assets,
    'Maximum Return': weights_max_return_list,
    'Minimum Variance': weights_min_vol_list,
    'Maximum Skewness': weights_max_skew_list,
    'Minimum Kurtosis': weights_min_kurt_list,
})

sns.set_style("whitegrid")


fig, axs = plt.subplots(2, 2, figsize=(15, 10))
axs = axs.flatten()

# Portfolio types
portfolio_types = df_2.columns[1:]

# Loop through each subplot and create the bar plots
for i, portfolio_type in enumerate(portfolio_types):
    ax = axs[i]
    sns.barplot(x="Assets", y=portfolio_type, data=df_2, ax=ax, palette="muted")
    axs[i].set_title(f'{portfolio_type} Portfolio Optimal Weights')
    axs[i].set_ylabel('Weights (%)')
    axs[i].set_xlabel('')
    axs[i].tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability
# Add labels on top of the bars
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width() / 2., p.get_height(),
                '{:1.2f}%'.format(p.get_height()*100),
                fontsize=10, color='black', ha='center', va='bottom')

# Adjust layout for better visualization
#plt.tight_layout()
# Show the plots
#plt.show()

R_star = optimize_for_R_star(exp_returns)[1]
S_star = optimize_for_S_star(log_returns)[1]
K_star = optimize_for_K_star(log_returns)[1]

print("R_star:", R_star)
print("V_star:", V_star)
print("S_star:", S_star)
print("K_star:", K_star)

_, S_star, co_skew_matrix = optimize_for_S_star(log_returns)
_, K_star, co_kurt_matrix = optimize_for_K_star(log_returns)

## Feed the aspired levels to the PGP functon

# Objective function for P6
def objective(x, R_star, V_star, S_star, K_star, lambda_1, lambda_2, lambda_3, lambda_4):
    d1 = R_star - np.dot(exp_returns.T, x)
    d2 = np.dot(x.T, cov_matrix @ x) - V_star

    portfolio_skewness = 0
    for i in range(n_assets):
        for j in range(n_assets):
            for k in range(n_assets):
                portfolio_skewness += co_skew_matrix[i, j, k] * x[i] * x[j] * x[k]
    d3 = S_star - portfolio_skewness

    portfolio_kurtosis = 0
    for i in range(n_assets):
        for j in range(n_assets):
            for k in range(n_assets):
                for l in range(n_assets):
                    portfolio_kurtosis += co_kurt_matrix[i, j, k, l] * x[i] * x[j] * x[k] * x[l]
    d4 = portfolio_kurtosis - K_star

    return abs(d1 / R_star)**lambda_1 + abs(d2 / V_star)**lambda_2 + abs(d3 / S_star)**lambda_3 + abs(d4 / K_star)**lambda_4

# Constraints
def constraint_weights_sum(x):
    return np.sum(x) - 1

# Initial guess
x0 = np.random.dirichlet(np.ones(5), size=1)[0]

# Constraints
con5 = {'type': 'eq', 'fun': constraint_weights_sum}

# Lambda values
lambda_1_values = [1, 2, 3]
lambda_2_values = [1, 2, 3]
lambda_3_values = [0, 1, 2, 3]
lambda_4_values = [0, 1, 2, 3]

# Results storage
results = []

# Optimization
bounds = [(0.05, 0.75) for _ in range(n_assets)]
for lambda_1 in lambda_1_values:
    for lambda_2 in lambda_2_values:
        for lambda_3 in lambda_3_values:
            for lambda_4 in lambda_4_values:
                solution = minimize(objective, x0, args=(R_star, V_star, S_star, K_star, lambda_1, lambda_2, lambda_3, lambda_4), constraints=[con5],
                bounds=bounds,
                method='SLSQP'
                                    )
                weights_opt = solution.x
                results_dict = {
                    'lambda_1': lambda_1,
                    'lambda_2': lambda_2,
                    'lambda_3': lambda_3,
                    'lambda_4': lambda_4,
                }
                for asset_name, weight in zip(df_1.columns, weights_opt):
                    results_dict[asset_name] = weight
                results.append(results_dict)

# Convert results to DataFrame for easier analysis
results_df = pd.DataFrame(results)

results_df.to_excel('results.xlsx', index=False, engine='openpyxl')

stats_df = pd.DataFrame(columns=['Mean', 'Variance', 'Skewness', 'Kurtosis'])

# Compute statistics for each portfolio
for index, row in results_df.iterrows():
    weights = row[df_1.columns].values  # Extracting weights
    port_mean = np.dot(exp_returns, weights)
    port_var = np.dot(weights.T, np.dot(cov_matrix, weights))

    port_skew = 0
    for i in range(n_assets):
        for j in range(n_assets):
            for k in range(n_assets):
                port_skew += weights[i] * weights[j] * weights[k] * co_skew_matrix[i, j, k]

    port_kurt = 0
    for i in range(n_assets):
        for j in range(n_assets):
            for k in range(n_assets):
                for l in range(n_assets):
                    port_kurt += weights[i] * weights[j] * weights[k] * weights[l] * co_kurt_matrix[i, j, k, l]

    curr_stats = pd.DataFrame({
        'Mean': [port_mean],
        'Variance': [port_var],
        'Skewness': [port_skew],
        'Kurtosis': [port_kurt]
    }, index=[index])

    stats_df = pd.concat([stats_df, curr_stats], ignore_index=True)

final_df = pd.concat([results_df, stats_df], axis=1)

## CALCULATE THE OMEGA RATIO FOR ALL THE PORTFOLIO

# Extract asset returns from log_returns
asset_returns = log_returns.iloc[:, 0:]
portfolio_weights = final_df.iloc[:, 4:9].values
portfolio_returns = final_df.iloc[:, 9].values
# risk-free rate
rf = 0.025

# Step 1: Lower Partial Moment (LPM) Calculation for each asset
LPM = []
for column in asset_returns.columns:
    y = asset_returns[column][asset_returns[column] - rf < 0]
    m = len(y)
    total = 0.0
    for i in np.arange(m):
        total += (y.iloc[i] - rf)**2
    LPM.append(total / (m - 1))

# Step 2: Weighted Average LPM Calculation for each portfolio
Weighted_Avg_LPM = np.dot(portfolio_weights, np.array(LPM))
# Step 3: Omega Ratio Calculation for each portfolio
ms_omega_ratio = (portfolio_returns - rf) / Weighted_Avg_LPM

final_df['Omega Ratio'] = ms_omega_ratio




## CALCULATE THE STUTZER INDEX FOR EACH PORTFOLIO

benchmark_data = pd.read_excel("Stutzer Benchmark.xlsx")
benchmark_data.set_index('Date', inplace=True)

benchmark_returns = benchmark_data['Benchmark'].values

def calculate_portfolio_returns(weights, log_returns):
    portfolio_returns = np.dot(log_returns, weights)
    return portfolio_returns

def stutzer(portfolio_returns, benchmark_returns, no_of_iterations):
    try:
        n = len(portfolio_returns)
        m = len(benchmark_returns)
        if n > 0 and n == m:
            largest_d_so_far = float('-inf')
            best_gamma_so_far = None
            data2 = np.array(portfolio_returns) / 100
            benchmark_data2 = np.array(benchmark_returns) / 100
            O = np.cumprod(1 + data2)
            p = np.cumprod(1 + benchmark_data2)
            for gamma in np.arange(0, no_of_iterations + 1, 5):
                Q = (-1) * (O / p) ** (-gamma)
                r = np.cumsum(Q) / (np.arange(n) + 1)
                s = -np.log(-r) / (np.arange(n) + 1)
                d = s[-1]
                if d > largest_d_so_far:
                    largest_d_so_far = d
                    best_gamma_so_far = gamma
            return 100 * largest_d_so_far
        else:
            print("Length mismatch or zero length detected.")
            return "undefined"
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return "undefined"

# Number of iterations for Stutzer function
no_of_iterations = 100
# Calculating the Stutzer Index for each portfolio
stutzer_indexes = []

for index, row in final_df.iterrows():
    weights = row[4:9].values
    portfolio_returns = calculate_portfolio_returns(weights, log_returns)
    stutzer_index = stutzer(portfolio_returns, benchmark_returns, no_of_iterations)
    stutzer_indexes.append(stutzer_index)

# Adding Stutzer Indexes to the final_df
final_df['Stutzer Index'] = stutzer_indexes



end_time = time.time()
final_df.to_excel(f'final_results_{end_time:.2f}.xlsx', index=False, engine='openpyxl')
end_time = time.time()
run_time = end_time - start_time
print(f"The code ran in {run_time:.2f} seconds.")
print("The code has finished running.")