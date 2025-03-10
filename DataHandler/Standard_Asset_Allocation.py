import os
import pandas as pd
import numpy as np
import cvxpy as cp

def calculate_expected_returns(returns_df, window=120):
    """
    Calculate the expected monthly returns for each firm as the arithmetic average
    of the monthly returns over the last 'window' months.
    
    Parameters:
        returns_df (pd.DataFrame): DataFrame containing monthly returns data for each firm.
                                   Expected structure: first two columns are metadata (e.g., 'Name', 'ISIN')
                                   and the remaining columns (sorted chronologically) are monthly returns.
        window (int): Number of months to use for the calculation (default is 120).
    
    Returns:
        expected_returns (pd.Series): Series containing the computed expected return for each firm.
    """
    numeric_data = returns_df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
    if numeric_data.shape[1] < window:
        window = numeric_data.shape[1]
    expected_returns = numeric_data.iloc[:, -window:].mean(axis=1)
    return expected_returns

def compute_covariance_matrix(returns_df, window=120):
    """
    Compute the covariance matrix of monthly returns for each firm using the last 'window' months,
    following the formula:
    
      Σ = (1/τ) * Σ_{k=0}^{τ-1} (R_{t-k} - μ)(R_{t-k} - μ)'
    
    where μ is the expected return computed over the last 'window' months.
    
    This function reuses calculate_expected_returns to avoid duplicate computations.
    
    Parameters:
        returns_df (pd.DataFrame): DataFrame containing monthly returns data with metadata in the first two columns.
        window (int): Number of months to use for estimation (default is 120).
    
    Returns:
        cov_matrix (pd.DataFrame): Covariance matrix (symmetric) of the firms' returns. The index and columns
                                   are set using the firm identifiers from the first metadata column.
    """
    numeric_data = returns_df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
    if numeric_data.shape[1] < window:
        window = numeric_data.shape[1]
    window_data = numeric_data.iloc[:, -window:]
    
    # Get expected returns over the window
    mu = calculate_expected_returns(returns_df, window=window)
    diff = window_data.sub(mu, axis=0)
    Sigma_np = diff.dot(diff.T) / window
    
    identifiers = returns_df.iloc[:, 0]
    cov_matrix = pd.DataFrame(Sigma_np, index=identifiers, columns=identifiers)
    return cov_matrix

def optimize_portfolio(Sigma):
    """
    Solve the minimum variance portfolio optimization problem:
    
      minimize    α' Σ α
      subject to  sum(α) = 1 and α_i >= 0 for all i.
      
    The covariance matrix is forced to be symmetric.
    
    Parameters:
        Sigma (pd.DataFrame or np.array): Covariance matrix of firm returns.
    
    Returns:
        optimal_weights: Optimal portfolio weights as a pandas Series (if Sigma is a DataFrame) or a NumPy array.
    """
    n = Sigma.shape[0]
    if isinstance(Sigma, pd.DataFrame):
        Sigma_np = Sigma.to_numpy()
    else:
        Sigma_np = np.array(Sigma)
        
    # Force symmetry to avoid numerical issues.
    Sigma_np = (Sigma_np + Sigma_np.T) / 2

    alpha = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(alpha, Sigma_np))
    constraints = [cp.sum(alpha) == 1, alpha >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    if alpha.value is None:
        raise ValueError("Optimization did not converge.")
    
    weights = alpha.value
    if isinstance(Sigma, pd.DataFrame):
        optimal_weights = pd.Series(weights, index=Sigma.index)
    else:
        optimal_weights = weights
    return optimal_weights

def compute_rolling_optimal_weights(returns_df, rebalancing_dates, window=120):
    """
    Compute optimal portfolio weights using a rolling window approach.
    
    For each rebalancing date (ISO string), the function extracts the last 'window' months of
    return data (from returns_df) that occur on or before the rebalancing date, computes the
    covariance matrix (using the method above), and calls optimize_portfolio to obtain the
    minimum variance (long-only) portfolio weights.
    
    Parameters:
        returns_df (pd.DataFrame): DataFrame with monthly returns (metadata in the first two columns).
                                   Monthly return columns must be in ISO format ("YYYY-MM-DD") sorted chronologically.
        rebalancing_dates (list): List of rebalancing dates as ISO strings (e.g., "2013-12-31").
        window (int): Number of months to use for estimation (default is 120).
    
    Returns:
        weights_dict (dict): Dictionary mapping each rebalancing date to its optimal weight vector (pandas Series).
    """
    time_cols = returns_df.columns[2:]
    dt_cols = pd.to_datetime(time_cols, format='%Y-%m-%d', errors='coerce')
    numeric_data = returns_df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
    weights_dict = {}
    
    for r_date in rebalancing_dates:
        r_date_ts = pd.to_datetime(r_date, format='%Y-%m-%d', errors='coerce')
        mask = dt_cols <= r_date_ts
        available_cols = numeric_data.columns[mask]
        
        if len(available_cols) < window:
            print(f"Not enough data for rebalancing date {r_date}: required {window} months, available {len(available_cols)}. Skipping.")
            continue
        
        window_cols = available_cols[-window:]
        window_data = numeric_data[window_cols]
        # Compute covariance matrix using the outer-product averaging method.
        diff = window_data.sub(window_data.mean(axis=1), axis=0)
        Sigma_np = diff.dot(diff.T) / window
        identifiers = returns_df.iloc[:, 0]
        cov_matrix = pd.DataFrame(Sigma_np, index=identifiers, columns=identifiers)
        
        weights = optimize_portfolio(cov_matrix)
        weights_dict[r_date] = weights
        
    return weights_dict

def compute_ex_post_returns(returns_df, weights_dict, horizon=12):
    """
    Compute the ex-post monthly portfolio returns for the following 'horizon' months for each rebalancing date.
    
    For each rebalancing date, the function identifies the first 'horizon' months of return data after that date
    (columns in ISO format) and computes the portfolio return as the weighted sum of individual firm returns.
    
    Parameters:
        returns_df (pd.DataFrame): DataFrame with monthly returns (metadata in first two columns).
        weights_dict (dict): Dictionary with keys as rebalancing dates and values as optimal weight Series.
        horizon (int): Number of months to compute ex-post returns for each rebalancing date (default is 12).
    
    Returns:
        ex_post_dict (dict): Dictionary mapping each rebalancing date to a pandas Series of ex-post portfolio returns.
    """
    time_cols = returns_df.columns[2:]
    dt_cols = pd.to_datetime(time_cols, format='%Y-%m-%d', errors='coerce')
    numeric_data = returns_df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
    ex_post_dict = {}
    
    for r_date, weights in weights_dict.items():
        r_date_ts = pd.to_datetime(r_date, format='%Y-%m-%d', errors='coerce')
        mask = dt_cols > r_date_ts
        future_cols = numeric_data.columns[mask]
        
        if len(future_cols) < horizon:
            print(f"Not enough future data for rebalancing date {r_date}. Expected horizon: {horizon}, available: {len(future_cols)}. Skipping.")
            continue
        
        horizon_cols = future_cols[:horizon]
        horizon_data = numeric_data[horizon_cols]
        portfolio_returns = horizon_data.mul(weights, axis=0).sum(axis=0)
        ex_post_dict[r_date] = portfolio_returns
        
    return ex_post_dict

def compute_performance_metrics(portfolio_returns):
    """
    Compute key performance metrics from a Series of monthly portfolio returns.
    
    Metrics computed:
      - Annualized Average Return: (mean monthly return * 12)
      - Annualized Volatility: (std monthly return * sqrt(12))
      - Sharpe Ratio: Annualized Return / Annualized Volatility (risk-free rate assumed 0)
      - Minimum and Maximum monthly returns.
    
    Parameters:
        portfolio_returns (pd.Series): Series of monthly portfolio returns.
    
    Returns:
        metrics (dict): Dictionary containing 'annualized_return', 'annualized_volatility',
                        'sharpe_ratio', 'min_return', and 'max_return'.
    """
    monthly_mean = portfolio_returns.mean()
    monthly_std = portfolio_returns.std()
    annualized_return = monthly_mean * 12
    annualized_volatility = monthly_std * np.sqrt(12)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else np.nan
    min_return = portfolio_returns.min()
    max_return = portfolio_returns.max()
    
    metrics = {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "min_return": min_return,
        "max_return": max_return
    }
    return metrics

def run_optimization_routine(returns_df, rebalancing_dates, estimation_window=120, horizon=12):
    """
    Run the full optimization routine over the sample period.
    
    For each rebalancing date (ISO string), the function:
      - Uses the last 'estimation_window' months of returns prior to the rebalancing date to compute the covariance matrix
        (using the outer-product averaging method) and then the optimal portfolio weights via optimize_portfolio.
      - Computes the ex-post portfolio returns for the following 'horizon' months as the weighted sum of individual firm returns.
      - Calculates performance metrics for these ex-post returns:
            * Annualized Return, Annualized Volatility, Sharpe Ratio, Minimum and Maximum monthly returns.
    
    Parameters:
        returns_df (pd.DataFrame): DataFrame containing monthly returns data for each firm.
                                   Expected structure: first two columns are metadata (e.g., 'Name', 'ISIN'),
                                   and the remaining columns are monthly returns (in ISO format "YYYY-MM-DD")
                                   sorted in chronological order.
        rebalancing_dates (list): List of rebalancing dates as ISO strings (e.g., "2013-12-31").
        estimation_window (int): Number of months used for estimation (default 120).
        horizon (int): Number of months to compute ex-post returns for each rebalancing date (default 12).
    
    Returns:
        results (dict): Dictionary where each key is a rebalancing date and its value is another dictionary with:
             'weights'              : Optimal portfolio weights (pd.Series) at that rebalancing date.
             'ex_post_returns'      : Series of ex-post monthly portfolio returns for the following 'horizon' months.
             'performance_metrics'  : Dictionary with performance metrics.
    """
    weights_dict = compute_rolling_optimal_weights(returns_df, rebalancing_dates, window=estimation_window)
    ex_post_dict = compute_ex_post_returns(returns_df, weights_dict, horizon=horizon)
    
    results = {}
    for r_date in weights_dict:
        if r_date not in ex_post_dict:
            print(f"Skipping {r_date} due to insufficient future data.")
            continue
        
        weights = weights_dict[r_date]
        ex_post_returns = ex_post_dict[r_date]
        metrics = compute_performance_metrics(ex_post_returns)
        
        results[r_date] = {
            "weights": weights,
            "ex_post_returns": ex_post_returns,
            "performance_metrics": metrics
        }
    return results

# Allow the module to run independently for testing purposes.
if __name__ == "__main__":
    # Assume the returns file "Simple_Returns.xlsx" is stored in the Data folder relative to the project root.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(base_dir, "..", "Data", "Simple_Returns.xlsx")
    
    if os.path.exists(data_file):
        returns_df = pd.read_excel(data_file)
        print("Calculating expected returns:")
        exp_returns = calculate_expected_returns(returns_df)
        print(exp_returns.head())
        print("\nComputing covariance matrix:")
        cov_matrix = compute_covariance_matrix(returns_df)
        print(cov_matrix.head())
    else:
        print(f"Returns file not found at {data_file}")
