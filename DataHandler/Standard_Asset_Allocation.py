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

def optimize_portfolio(returns, cov_matrix, risk_aversion=1.0):
    """
    Optimize portfolio weights using mean-variance optimization with robust constraints.
    """
    try:
        n_assets = cov_matrix.shape[0]
        
        # Calculate expected returns (sample mean)
        mu = returns.mean(axis=0).to_numpy()  # Convert to numpy array
        
        # Handle extreme values in expected returns
        mu = np.clip(mu, -0.05, 0.05)  # Limit monthly returns to ±5%
        
        # Convert covariance matrix to numpy array
        cov_matrix_np = cov_matrix.to_numpy()
        
        # Ensure covariance matrix is symmetric
        cov_matrix_np = (cov_matrix_np + cov_matrix_np.T) / 2
        
        # Add regularization to diagonal to improve conditioning
        min_eigenval = np.linalg.eigvalsh(cov_matrix_np)[0]
        if min_eigenval < 1e-6:
            regularization = abs(min_eigenval) + 1e-6
            cov_matrix_np += regularization * np.eye(n_assets)
        
        # Define optimization problem
        w = cp.Variable(n_assets)
        ret = mu @ w
        risk = cp.quad_form(w, cov_matrix_np)
        
        # Define objective function (mean-variance utility)
        objective = cp.Maximize(ret - risk_aversion * risk)
        
        # Define robust constraints
        constraints = [
            cp.sum(w) == 1,     # Budget constraint
            w >= 0.001,         # Minimum weight of 0.1%
            w <= 0.05,          # Maximum weight of 5%
            cp.sum(w >= 0.01) >= 20  # At least 20 assets with weight >= 1%
        ]
        
        # Solve optimization problem
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP, eps_abs=1e-6, eps_rel=1e-6, max_iter=10000)
            
            if prob.status == 'optimal':
                # Get the optimal weights
                weights = w.value
                
                # Clean up small weights
                weights[weights < 0.001] = 0.0
                
                # Renormalize to ensure sum is exactly 1.0
                if np.sum(weights) > 0:
                    weights = weights / np.sum(weights)
                else:
                    print("Warning: All weights are zero after cleanup")
                    # Use equal weights for top 20 assets by Sharpe ratio
                    asset_sharpe = mu / np.sqrt(np.diag(cov_matrix_np))
                    top_20_idx = np.argsort(asset_sharpe)[-20:]
                    weights = np.zeros(n_assets)
                    weights[top_20_idx] = 1/20
                
                return weights
            else:
                print(f"Warning: Optimization not optimal. Status: {prob.status}")
                # Use equal weights for top 20 assets by Sharpe ratio
                asset_sharpe = mu / np.sqrt(np.diag(cov_matrix_np))
                top_20_idx = np.argsort(asset_sharpe)[-20:]
                weights = np.zeros(n_assets)
                weights[top_20_idx] = 1/20
                return weights
        except Exception as e:
            print(f"Error in optimization: {str(e)}")
            # Use equal weights for top 20 assets by Sharpe ratio
            asset_sharpe = mu / np.sqrt(np.diag(cov_matrix_np))
            top_20_idx = np.argsort(asset_sharpe)[-20:]
            weights = np.zeros(n_assets)
            weights[top_20_idx] = 1/20
            return weights
            
    except Exception as e:
        print(f"Error in portfolio optimization: {str(e)}")
        # Use equal weights for top 20 assets by Sharpe ratio
        asset_sharpe = mu / np.sqrt(np.diag(cov_matrix_np))
        top_20_idx = np.argsort(asset_sharpe)[-20:]
        weights = np.zeros(n_assets)
        weights[top_20_idx] = 1/20
        return weights

def compute_rolling_optimal_weights(returns_df, window_size=120, risk_aversion=1.0):
    """
    Compute optimal portfolio weights using a rolling window approach.
    """
    # Get numeric data only (skip metadata columns)
    numeric_data = returns_df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
    time_cols = returns_df.columns[2:]
    
    # Convert time columns to datetime
    dt_cols = pd.to_datetime(time_cols, format='%Y-%m-%d', errors='coerce')
    
    # Initialize weights DataFrame with same structure as returns
    weights = pd.DataFrame(0, index=returns_df.index, columns=time_cols)
    portfolio_returns = pd.Series(0, index=time_cols)
    
    # For each time period after the initial window
    for t in range(window_size, len(time_cols)):
        r_date = time_cols[t]
        print(f"\nProcessing rebalancing date: {r_date}")
        
        try:
            # Get window data
            window_cols = time_cols[t-window_size:t]
            window_data = numeric_data[window_cols].T  # Transpose to get time x assets
            print(f"Window data shape: {window_data.shape}")
            
            # Handle missing and infinite values
            missing_count = window_data.isna().sum().sum()
            inf_count = np.isinf(window_data).sum().sum()
            print(f"Missing values in window: {missing_count}")
            print(f"Infinite values in window: {inf_count}")
            
            # Replace infinite values with NaN first
            window_data = window_data.replace([np.inf, -np.inf], np.nan)
            
            # Fill missing values with column means
            window_data = window_data.fillna(window_data.mean())
            
            # Replace any remaining NaN (if entire column was NaN) with 0
            window_data = window_data.fillna(0)
            
            # Check for any remaining invalid values
            if window_data.isna().any().any() or np.isinf(window_data).any().any():
                print("Warning: Invalid values still present after cleaning")
                continue
            
            # Compute covariance matrix for assets
            cov_matrix = window_data.cov()  # This will be assets x assets
            print(f"Covariance matrix shape: {cov_matrix.shape}")
            print(f"Covariance matrix range: [{cov_matrix.min().min()}, {cov_matrix.max().max()}]")
            
            # Check if covariance matrix is valid
            if cov_matrix.isna().any().any() or np.isinf(cov_matrix).any().any():
                print("Warning: Invalid values in covariance matrix")
                continue
            
            # Optimize portfolio
            optimal_weights = optimize_portfolio(window_data, cov_matrix, risk_aversion)
            
            if optimal_weights is not None:
                weights.iloc[:, t] = optimal_weights
                print(f"Sum of weights: {np.sum(optimal_weights)}")
                print(f"Number of non-zero weights: {np.sum(np.abs(optimal_weights) > 0.001)}")
                
                # Calculate portfolio return for the next period
                if t + 1 < len(time_cols):
                    next_return = numeric_data[time_cols[t+1]]
                    portfolio_return = np.sum(optimal_weights * next_return)
                    portfolio_returns[time_cols[t]] = portfolio_return
            else:
                print("Warning: Optimization failed for this window")
                # Use equal weights as fallback
                equal_weights = np.ones(len(returns_df)) / len(returns_df)
                weights.iloc[:, t] = equal_weights
                
                if t + 1 < len(time_cols):
                    next_return = numeric_data[time_cols[t+1]]
                    portfolio_return = np.sum(equal_weights * next_return)
                    portfolio_returns[time_cols[t]] = portfolio_return
                    
        except Exception as e:
            print(f"Error processing window at {r_date}: {str(e)}")
            continue
    
    # Create a dictionary of weights for each rebalancing date
    weights_dict = {col: weights[col] for col in weights.columns[window_size:]}
    
    return weights_dict, portfolio_returns[:-1]  # Exclude the last period since we don't have next period returns

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
    weights_dict, portfolio_returns = compute_rolling_optimal_weights(returns_df, window_size=estimation_window)
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
