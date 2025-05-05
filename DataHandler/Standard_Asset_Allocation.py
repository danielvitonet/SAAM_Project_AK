import numpy as np
import pandas as pd
from scipy import stats
import cvxpy as cp
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_portfolio_optimization(returns_df, window_size=120):
    """
    Run minimum variance portfolio optimization with a rolling window.

    Parameters:
    - returns_df: DataFrame with ISIN, Company Name, and monthly returns
    - window_size: Number of months for the rolling window

    Returns:
    - metrics: Dictionary with portfolio performance metrics
    - portfolio_returns: Series of out-of-sample portfolio returns
    - weights_dict: Dictionary of weights for each rebalance date
    - valid_cols_dict: Dictionary of valid column indices for each rebalance date
    """
    # Ensure returns_df columns from index 2 onwards are dates
    date_columns = pd.to_datetime(returns_df.columns[2:], errors='coerce')
    if date_columns.isna().any():
        logging.error("Invalid date columns in returns_df")
        raise ValueError("Invalid date columns in returns_df")

    returns_df = returns_df.copy()
    returns_df.columns = ['ISIN', 'NAME'] + date_columns.tolist()

    # Initialize outputs
    portfolio_returns = pd.Series(dtype=float)
    weights_dict = {}
    valid_cols_dict = {}
    metrics = {
        'annualized_return': np.nan,
        'annualized_volatility': np.nan,
        'sharpe_ratio': np.nan,
        'min_return': np.nan,
        'max_return': np.nan
    }

    # Get rebalance dates (end of each year from 2013 to 2022)
    rebalance_dates = pd.date_range(start='2013-12-31', end='2022-12-31', freq='Y')

    for rebalance_date in rebalance_dates:
        logging.info(f"Optimizing for {rebalance_date.strftime('%Y-%m')}")
        try:
            # Find the closest date to rebalance_date in the returns data
            closest_date_idx = None
            for i, date in enumerate(date_columns):
                if date <= rebalance_date:
                    closest_date_idx = i

            if closest_date_idx is None:
                logging.warning(f"No data available before {rebalance_date}. Skipping.")
                continue

            # Select window of returns data
            window_start_idx = max(0, closest_date_idx - window_size + 1)
            window_dates = date_columns[window_start_idx:closest_date_idx + 1]

            if len(window_dates) < window_size * 0.8:  # Require at least 80% of the window
                logging.warning(
                    f"Insufficient data for {rebalance_date} (only {len(window_dates)} months). Proceeding with available data.")

            # Extract returns matrix
            returns_matrix = np.zeros((len(returns_df), len(window_dates)))

            for i, date in enumerate(window_dates):
                returns_matrix[:, i] = returns_df[date].values

            # Clean data: remove assets with too many NaNs
            na_ratio = np.isnan(returns_matrix).mean(axis=1)
            valid_assets = np.where(na_ratio < 0.2)[0]  # Less than 20% NaNs

            if len(valid_assets) < 10:  # Require at least 10 assets
                logging.warning(
                    f"Too few valid assets ({len(valid_assets)}) for {rebalance_date}. Using equal weights.")
                equal_weights = np.ones(len(returns_df)) / len(returns_df)
                weights_dict[rebalance_date] = pd.Series(equal_weights, index=returns_df['ISIN'])
                valid_cols_dict[rebalance_date] = np.arange(len(returns_df))
                continue

            valid_returns = returns_matrix[valid_assets, :]
            valid_cols_dict[rebalance_date] = valid_assets

            # Handle remaining NaNs by filling with column (time) means
            for col in range(valid_returns.shape[1]):
                col_data = valid_returns[:, col]
                nan_mask = np.isnan(col_data)
                if np.all(nan_mask):
                    valid_returns[:, col] = 0  # If all NaN, use zeros
                else:
                    col_mean = np.nanmean(col_data)
                    valid_returns[nan_mask, col] = col_mean

            # Compute expected returns and covariance matrix
            expected_returns = np.nanmean(valid_returns, axis=1)

            # Regularized covariance estimation
            cov_matrix = np.cov(valid_returns, rowvar=True, bias=True)

            # Ensure positive definiteness
            min_eigenval = np.min(np.linalg.eigvals(cov_matrix))
            if min_eigenval < 1e-8:
                logging.info(f"Adding regularization to ensure positive definite covariance matrix")
                cov_matrix += (abs(min_eigenval) + 1e-5) * np.eye(cov_matrix.shape[0])

            # Minimum variance optimization
            n_assets = len(valid_assets)
            w = cp.Variable(n_assets)

            objective = cp.Minimize(cp.quad_form(w, cov_matrix))
            constraints = [
                cp.sum(w) == 1,
                w >= 0
            ]

            try:
                # Try with OSQP solver first
                problem = cp.Problem(objective, constraints)
                problem.solve(solver=cp.OSQP, eps_abs=1e-5, eps_rel=1e-5)

                if problem.status not in ["optimal", "optimal_inaccurate"]:
                    # Try with SCS solver if OSQP fails
                    logging.warning(f"OSQP solver failed with status {problem.status}. Trying SCS.")
                    problem.solve(solver=cp.SCS, eps=1e-5)

                if problem.status not in ["optimal", "optimal_inaccurate"]:
                    # If both solvers fail, use equal weights
                    logging.warning(f"Both solvers failed. Using equal weights.")
                    weights = np.ones(n_assets) / n_assets
                else:
                    weights = w.value
                    weights = np.maximum(weights, 0)  # Ensure no negative weights
                    weights = weights / np.sum(weights)  # Renormalize
            except Exception as e:
                logging.error(f"Optimization error: {str(e)}. Using equal weights.")
                weights = np.ones(n_assets) / n_assets

            # Convert to Series with ISIN index
            valid_isins = returns_df['ISIN'].iloc[valid_assets].values
            weights_series = pd.Series(weights, index=valid_isins)

            # Store weights for this rebalance date
            weights_dict[rebalance_date] = weights_series

            # Calculate out-of-sample returns for the next year
            next_year_start = rebalance_date
            next_year_end = pd.Timestamp(f"{rebalance_date.year + 1}-12-31")

            next_year_dates = [date for date in date_columns if next_year_start < date <= next_year_end]

            for next_date in next_year_dates:
                next_returns = returns_df[next_date]

                # Calculate portfolio return using weights
                portfolio_return = 0
                for isin, weight in weights_series.items():
                    asset_idx = returns_df[returns_df['ISIN'] == isin].index[0]
                    asset_return = next_returns.iloc[asset_idx]
                    if not np.isnan(asset_return):
                        portfolio_return += weight * asset_return

                portfolio_returns[next_date] = portfolio_return

            logging.info(f"  Non-zero weights: {np.sum(weights > 1e-4)}/{n_assets}")
            logging.info(f"  Max weight: {np.max(weights):.4f}")

        except Exception as e:
            logging.error(f"Error in optimization for {rebalance_date}: {str(e)}")
            # Use equal weights as fallback
            equal_weights = np.ones(len(returns_df)) / len(returns_df)
            weights_dict[rebalance_date] = pd.Series(equal_weights, index=returns_df['ISIN'])
            valid_cols_dict[rebalance_date] = np.arange(len(returns_df))

    # Compute performance metrics
    if not portfolio_returns.empty:
        portfolio_returns = portfolio_returns.dropna()

        if len(portfolio_returns) > 0:
            # Compute annualized performance metrics
            annualized_return = (1 + portfolio_returns).prod() ** (12 / len(portfolio_returns)) - 1
            annualized_volatility = portfolio_returns.std() * np.sqrt(12)
            sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else np.nan

            metrics = {
                'annualized_return': annualized_return,
                'annualized_volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'min_return': portfolio_returns.min(),
                'max_return': portfolio_returns.max()
            }

            logging.info(
                f"Portfolio metrics calculated: Return={annualized_return:.4f}, Vol={annualized_volatility:.4f}, SR={sharpe_ratio:.4f}")

    return metrics, portfolio_returns, weights_dict, valid_cols_dict