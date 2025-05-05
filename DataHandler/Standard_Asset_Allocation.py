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
    rebalance_dates = pd.date_range(start='2013-12-31', end='2022-12-31', freq='YE')

    for rebalance_date in rebalance_dates:
        logging.info(f"Optimizing for {rebalance_date.strftime('%Y-%m')}")
        try:
            # Find the closest date to rebalance_date in the returns data
            closest_date_idx = None
            closest_distance = float('inf')

            for i, date in enumerate(date_columns):
                if pd.notna(date):
                    distance = abs((date - rebalance_date).days)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_date_idx = i

            if closest_date_idx is None:
                logging.warning(f"No data available near {rebalance_date}. Skipping.")
                continue

            if closest_distance > 45:  # More than 45 days from target date
                logging.warning(
                    f"Closest date is {closest_distance} days from {rebalance_date}. Proceeding with caution.")

            # Select window of returns data - ensure window is valid
            window_start_idx = max(0, closest_date_idx - window_size + 1)
            window_end_idx = closest_date_idx + 1  # Include the closest date

            # Ensure indices are within bounds
            if window_start_idx >= len(date_columns) or window_end_idx > len(date_columns):
                logging.warning(f"Window indices out of bounds for {rebalance_date}. Skipping.")
                continue

            window_dates = date_columns[window_start_idx:window_end_idx]

            if len(window_dates) < window_size * 0.5:  # Require at least 50% of the window
                logging.warning(
                    f"Insufficient data for {rebalance_date} (only {len(window_dates)}/{window_size} months). Skipping.")
                continue

            # Extract returns matrix safely
            returns_matrix = np.zeros((len(returns_df), len(window_dates)))

            for i, date in enumerate(window_dates):
                # Ensure date is in the dataframe's columns
                if date in returns_df.columns:
                    returns_matrix[:, i] = returns_df[date].values
                else:
                    logging.warning(f"Date {date} not found in returns_df. Using zeros.")
                    returns_matrix[:, i] = np.nan

            # Check for NaN values and handle them
            nan_count = np.isnan(returns_matrix).sum()
            if nan_count > 0:
                logging.warning(f"Found {nan_count} NaN values in returns matrix for {rebalance_date}")

            # Clean data: remove assets with too many NaNs
            na_ratio = np.isnan(returns_matrix).mean(axis=1)
            valid_assets = np.where(na_ratio < 0.2)[0]  # Less than 20% NaNs

            if len(valid_assets) < 10:  # Require at least 10 assets
                logging.warning(
                    f"Too few valid assets ({len(valid_assets)}) for {rebalance_date}. Skipping.")
                continue

            # Extract valid returns safely
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

            # Check if there are any NaNs left
            if np.any(np.isnan(valid_returns)):
                logging.warning("NaNs still present in valid_returns after fillna. Replacing with zeros.")
                valid_returns = np.nan_to_num(valid_returns, nan=0.0)

            # Compute expected returns and covariance matrix
            expected_returns = np.nanmean(valid_returns, axis=1)

            # Regularized covariance estimation with stronger regularization
            try:
                cov_matrix = np.cov(valid_returns, rowvar=True, bias=True)

                # Check if cov_matrix computation failed
                if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
                    logging.warning(f"Invalid covariance matrix. Using identity matrix with small random perturbation.")
                    n = len(valid_assets)
                    cov_matrix = np.eye(n) + np.random.normal(0, 0.01, (n, n))
                    cov_matrix = (cov_matrix + cov_matrix.T) / 2  # Ensure symmetry

                # Ensure positive definiteness with stronger regularization
                min_eigenval = np.min(np.linalg.eigvals(cov_matrix))
                if min_eigenval < 1e-3:  # Increased threshold for regularization
                    logging.info(f"Adding regularization to ensure positive definite covariance matrix")
                    regularization = (abs(min_eigenval) + 1e-2) * np.eye(cov_matrix.shape[0])
                    cov_matrix += regularization
            except Exception as e:
                logging.warning(f"Error computing covariance matrix: {str(e)}. Using identity matrix.")
                cov_matrix = np.eye(len(valid_assets))

            # Minimum variance optimization
            n_assets = len(valid_assets)
            w = cp.Variable(n_assets)

            objective = cp.Minimize(cp.quad_form(w, cov_matrix))
            constraints = [
                cp.sum(w) == 1,
                w >= 0
            ]

            try:
                # Try with multiple solvers for robustness
                weights = None
                for solver, solver_name, solver_opts in [
                    (cp.OSQP, "OSQP", {"eps_abs": 1e-3, "eps_rel": 1e-3, "max_iter": 10000}),
                    (cp.SCS, "SCS", {"eps": 1e-3, "max_iters": 10000}),
                    (cp.ECOS, "ECOS", {"abstol": 1e-3, "reltol": 1e-3})
                ]:
                    try:
                        problem = cp.Problem(objective, constraints)
                        problem.solve(solver=solver, verbose=True, **solver_opts)

                        if problem.status in ["optimal", "optimal_inaccurate"]:
                            logging.info(f"Solver {solver_name} succeeded with status {problem.status}")
                            weights = w.value
                            break
                        else:
                            logging.warning(f"Solver {solver_name} failed with status {problem.status}")
                    except Exception as e:
                        logging.warning(f"Error with {solver_name} solver: {str(e)}")
                        continue

                # If all solvers failed, use equal weights
                if weights is None or problem.status not in ["optimal", "optimal_inaccurate"]:
                    logging.warning(f"All solvers failed. Using equal weights.")
                    weights = np.ones(n_assets) / n_assets
                else:
                    # Clean weights
                    weights = np.maximum(weights, 0)  # Ensure no negative weights
                    if np.sum(weights) > 0:
                        weights = weights / np.sum(weights)  # Renormalize
                    else:
                        logging.warning(f"Sum of weights is zero. Using equal weights.")
                        weights = np.ones(n_assets) / n_assets
            except Exception as e:
                logging.error(f"Optimization error: {str(e)}. Using equal weights.")
                weights = np.ones(n_assets) / n_assets

            # Validate weights
            if np.any(np.isnan(weights)) or np.sum(weights) < 0.999:
                logging.warning(f"Invalid weights produced. Using equal weights.")
                weights = np.ones(n_assets) / n_assets

            # Convert to Series with ISIN index
            valid_isins = returns_df['ISIN'].iloc[valid_assets].values
            weights_series = pd.Series(weights, index=valid_isins)

            # Store weights for this rebalance date
            weights_dict[rebalance_date] = weights_series

            # Calculate out-of-sample returns for the next year
            next_year_start = rebalance_date
            next_year_end = pd.Timestamp(f"{rebalance_date.year + 1}-12-31")

            next_year_dates = [date for date in date_columns if
                               pd.notna(date) and next_year_start < date <= next_year_end]

            for next_date in next_year_dates:
                try:
                    if next_date in returns_df.columns:
                        next_returns = returns_df[next_date]

                        # Calculate portfolio return using weights
                        portfolio_return = 0
                        weight_sum = 0

                        for isin, weight in weights_series.items():
                            asset_indices = returns_df.index[returns_df['ISIN'] == isin].tolist()
                            if not asset_indices:
                                logging.warning(f"ISIN {isin} not found in returns_df for {next_date}. Skipping.")
                                continue
                            asset_idx = asset_indices[0]
                            asset_return = next_returns.iloc[asset_idx]
                            if pd.notna(asset_return):
                                portfolio_return += weight * asset_return
                                weight_sum += weight

                        # Normalize by the sum of weights with valid returns
                        if weight_sum > 0:
                            portfolio_return = portfolio_return / weight_sum
                            portfolio_returns[next_date] = portfolio_return
                        else:
                            logging.warning(f"No valid returns for {next_date}. Skipping.")
                except Exception as e:
                    logging.warning(f"Error calculating return for {next_date}: {str(e)}. Skipping.")

            # Log portfolio statistics
            non_zero_weights = np.sum(weights > 1e-4)
            logging.info(f"  Non-zero weights: {non_zero_weights}/{n_assets}")
            if len(weights) > 0:
                logging.info(f"  Max weight: {np.max(weights):.4f}")

                # Calculate effective number of assets
                if np.sum(weights ** 2) > 0:
                    effective_n = 1 / np.sum(weights ** 2)
                    logging.info(f"  Effective N: {effective_n:.1f}")

        except Exception as e:
            logging.error(f"Error in optimization for {rebalance_date}: {str(e)}")
            # Skip this period instead of using equal weights
            logging.warning(f"Skipping portfolio optimization for {rebalance_date}")
            continue

    # Compute performance metrics
    if not portfolio_returns.empty:
        portfolio_returns = portfolio_returns.dropna()

        if len(portfolio_returns) > 0:
            # Compute annualized performance metrics with robust calculation
            try:
                # Ensure all returns are finite
                finite_returns = portfolio_returns[np.isfinite(portfolio_returns)]

                if len(finite_returns) > 0:
                    # Annualized return using geometric mean
                    cumulative_return = (1 + finite_returns).prod()
                    n_periods = len(finite_returns)
                    annualized_return = cumulative_return ** (12 / n_periods) - 1

                    # Annualized volatility
                    annualized_volatility = finite_returns.std() * np.sqrt(12)

                    # Sharpe ratio
                    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else np.nan

                    metrics = {
                        'annualized_return': annualized_return,
                        'annualized_volatility': annualized_volatility,
                        'sharpe_ratio': sharpe_ratio,
                        'min_return': finite_returns.min(),
                        'max_return': finite_returns.max()
                    }

                    logging.info(f"Portfolio metrics calculated: Return={annualized_return:.4f}, "
                                 f"Vol={annualized_volatility:.4f}, SR={sharpe_ratio:.4f}")
            except Exception as e:
                logging.error(f"Error calculating performance metrics: {str(e)}")

    return metrics, portfolio_returns, weights_dict, valid_cols_dict