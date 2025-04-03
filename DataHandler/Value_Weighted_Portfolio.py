import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def calculate_value_weighted_portfolio(market_cap_df, returns_df, start_date=None, end_date=None):
    """
    Calculate the value-weighted portfolio returns.

    Parameters:
        market_cap_df (pd.DataFrame): DataFrame containing monthly market capitalization data.
                                      Expected structure: first two columns are metadata (e.g., 'ISIN', 'Name')
                                      and the remaining columns are monthly market capitalizations.
        returns_df (pd.DataFrame): DataFrame containing monthly returns data.
                                  Expected structure: same as market_cap_df but with returns.
        start_date (str, optional): Start date for the analysis in 'YYYY-MM-DD' format.
        end_date (str, optional): End date for the analysis in 'YYYY-MM-DD' format.

    Returns:
        pd.Series: Monthly value-weighted portfolio returns.
    """
    print("Starting value-weighted portfolio calculation...")

    # Make copies to avoid modifying original DataFrames
    market_cap_df = market_cap_df.copy()
    returns_df = returns_df.copy()

    # Extract the date columns (excluding the first two metadata columns)
    market_cap_dates = list(market_cap_df.columns[2:])
    returns_dates = list(returns_df.columns[2:])

    # Create standardized date mappings (for more consistent date matching)
    mcap_date_map = {}
    for date_str in market_cap_dates:
        try:
            # Try to parse and standardize the date format
            dt = pd.to_datetime(date_str)
            # Use the last day of the month for consistency
            last_day = pd.Timestamp(dt.year, dt.month, 1) + pd.offsets.MonthEnd(1)
            # Store in standardized format
            mcap_date_map[last_day.strftime('%Y-%m-%d')] = date_str
        except:
            # Skip dates that can't be parsed
            continue

    returns_date_map = {}
    for date_str in returns_dates:
        try:
            # Try to parse and standardize the date format
            dt = pd.to_datetime(date_str)
            # Use the last day of the month for consistency
            last_day = pd.Timestamp(dt.year, dt.month, 1) + pd.offsets.MonthEnd(1)
            # Store in standardized format
            returns_date_map[last_day.strftime('%Y-%m-%d')] = date_str
        except:
            # Skip dates that can't be parsed
            continue

    print(f"Standardized market cap dates: {len(mcap_date_map)}")
    print(f"Standardized returns dates: {len(returns_date_map)}")

    # Create a list of all months between start and end date
    if start_date:
        start_date = pd.to_datetime(start_date)
    else:
        # Default to first common date
        common_dates = sorted(set(mcap_date_map.keys()) & set(returns_date_map.keys()))
        if common_dates:
            start_date = pd.to_datetime(common_dates[0])
        else:
            print("No common dates found between market cap and returns data")
            return pd.Series()

    if end_date:
        end_date = pd.to_datetime(end_date)
    else:
        # Default to last common date
        common_dates = sorted(set(mcap_date_map.keys()) & set(returns_date_map.keys()))
        if common_dates:
            end_date = pd.to_datetime(common_dates[-1])
        else:
            print("No common dates found between market cap and returns data")
            return pd.Series()

    print(f"Analysis period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Generate all month-end dates in the range
    all_months = pd.date_range(start=start_date, end=end_date, freq='ME')
    all_months = [d.strftime('%Y-%m-%d') for d in all_months]

    print(f"Total months to analyze: {len(all_months)}")

    # Initialize results dictionary
    vw_returns_dict = {}

    # Calculate value-weighted returns for each month
    for i in range(1, len(all_months)):
        current_month = all_months[i]
        prev_month = all_months[i - 1]

        # Check if these months exist in our datasets
        if prev_month in mcap_date_map and current_month in returns_date_map:
            print(f"Processing month: {current_month}")

            # Get original column names
            mcap_col = mcap_date_map[prev_month]
            returns_col = returns_date_map[current_month]

            # Get market caps for the previous month
            market_caps = market_cap_df[mcap_col].values

            # Get returns for the current month
            returns = returns_df[returns_col].values

            # Replace any non-finite values with NaN
            market_caps = np.where(np.isfinite(market_caps), market_caps, np.nan)
            returns = np.where(np.isfinite(returns), returns, np.nan)

            # Calculate weights based on market cap
            total_mcap = np.nansum(market_caps)
            if total_mcap > 0:
                weights = np.where(np.isnan(market_caps), 0, market_caps / total_mcap)

                # Replace NaN with 0 in returns for the weight calculation
                returns_for_calc = np.where(np.isnan(returns), 0, returns)

                # Calculate portfolio return
                portfolio_return = np.sum(weights * returns_for_calc)

                # Ensure the result is finite
                if np.isfinite(portfolio_return):
                    vw_returns_dict[current_month] = portfolio_return
                    print(f"Portfolio return for {current_month}: {portfolio_return:.4f}")
                else:
                    print(f"Skipping {current_month} - infinite or NaN portfolio return")
            else:
                print(f"Skipping {current_month} - zero or NaN total market cap")
        else:
            if prev_month not in mcap_date_map:
                print(f"Missing market cap data for {prev_month}")
            if current_month not in returns_date_map:
                print(f"Missing returns data for {current_month}")

    # Convert dictionary to Series
    vw_returns = pd.Series(vw_returns_dict)
    vw_returns.index = pd.to_datetime(vw_returns.index)
    vw_returns = vw_returns.sort_index()

    print(f"Value-weighted portfolio calculation complete. Found {len(vw_returns)} months of data.")

    return vw_returns


def plot_cumulative_returns(mv_returns, vw_returns, title="Cumulative Returns Comparison"):
    """
    Plot cumulative returns of minimum variance and value-weighted portfolios.

    Parameters:
        mv_returns (pd.Series): Minimum variance portfolio returns.
        vw_returns (pd.Series): Value-weighted portfolio returns.
        title (str): Plot title.
    """
    # Check if we have valid returns to plot
    if len(mv_returns) == 0 or len(vw_returns) == 0:
        print("Warning: One or both return series is empty. Cannot plot cumulative returns.")
        # Create a dummy plot to avoid errors
        plt.figure(figsize=(12, 8))
        plt.title("Insufficient data for cumulative returns plot")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.text(0.5, 0.5, "No data available", ha='center', va='center', transform=plt.gca().transAxes)
        plt.savefig('cumulative_returns_comparison.png', dpi=300)
        plt.close()
        return None

    # Convert to Series if they're not already
    if not isinstance(mv_returns, pd.Series):
        mv_returns = pd.Series(mv_returns)
    if not isinstance(vw_returns, pd.Series):
        vw_returns = pd.Series(vw_returns)

    # Ensure indices are datetime
    mv_returns.index = pd.to_datetime(mv_returns.index)
    vw_returns.index = pd.to_datetime(vw_returns.index)

    # Remove any non-finite values
    mv_returns = mv_returns[np.isfinite(mv_returns)]
    vw_returns = vw_returns[np.isfinite(vw_returns)]

    # Align the series on the same index
    common_index = mv_returns.index.intersection(vw_returns.index)
    if len(common_index) == 0:
        print("Warning: No common dates between the two return series.")
        plt.figure(figsize=(12, 8))
        plt.title("No common dates between portfolios")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.text(0.5, 0.5, "No common dates available", ha='center', va='center', transform=plt.gca().transAxes)
        plt.savefig('cumulative_returns_comparison.png', dpi=300)
        plt.close()
        return None

    aligned_returns = pd.DataFrame({
        'Minimum Variance': mv_returns[common_index],
        'Value-Weighted': vw_returns[common_index]
    })

    # Drop any rows with NaN values
    aligned_returns = aligned_returns.dropna()

    # Calculate cumulative returns
    cumulative_returns = (1 + aligned_returns).cumprod() - 1

    # Plot
    plt.figure(figsize=(12, 8))
    for col in cumulative_returns.columns:
        plt.plot(cumulative_returns.index, cumulative_returns[col], label=col)

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cumulative_returns_comparison.png', dpi=300)
    plt.close()

    return cumulative_returns


def compare_portfolio_performance(mv_returns, vw_returns, rf_rates=None):
    """
    Compare performance metrics between minimum variance and value-weighted portfolios.

    Parameters:
        mv_returns (pd.Series): Minimum variance portfolio returns.
        vw_returns (pd.Series): Value-weighted portfolio returns.
        rf_rates (pd.Series, optional): Risk-free rates aligned with return dates.

    Returns:
        pd.DataFrame: Comparison of portfolio metrics.
    """
    # Check if we have valid returns to compare
    if len(mv_returns) == 0 or len(vw_returns) == 0:
        print("Warning: One or both return series is empty. Cannot compare performance.")
        # Return empty DataFrame with the expected structure
        return pd.DataFrame({
            'Minimum Variance': [np.nan, np.nan, 0.0, np.nan, np.nan],
            'Value-Weighted': [np.nan, np.nan, 0.0, np.nan, np.nan]
        }, index=['Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Min Monthly Return', 'Max Monthly Return'])

    # Convert to Series if they're not already
    if not isinstance(mv_returns, pd.Series):
        mv_returns = pd.Series(mv_returns)
    if not isinstance(vw_returns, pd.Series):
        vw_returns = pd.Series(vw_returns)

    # Ensure indices are datetime
    mv_returns.index = pd.to_datetime(mv_returns.index)
    vw_returns.index = pd.to_datetime(vw_returns.index)

    # Remove any non-finite values
    mv_returns_clean = mv_returns[np.isfinite(mv_returns)]
    vw_returns_clean = vw_returns[np.isfinite(vw_returns)]

    # Annualization factor (assuming monthly returns)
    ann_factor = 12

    # Align the series on the same index
    common_index = mv_returns_clean.index.intersection(vw_returns_clean.index)
    if len(common_index) == 0:
        print("Warning: No common dates between the two return series.")
        return pd.DataFrame({
            'Minimum Variance': [np.nan, np.nan, 0.0, np.nan, np.nan],
            'Value-Weighted': [np.nan, np.nan, 0.0, np.nan, np.nan]
        }, index=['Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Min Monthly Return', 'Max Monthly Return'])

    aligned_returns = pd.DataFrame({
        'Minimum Variance': mv_returns_clean[common_index],
        'Value-Weighted': vw_returns_clean[common_index]
    })

    # Drop any rows with NaN values
    aligned_returns = aligned_returns.dropna()

    # Initialize metrics dictionary
    metrics = {}

    # If risk-free rates are not provided, assume zero
    if rf_rates is None:
        rf_rates = pd.Series(0, index=aligned_returns.index)
    else:
        # Ensure risk-free rates are aligned with returns
        rf_rates = rf_rates.reindex(aligned_returns.index, method='ffill')

    # Calculate metrics for each portfolio
    for col in aligned_returns.columns:
        portfolio_returns = aligned_returns[col]

        # Annual average return
        annual_return = portfolio_returns.mean() * ann_factor

        # Annual volatility
        annual_vol = portfolio_returns.std() * np.sqrt(ann_factor)

        # Average risk-free rate
        avg_rf = rf_rates.mean() * ann_factor

        # Sharpe ratio
        sharpe = (annual_return - avg_rf) / annual_vol if annual_vol > 0 else 0

        metrics[col] = {
            'Annual Return': annual_return,
            'Annual Volatility': annual_vol,
            'Sharpe Ratio': sharpe,
            'Min Monthly Return': portfolio_returns.min(),
            'Max Monthly Return': portfolio_returns.max()
        }

    # Convert to DataFrame for easy comparison
    metrics_df = pd.DataFrame(metrics)

    # Create visualizations if we have valid data
    if not aligned_returns.empty:
        # Plot annual return and volatility
        plt.figure(figsize=(12, 8))

        x = np.arange(2)
        width = 0.35

        plt.bar(x - width / 2, metrics_df.loc[['Annual Return', 'Annual Volatility'], 'Minimum Variance'],
                width, label='Minimum Variance')
        plt.bar(x + width / 2, metrics_df.loc[['Annual Return', 'Annual Volatility'], 'Value-Weighted'],
                width, label='Value-Weighted')

        plt.xticks(x, ['Annual Return', 'Annual Volatility'])
        plt.ylabel('Value')
        plt.title('Portfolio Performance Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('portfolio_metrics_comparison.png', dpi=300)
        plt.close()

        # Plot Sharpe ratio separately
        plt.figure(figsize=(8, 6))
        plt.bar(['Minimum Variance', 'Value-Weighted'],
                [metrics_df.loc['Sharpe Ratio', 'Minimum Variance'],
                 metrics_df.loc['Sharpe Ratio', 'Value-Weighted']])
        plt.ylabel('Sharpe Ratio')
        plt.title('Sharpe Ratio Comparison')
        plt.grid(True, alpha=0.3)
        plt.savefig('sharpe_ratio_comparison.png', dpi=300)
        plt.close()

    return metrics_df