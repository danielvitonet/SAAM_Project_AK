import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging


def calculate_value_weighted_portfolio(market_cap_df, returns_df):
    """
    Calculate value-weighted portfolio returns with annual rebalancing
    according to project specifications, with improved error handling
    """
    logging.info("Starting value-weighted portfolio calculation...")

    # Convert to numeric
    market_cap_data = market_cap_df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
    returns_data = returns_df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')

    # Convert columns to datetime
    market_cap_dates = pd.to_datetime(market_cap_data.columns, errors='coerce')
    returns_dates = pd.to_datetime(returns_data.columns, errors='coerce')

    # Set column and index labels
    market_cap_data.columns = market_cap_dates
    returns_data.columns = returns_dates
    market_cap_data.index = market_cap_df['ISIN']
    returns_data.index = returns_df['ISIN']

    # Transpose for time-series operations
    market_cap_data = market_cap_data.T
    returns_data = returns_data.T

    # Filter for 2014-2023 period
    returns_mask = (returns_data.index >= '2014-01-01') & (returns_data.index <= '2023-12-31')
    returns_data = returns_data[returns_mask]

    market_cap_mask = (market_cap_data.index >= '2013-12-31') & (market_cap_data.index <= '2023-12-31')
    market_cap_data = market_cap_data[market_cap_mask]

    logging.info(f"Filtered returns data: {len(returns_data)} dates, {len(returns_data.columns)} assets")
    logging.info(f"Filtered market cap data: {len(market_cap_data)} dates, {len(market_cap_data.columns)} assets")

    # Initialize outputs
    portfolio_returns = []
    dates = []
    weights_dict = {}

    # Find rebalance dates (end of each year)
    rebalance_dates = []
    for year in range(2013, 2023):
        dec_date = pd.Timestamp(f"{year}-12-31")
        # Find closest available date
        closest_date = min(market_cap_data.index, key=lambda x: abs(x - dec_date))
        # Only use if it's within 45 days of the target date
        if abs((closest_date - dec_date).days) <= 45:
            rebalance_dates.append(closest_date)
            logging.info(f"Using {closest_date} as rebalance date for year {year}")
        else:
            logging.warning(f"No suitable rebalance date found near {dec_date}. Skipping.")

    # Add final date
    last_date = max(market_cap_data.index)
    if abs((last_date - pd.Timestamp('2023-12-31')).days) <= 45:
        rebalance_dates.append(last_date)
        logging.info(f"Using {last_date} as final rebalance date")
    else:
        logging.warning(f"No suitable final rebalance date found near 2023-12-31.")

    # Calculate portfolio returns for each period
    for i in range(len(rebalance_dates) - 1):
        start_date = rebalance_dates[i]
        end_date = rebalance_dates[i + 1]

        logging.info(f"Processing period {start_date} to {end_date}...")

        # Get market caps for the start date
        market_caps = market_cap_data.loc[start_date]

        # Handle missing or non-positive values
        valid_mask = (~market_caps.isna()) & (market_caps > 0)
        valid_count = valid_mask.sum()

        if valid_count == 0:
            logging.warning(f"No valid market caps for {start_date}, using equal weights")
            valid_mask = ~market_caps.isna()  # Just remove NaNs
            if valid_mask.sum() == 0:
                logging.error(f"No valid market cap data at all for {start_date}, skipping period")
                weights_dict[start_date] = None
                continue

            valid_caps = pd.Series(1.0, index=market_caps[valid_mask].index)
        else:
            valid_caps = market_caps[valid_mask]
            logging.info(f"Using {valid_count} assets with valid market caps for {start_date}")

        # Calculate weights
        vw_weights = pd.Series(0.0, index=market_caps.index)
        vw_weights[valid_mask] = valid_caps / valid_caps.sum()
        weights_dict[start_date] = vw_weights

        # Filter returns for this period
        period_mask = (returns_data.index > start_date) & (returns_data.index <= end_date)
        period_returns = returns_data[period_mask]

        if len(period_returns) == 0:
            logging.warning(f"No returns data for period {start_date} to {end_date}, skipping")
            continue

        # Track weights through the period
        current_weights = vw_weights.copy()

        # Calculate returns for each month in the period
        for idx, date in enumerate(period_returns.index):
            monthly_returns = period_returns.loc[date]

            # Find common assets between weights and returns
            common_index = current_weights.index.intersection(monthly_returns.index)
            weights_aligned = current_weights[common_index]
            returns_aligned = monthly_returns[common_index]

            # Handle NaN returns
            returns_aligned = returns_aligned.fillna(0)

            # Calculate portfolio return
            if len(weights_aligned) > 0 and weights_aligned.sum() > 0:
                # Normalize weights if needed
                if abs(weights_aligned.sum() - 1.0) > 1e-4:
                    weights_aligned = weights_aligned / weights_aligned.sum()

                portfolio_return = np.sum(weights_aligned * returns_aligned)
                portfolio_returns.append(portfolio_return)
                dates.append(date)

                # Update weights for next month (except for the last month)
                if idx < len(period_returns) - 1:
                    denominator = 1 + portfolio_return
                    if abs(denominator) > 1e-8:
                        numerator = current_weights * (1 + monthly_returns.fillna(0))
                        current_weights = numerator / denominator

                        # Renormalize weights
                        valid_weights = ~current_weights.isna() & (current_weights > 0)
                        if valid_weights.sum() > 0:
                            current_weights[valid_weights] = current_weights[valid_weights] / current_weights[
                                valid_weights].sum()
                            current_weights[~valid_weights] = 0
            else:
                logging.warning(f"No valid weights or returns overlap for {date}, skipping")

    # Convert to Series
    if portfolio_returns:
        portfolio_series = pd.Series(portfolio_returns, index=dates)
        logging.info(f"Value-weighted portfolio calculated with {len(portfolio_series)} monthly returns")
        return portfolio_series, weights_dict
    else:
        logging.error("No portfolio returns could be calculated")
        return pd.Series(dtype=float), {}


def plot_cumulative_returns(mv_returns, vw_returns, output_path):
    """
    Plot cumulative returns comparison with improved visualization
    """
    # Handle missing data
    mv_returns = mv_returns[np.isfinite(mv_returns)]
    vw_returns = vw_returns[np.isfinite(vw_returns)]

    # Get common date range
    common_dates = mv_returns.index.intersection(vw_returns.index)
    if len(common_dates) == 0:
        logging.error("No common dates between MV and VW returns. Cannot create plot.")
        return None, None

    mv_returns = mv_returns[common_dates]
    vw_returns = vw_returns[common_dates]

    # Calculate cumulative returns
    mv_cumulative = (1 + mv_returns).cumprod() - 1
    vw_cumulative = (1 + vw_returns).cumprod() - 1

    # Create figure
    plt.figure(figsize=(12, 8))
    plt.plot(mv_cumulative.index, mv_cumulative.values * 100, 'b-', label='Minimum Variance', linewidth=2)
    plt.plot(vw_cumulative.index, vw_cumulative.values * 100, 'r-', label='Value-Weighted', linewidth=2)

    # Add a horizontal line at y=0
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

    # Add grid
    plt.grid(True, alpha=0.3)

    # Add labels and title
    plt.title('Cumulative Returns: Minimum Variance vs Value-Weighted', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (%)', fontsize=12)
    plt.legend(fontsize=12, loc='best')

    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))

    # Add annotations for final values
    final_mv = mv_cumulative.iloc[-1] * 100
    final_vw = vw_cumulative.iloc[-1] * 100
    plt.annotate(f'{final_mv:.1f}%',
                 xy=(mv_cumulative.index[-1], final_mv),
                 xytext=(5, 5),
                 textcoords='offset points',
                 fontweight='bold')
    plt.annotate(f'{final_vw:.1f}%',
                 xy=(vw_cumulative.index[-1], final_vw),
                 xytext=(5, 5),
                 textcoords='offset points',
                 fontweight='bold')

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Cumulative returns plot saved to {output_path}")

    return mv_cumulative, vw_cumulative


def compare_portfolio_performance(mv_returns, vw_returns, risk_free_rate=0.0):
    """
    Compare performance metrics between portfolios using the correct formula
    with improved calculation and display
    """
    # Handle missing data
    mv_returns = mv_returns[np.isfinite(mv_returns)]
    vw_returns = vw_returns[np.isfinite(vw_returns)]

    # Initialize metrics dictionary
    metrics = {}

    for name, returns in [('Minimum Variance', mv_returns), ('Value-Weighted', vw_returns)]:
        if len(returns) == 0:
            logging.warning(f"No valid returns for {name} portfolio")
            metrics[name] = {
                'Annual Return (%)': np.nan,
                'Annual Volatility (%)': np.nan,
                'Sharpe Ratio': np.nan,
                'Min Return (%)': np.nan,
                'Max Return (%)': np.nan
            }
            continue

        # Calculate annualized metrics properly
        ann_factor = 12  # Monthly to annual

        # Calculate cumulative return
        cumulative_return = (1 + returns).prod()
        n_periods = len(returns)

        # Annualized return using geometric mean
        annual_return = cumulative_return ** (ann_factor / n_periods) - 1

        # Annualized volatility
        annual_vol = returns.std() * np.sqrt(ann_factor)

        # Sharpe ratio
        excess_return = annual_return - risk_free_rate
        sharpe = excess_return / annual_vol if annual_vol > 0 else np.nan

        # Metrics as percentages for better readability
        metrics[name] = {
            'Annual Return (%)': annual_return * 100,
            'Annual Volatility (%)': annual_vol * 100,
            'Sharpe Ratio': sharpe,
            'Min Return (%)': returns.min() * 100,
            'Max Return (%)': returns.max() * 100
        }

        logging.info(
            f"{name} portfolio metrics: Return={annual_return * 100:.2f}%, Vol={annual_vol * 100:.2f}%, SR={sharpe:.4f}")

    # Create DataFrame for display
    metrics_df = pd.DataFrame(metrics).T

    # Round values for better display
    metrics_df = metrics_df.round(2)
    metrics_df['Sharpe Ratio'] = metrics_df['Sharpe Ratio'].round(3)

    # Return the DataFrame (fixed the missing return)
    return metrics_df


def calculate_annual_rebalancing_returns(weights_dict, returns_df):
    """
    Calculate portfolio returns with annual rebalancing given pre-computed weights,
    with improved error handling
    """
    logging.info("Calculating portfolio returns with annual rebalancing...")

    # Convert to numeric and datetime
    returns_data = returns_df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
    date_columns = pd.to_datetime(returns_data.columns, errors='coerce')
    returns_data.columns = date_columns
    returns_data = returns_data.T

    # Filter for 2014-2023
    date_mask = (returns_data.index >= '2014-01-01') & (returns_data.index <= '2023-12-31')
    returns_data = returns_data[date_mask]

    # Initialize outputs
    portfolio_returns = []
    dates = []

    # Sort dates for proper chronological processing
    sorted_dates = sorted(weights_dict.keys())

    # Process each rebalancing period
    for i in range(len(sorted_dates) - 1):
        start_date = sorted_dates[i]
        end_date = sorted_dates[i + 1]

        logging.info(f"Processing period {start_date} to {end_date}...")

        # Get weights for this period
        weights = weights_dict[start_date]
        if weights is None or len(weights) == 0:
            logging.warning(f"No valid weights for {start_date}, skipping period")
            continue

        # Filter returns for this period
        period_mask = (returns_data.index > start_date) & (returns_data.index <= end_date)
        period_returns = returns_data[period_mask]

        if len(period_returns) == 0:
            logging.warning(f"No returns data for period {start_date} to {end_date}, skipping")
            continue

        # Track weights throughout the period
        current_weights = weights.copy()

        # Calculate returns for each month
        for idx, date in enumerate(period_returns.index):
            monthly_returns = period_returns.loc[date]

            # Find common assets between weights and returns
            common_index = current_weights.index.intersection(monthly_returns.index)
            weights_aligned = current_weights[common_index]
            returns_aligned = monthly_returns[common_index]

            # Skip if no valid overlap
            if len(weights_aligned) == 0 or weights_aligned.sum() == 0:
                logging.warning(f"No valid weights/returns overlap for {date}, skipping")
                continue

            # Normalize weights if needed
            if abs(weights_aligned.sum() - 1.0) > 1e-4:
                weights_aligned = weights_aligned / weights_aligned.sum()

            # Calculate portfolio return
            portfolio_return = np.sum(weights_aligned * returns_aligned.fillna(0))
            portfolio_returns.append(portfolio_return)
            dates.append(date)

            # Update weights for next month (except for last month)
            if idx < len(period_returns) - 1:
                denominator = 1 + portfolio_return
                if abs(denominator) > 1e-8:
                    numerator = current_weights * (1 + monthly_returns.fillna(0))
                    current_weights = numerator / denominator

                    # Renormalize weights
                    valid_weights = ~current_weights.isna() & (current_weights > 0)
                    if valid_weights.sum() > 0:
                        current_weights[valid_weights] = current_weights[valid_weights] / current_weights[
                            valid_weights].sum()
                        current_weights[~valid_weights] = 0

    # Convert to Series
    if portfolio_returns:
        return pd.Series(portfolio_returns, index=dates)
    else:
        logging.error("No portfolio returns could be calculated")
        return pd.Series(dtype=float)