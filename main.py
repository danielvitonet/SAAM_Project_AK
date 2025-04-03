###############################################################################
# SUSTAINABILITY AWARE ASSET MANAGEMENT
# =============================================================================
# GROUP MEMBERS:
# Antonio Lavenia
# Daniel Vito Lobasso
# Andrea Marchese
# Thomas Nava
# Daniele Parini
# =============================================================================
# Project: "Asset Allocation with a Carbon Objective"
# Goal: Implement climate aware asset management concepts seen in class.
###############################################################################

# The code is optimized for Python 3.11.

###############################################################################
# PART 1: Standard Asset Allocation
# Building a portfolio based on the mean-variance criterion.
###############################################################################

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set the working directory to the directory where main.py is located.
base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(base_dir)
sys.path.insert(0, base_dir)
print("Current working directory:", os.getcwd())

# Import the necessary functions from the DataHandler package
from DataHandler.Data_SetUP import Initializer
from DataHandler.Standard_Asset_Allocation import run_portfolio_optimization
# Import the new functions for value-weighted portfolio
from DataHandler.Value_Weighted_Portfolio import calculate_value_weighted_portfolio, plot_cumulative_returns, \
    compare_portfolio_performance


def clean_date_columns(df):
    """Helper function to ensure dates in the columns are in YYYY-MM-DD format"""
    meta_cols = list(df.columns[:2])  # Keep first two columns as metadata
    date_cols = list(df.columns[2:])  # Get the date columns

    # Try to convert date columns to standard format
    formatted_date_cols = []
    for col in date_cols:
        try:
            # If it's already a date object, format it as string
            if isinstance(col, pd.Timestamp):
                formatted_date_cols.append(col.strftime('%Y-%m-%d'))
            else:
                # If it's a string, try to parse and format
                dt = pd.to_datetime(col)
                formatted_date_cols.append(dt.strftime('%Y-%m-%d'))
        except:
            # If conversion fails, keep as is
            formatted_date_cols.append(col)

    # Create new DataFrame with formatted column names
    new_df = df.copy()
    new_df.columns = meta_cols + formatted_date_cols
    return new_df


def main():
    # Define file paths using os.path.join for portability.
    static_file = os.path.join("Data", "Static.xlsx")
    original_data_folder = os.path.join("Data", "Original Data")
    filtered_data_folder = os.path.join("Data", "Filtered Data")

    # Call the Initializer:
    #  - Filters raw datasets, loads the filtered datasets, prints summary statistics,
    #  - Calculates simple returns from DS_RI_T_USD_M.xlsx and adds them to the output dictionary.
    filtered_datasets = Initializer(
        static_file,
        original_data_folder,
        filtered_data_folder,
        produce_visuals=False  # Change to True if you want to see graphs.
    )

    print("\nAll filtered datasets and simple returns have been loaded.")
    print("Ready for portfolio construction.")

    # ------------------------
    # PART 1.1
    # ------------------------
    # Retrieve the simple returns DataFrame.
    # It was added to the dictionary with key "Simple_Returns.xlsx".

    returns_df = filtered_datasets.get("Simple_Returns.xlsx")
    if returns_df is None:
        returns_df = pd.read_excel(os.path.join("Data", "Simple_Returns.xlsx"))

    # Run the portfolio optimization for minimum variance portfolio
    mv_metrics, mv_returns = run_portfolio_optimization(returns_df)

    # Print results for minimum variance portfolio
    print("\nMinimum Variance Portfolio Characteristics (P(mv)oos):")
    print(f"Annualized Average Return (μ̄p): {mv_metrics['annualized_return']:.4f}")
    print(f"Annualized Volatility (σp): {mv_metrics['annualized_volatility']:.4f}")
    print(f"Average Risk-free Rate: {mv_metrics['avg_rf_rate']:.4f}")
    print(f"Sharpe Ratio (SRp): {mv_metrics['sharpe_ratio']:.4f}")
    print(f"Minimum Return: {mv_metrics['min_return']:.4f}")
    print(f"Maximum Return: {mv_metrics['max_return']:.4f}")

    # ------------------------
    # PART 1.2
    # ------------------------
    # Retrieve the market capitalization DataFrame
    market_cap_file = os.path.join(filtered_data_folder, "DS_MV_T_USD_M.xlsx")
    if os.path.exists(market_cap_file):
        print("\nLoading market capitalization data...")

        # Read market cap data
        market_cap_df = pd.read_excel(market_cap_file)
        print("Market capitalization data loaded. Shape:", market_cap_df.shape)

        # Clean and standardize column names (especially dates)
        market_cap_df = clean_date_columns(market_cap_df)
        returns_df = clean_date_columns(returns_df)

        # Print some sample column names to verify formats
        print("\nSample market cap columns:", market_cap_df.columns[2:10])
        print("Sample returns columns:", returns_df.columns[2:10])

        # Calculate value-weighted portfolio returns
        print("\nCalculating value-weighted portfolio returns...")

        vw_returns = calculate_value_weighted_portfolio(
            market_cap_df,
            returns_df,
            start_date="2014-01-01",  # Starting from Jan 2014 (after initialization period)
            end_date="2023-12-31"  # Ending in Dec 2023
        )

        print(f"Value-weighted portfolio returns calculated. Length: {len(vw_returns)}")

        if len(vw_returns) > 0:
            print("First 5 value-weighted returns:")
            print(vw_returns.head())

        # Get risk-free rates
        rf_file = os.path.join("Data", "Risk_Free_Rate.xlsx")
        try:
            rf_data = pd.read_excel(rf_file)
            dates = pd.to_datetime(rf_data.iloc[:, 0].astype(str).str.pad(6, fillchar='0'), format='%Y%m')
            rates = pd.to_numeric(rf_data.iloc[:, 1], errors='coerce') / 100.0
            rf_rates = pd.Series(rates.values, index=dates)
        except Exception as e:
            print(f"\nError reading risk-free rates: {str(e)}")
            rf_rates = None

        # Compare portfolio performance
        print("\nComparing portfolio performances...")
        comparison_df = compare_portfolio_performance(mv_returns, vw_returns, rf_rates)

        print("\nPortfolio Performance Comparison:")
        print(comparison_df)

        # Plot cumulative returns
        print("\nPlotting cumulative returns...")
        cumulative_returns = plot_cumulative_returns(
            mv_returns,
            vw_returns,
            title="Cumulative Returns: Minimum Variance vs. Value-Weighted"
        )

        # Save the returns to CSV for future reference
        results_dir = os.path.join("Data", "Results")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Convert index to string before saving to avoid issues
        mv_returns_df = pd.DataFrame(mv_returns)
        mv_returns_df.index = mv_returns_df.index.astype(str)
        mv_returns_df.to_csv(os.path.join(results_dir, "mv_returns.csv"))

        vw_returns_df = pd.DataFrame(vw_returns)
        vw_returns_df.index = vw_returns_df.index.astype(str)
        vw_returns_df.to_csv(os.path.join(results_dir, "vw_returns.csv"))

        comparison_df.to_csv(os.path.join(results_dir, "portfolio_comparison.csv"))

        print("\nResults saved to Data/Results directory.")
        print("Point 1.2 completed: Value-weighted portfolio calculated and compared with minimum variance portfolio.")
    else:
        print(f"Market capitalization file not found: {market_cap_file}")
        print("Cannot complete point 1.2 without market capitalization data.")

    print("\nOptimization routine complete. Portfolio characteristics have been computed over the sample.")


if __name__ == "__main__":
    main()

###############################################################################
# PART 2: Impact of the portfolio on climate
# Asset Allocation with a Carbon Emissions Reduction.
# Asset Allocation with a Net Zero Objective.
###############################################################################


# Main file