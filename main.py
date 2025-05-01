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
import cvxpy as cp
from DataHandler.Data_SetUP import Initializer
from DataHandler.Standard_Asset_Allocation import run_portfolio_optimization
from DataHandler.Value_Weighted_Portfolio import calculate_value_weighted_portfolio, plot_cumulative_returns, compare_portfolio_performance
from DataHandler.CarbonAwarePortfolio import CarbonAwarePortfolio

# Set the working directory to the directory where main.py is located.
base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(base_dir)
sys.path.insert(0, base_dir)
print("Current working directory:", os.getcwd())

# Import the necessary functions from the DataHandler package
from DataHandler.Data_SetUP import Initializer
from DataHandler.Standard_Asset_Allocation import run_portfolio_optimization
from DataHandler.Value_Weighted_Portfolio import calculate_value_weighted_portfolio, plot_cumulative_returns, \
    compare_portfolio_performance

# Import CarbonAwarePortfolio class
from DataHandler.CarbonAwarePortfolio import CarbonAwarePortfolio


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
        market_cap_df.columns = list(market_cap_df.columns[:2]) + [
            str(col) for col in market_cap_df.columns[2:]
        ]
        returns_df.columns = list(returns_df.columns[:2]) + [
            str(col) for col in returns_df.columns[2:]
        ]

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

    ###############################################################################
    # PART 2: Asset Allocation with a Carbon Emissions Reduction
    # Adding a carbon footprint constraint to portfolio optimization.
    ###############################################################################

    print("\n" + "=" * 80)
    print("PART 2: Asset Allocation with a Carbon Emissions Reduction")
    print("=" * 80)

    # Load necessary data for carbon-aware portfolio construction
    data_dir = filtered_data_folder
    market_cap_annual_file = os.path.join(data_dir, "DS_MV_T_USD_Y.xlsx")
    scope1_file = os.path.join(data_dir, "Scope_1.xlsx")
    scope2_file = os.path.join(data_dir, "Scope_2.xlsx")
    revenue_file = os.path.join(data_dir, "DS_REV_USD_Y.xlsx")
    static_file = os.path.join("Data", "Static.xlsx")

    # Check if all required files exist
    required_files = [market_cap_annual_file, scope1_file, scope2_file, revenue_file, static_file]
    all_files_exist = all(os.path.exists(file) for file in required_files)

    if all_files_exist:
        # Load data
        print("Loading data for carbon-aware portfolio analysis...")
        market_cap_annual_df = pd.read_excel(market_cap_annual_file)
        scope1_df = pd.read_excel(scope1_file)
        scope2_df = pd.read_excel(scope2_file)
        revenue_df = pd.read_excel(revenue_file)
        static_df = pd.read_excel(static_file)

        # Create CarbonAwarePortfolio instance with error handling
        try:
            carbon_portfolio = CarbonAwarePortfolio(
                market_cap_annual_df=market_cap_annual_df,
                returns_df=returns_df,
                scope1_df=scope1_df,
                scope2_df=scope2_df,
                revenue_df=revenue_df,
                static_df=static_df  # Add the static dataframe
            )

            # Print data quality metrics
            print("\nData quality check:")
            carbon_portfolio.check_data_quality()

            # ------------------------
            # PART 2.1
            # ------------------------
            print("\n" + "-" * 80)
            print("PART 2.1: Computing carbon intensity and carbon footprint")
            print("-" * 80)

            # Compute portfolio weights and carbon footprints
            try:
                carbon_portfolio.compute_all_portfolio_weights(
                    window_size=120, start_year=2013, end_year=2023
                )

                # Print diagnostic information
                print("\nNumber of years successfully processed:",
                      len([k for k in carbon_portfolio.carbon_footprints.keys() if k.startswith('mv_')]))

                # Print carbon footprints of the minimum variance portfolio
                print("\nCarbon Footprints of the Minimum Variance Portfolio (P(mv)oos):")
                for year in range(2014, 2024):
                    if f"mv_{year}" in carbon_portfolio.carbon_footprints:
                        waci, cf = carbon_portfolio.carbon_footprints[f"mv_{year}"]
                        print(f"Year {year}: WACI = {waci:.2f}, Carbon Footprint = {cf:.2f}")
                    else:
                        print(f"Year {year}: Data not available")

                # ------------------------
                # PART 2.2
                # ------------------------
                print("\n" + "-" * 80)
                print("PART 2.2: Constructing portfolio with 50% reduction in carbon footprint vs. minimum variance")
                print("-" * 80)

                # Print carbon footprints of the carbon-constrained minimum variance portfolio
                print("\nCarbon Footprints of the Carbon-Constrained Minimum Variance Portfolio (P(mv)oos(0.5)):")
                for year in range(2014, 2024):
                    if f"mvc_{year}" in carbon_portfolio.carbon_footprints:
                        waci, cf = carbon_portfolio.carbon_footprints[f"mvc_{year}"]
                        print(f"Year {year}: WACI = {waci:.2f}, Carbon Footprint = {cf:.2f}")

                        # Calculate reduction percentage if both footprints are available
                        if f"mv_{year}" in carbon_portfolio.carbon_footprints:
                            _, mv_cf = carbon_portfolio.carbon_footprints[f"mv_{year}"]
                            reduction = 100 * (1 - cf / mv_cf) if mv_cf > 0 else 0
                            print(f"          Carbon Footprint Reduction: {reduction:.2f}%")
                    else:
                        print(f"Year {year}: Data not available")

                # ------------------------
                # PART 2.3
                # ------------------------
                print("\n" + "-" * 80)
                print("PART 2.3: Constructing portfolio with 50% reduction in carbon footprint vs. value-weighted")
                print("-" * 80)

                # Print carbon footprints of the carbon-constrained value-weighted portfolio
                print("\nCarbon Footprints of the Carbon-Constrained Value-Weighted Portfolio (P(vw)oos(0.5)):")
                for year in range(2014, 2024):
                    if f"vwc_{year}" in carbon_portfolio.carbon_footprints:
                        waci, cf = carbon_portfolio.carbon_footprints[f"vwc_{year}"]
                        print(f"Year {year}: WACI = {waci:.2f}, Carbon Footprint = {cf:.2f}")

                        # Calculate reduction percentage if both footprints are available
                        if f"vw_{year}" in carbon_portfolio.carbon_footprints:
                            _, vw_cf = carbon_portfolio.carbon_footprints[f"vw_{year}"]
                            reduction = 100 * (1 - cf / vw_cf) if vw_cf > 0 else 0
                            print(f"          Carbon Footprint Reduction: {reduction:.2f}%")
                    else:
                        print(f"Year {year}: Data not available")

                # ------------------------
                # PART 2.4
                # ------------------------
                print("\n" + "-" * 80)
                print("PART 2.4: Analyzing trade-off between financial performance and carbon footprint reduction")
                print("-" * 80)

                # Plot carbon footprints over time
                carbon_portfolio.plot_carbon_footprints(start_year=2014, end_year=2023)
                print("\nCarbon footprints plot generated: carbon_footprints.png")

                # Generate trade-off visualization
                carbon_portfolio.plot_performance_carbon_tradeoff(start_year=2014, end_year=2023)
                print("Performance-carbon trade-off plot generated: performance_carbon_tradeoff.png")

                ###############################################################################
                # PART 3: Allocation with a Net Zero Objective
                # Implementing a decreasing carbon footprint over time.
                ###############################################################################

                print("\n" + "=" * 80)
                print("PART 3: Allocation with a Net Zero Objective")
                print("=" * 80)

                # ------------------------
                # PART 3.1
                # ------------------------
                print("\n" + "-" * 80)
                print("PART 3.1: Implementing a decarbonization strategy with 10% reduction per year")
                print("-" * 80)

                # Print carbon footprints of the net zero portfolio
                print("\nCarbon Footprints of the Net Zero Portfolio (P(vw)oos(NZ)):")
                target_reductions = {}
                actual_reductions = {}

                for year in range(2014, 2024):
                    target_reduction = 100 * (1 - 0.9 ** (year - 2013))
                    target_reductions[year] = target_reduction

                    if f"nz_{year}" in carbon_portfolio.carbon_footprints:
                        waci, cf = carbon_portfolio.carbon_footprints[f"nz_{year}"]
                        print(f"Year {year}: WACI = {waci:.2f}, Carbon Footprint = {cf:.2f}")

                        # Calculate reduction percentage from base year (2013)
                        if f"vw_2013" in carbon_portfolio.carbon_footprints:
                            _, base_cf = carbon_portfolio.carbon_footprints[f"vw_2013"]
                            cumulative_reduction = 100 * (1 - cf / base_cf) if base_cf > 0 else 0
                            actual_reductions[year] = cumulative_reduction
                            print(f"          Cumulative Reduction from 2013: {cumulative_reduction:.2f}%")
                            print(f"          Target Reduction: {target_reduction:.2f}%")
                            print(f"          Gap: {(cumulative_reduction - target_reduction):.2f}%")
                    else:
                        print(f"Year {year}: Data not available")

                # Create a plot of actual vs target reduction
                plt.figure(figsize=(12, 8))
                years = sorted(target_reductions.keys())
                target_values = [target_reductions[y] for y in years]

                plt.plot(years, target_values, 'r--', label='Target Reduction')

                if actual_reductions:
                    actual_years = sorted(actual_reductions.keys())
                    actual_values = [actual_reductions[y] for y in actual_years]
                    plt.plot(actual_years, actual_values, 'b-', label='Actual Reduction')

                plt.title('Net Zero Strategy: Target vs. Actual Carbon Footprint Reduction')
                plt.xlabel('Year')
                plt.ylabel('Cumulative Reduction (%)')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.savefig('net_zero_reduction.png', dpi=300)
                plt.close()

                print("\nNet Zero reduction plot generated: net_zero_reduction.png")

                # ------------------------
                # PART 3.2
                # ------------------------
                print("\n" + "-" * 80)
                print("PART 3.2: Comparing performance of value-weighted, carbon-constrained, and net zero portfolios")
                print("-" * 80)

                # Calculate and print performance metrics for all portfolios
                metrics_df = carbon_portfolio.compare_portfolio_performance(
                    start_year=2014, end_year=2023
                )

                print("\nPortfolio Performance Metrics:")
                print(metrics_df)

                # Save metrics to CSV
                results_dir = os.path.join("Data", "Results")
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)

                metrics_df.to_csv(os.path.join(results_dir, "carbon_portfolio_metrics.csv"))

                # Create a more detailed plot for cumulative returns comparison
                print("\nCreating cumulative returns plot...")

                # Compute returns for all portfolios
                mv_returns = carbon_portfolio.compute_portfolio_returns(carbon_portfolio.mv_portfolio_weights, 2014,
                                                                        2023)
                vw_returns = carbon_portfolio.compute_portfolio_returns(carbon_portfolio.vw_portfolio_weights, 2014,
                                                                        2023)
                mvc_returns = carbon_portfolio.compute_portfolio_returns(carbon_portfolio.mv_carbon_weights, 2014, 2023)
                vwc_returns = carbon_portfolio.compute_portfolio_returns(carbon_portfolio.vw_carbon_weights, 2014, 2023)
                nz_returns = carbon_portfolio.compute_portfolio_returns(carbon_portfolio.nz_portfolio_weights, 2014,
                                                                        2023)

                # Combine all returns in a DataFrame
                returns_df = pd.DataFrame({
                    'Minimum Variance': mv_returns,
                    'Value-Weighted': vw_returns,
                    'MV Carbon-Constrained': mvc_returns,
                    'VW Carbon-Constrained': vwc_returns,
                    'Net Zero': nz_returns
                })

                # Calculate cumulative returns
                cum_returns = (1 + returns_df).cumprod()

                # Create plot
                plt.figure(figsize=(15, 10))

                # Main plot: all portfolios
                plt.subplot(2, 1, 1)
                for col in cum_returns.columns:
                    plt.plot(cum_returns.index, cum_returns[col], label=col)

                plt.title('Cumulative Returns of All Portfolios')
                plt.xlabel('Date')
                plt.ylabel('Cumulative Return')
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Subplot: Net Zero vs. VW comparison
                plt.subplot(2, 2, 3)
                plt.plot(cum_returns.index, cum_returns['Value-Weighted'], label='Value-Weighted')
                plt.plot(cum_returns.index, cum_returns['VW Carbon-Constrained'], label='VW Carbon-Const.')
                plt.plot(cum_returns.index, cum_returns['Net Zero'], label='Net Zero')
                plt.title('Value-Weighted Portfolios Comparison')
                plt.xlabel('Date')
                plt.ylabel('Cumulative Return')
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Subplot: MV vs. MVC comparison
                plt.subplot(2, 2, 4)
                plt.plot(cum_returns.index, cum_returns['Minimum Variance'], label='Min. Variance')
                plt.plot(cum_returns.index, cum_returns['MV Carbon-Constrained'], label='MV Carbon-Const.')
                plt.title('Minimum Variance Portfolios Comparison')
                plt.xlabel('Date')
                plt.ylabel('Cumulative Return')
                plt.legend()
                plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig('cumulative_returns_all_strategies.png', dpi=300)
                plt.close()

                print("\nCumulative returns plot generated: cumulative_returns_all_strategies.png")

                # Save the returns to CSV for future reference
                returns_df.to_csv(os.path.join(results_dir, "all_portfolio_returns.csv"))

                print("\nAll portfolio returns saved: all_portfolio_returns.csv")
                print("\nCarbon-aware portfolio analysis complete.")
                print("Results and visualizations saved to the Data/Results directory.")

            except Exception as e:
                print(f"Error during portfolio weight computation: {str(e)}")
                import traceback
                traceback.print_exc()

        except Exception as e:
            print(f"Error during CarbonAwarePortfolio initialization: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        missing_files = [file for file in required_files if not os.path.exists(file)]
        print(f"Cannot complete Part II and III. Missing files: {missing_files}")

    print("\nOptimization routine complete. Portfolio characteristics have been computed over the sample.")


if __name__ == "__main__":
    main()