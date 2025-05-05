###############################################################################
# SUSTAINABILITY AWARE ASSET MANAGEMENT
# =============================================================================
# GROUP MEMBERS:
# Group AK: Europe / Scope 1+2
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
from pathlib import Path
import logging
import cvxpy as cp
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("project_execution.log"),
        logging.StreamHandler()
    ]
)

# Add project directory to path
project_dir = Path(__file__).parent
sys.path.append(str(project_dir))

# Import necessary modules
from DataHandler.Data_SetUP import Initializer
from DataHandler.Standard_Asset_Allocation import run_portfolio_optimization
from DataHandler.Value_Weighted_Portfolio import (
    calculate_value_weighted_portfolio,
    plot_cumulative_returns,
    compare_portfolio_performance
)
from DataHandler.CarbonAwarePortfolio import CarbonAwarePortfolio


def main():
    """
    Main execution function for the Sustainability Aware Asset Management project
    """
    # Define paths
    data_dir = project_dir / "Data"
    original_data_dir = data_dir / "Original Data"
    filtered_data_dir = data_dir / "Filtered Data"
    failed_data_dir = data_dir / "Failed Companies"
    results_dir = data_dir / "Results"

    # Create directories
    for directory in [filtered_data_dir, failed_data_dir, results_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # File paths
    static_file = data_dir / "Static.xlsx"

    logging.info("=" * 80)
    logging.info("SUSTAINABILITY AWARE ASSET MANAGEMENT PROJECT")
    logging.info("Group AK: Europe / Scope 1+2")
    logging.info("=" * 80)

    # Part 1: Data Initialization and Standard Asset Allocation
    logging.info("\nPART 1: Standard Asset Allocation")
    logging.info("-" * 40)

    # Initialize data
    logging.info("Initializing data...")
    processed_data = Initializer(
        str(static_file),
        str(original_data_dir),
        str(filtered_data_dir),
        str(failed_data_dir),
        max_missing=0.2  # 20% threshold as specified
    )

    # Get returns data, preferring Fixed_Returns.csv if available
    returns_df = processed_data.get('Fixed_Returns.csv')
    if returns_df is None or returns_df.empty:
        logging.info("Fixed_Returns.csv not found or empty. Falling back to Simple_Returns.csv.")
        returns_df = processed_data.get('Simple_Returns.csv')

    if returns_df is None or returns_df.empty:
        logging.error("Returns data not found. Check data initialization.")
        raise ValueError("Returns data not found. Check data initialization.")

    # Validate returns_df structure
    required_cols = ['ISIN']
    if not all(col in returns_df.columns for col in required_cols):
        logging.error(f"returns_df missing required columns: {required_cols}")
        raise ValueError("returns_df missing required columns")

    # Check for valid date columns
    date_cols = [col for col in returns_df.columns if col not in ['ISIN', 'NAME', 'Name']]
    if not date_cols:
        logging.error("No valid date columns in returns_df")
        raise ValueError("No valid date columns in returns_df")

    logging.info(f"returns_df: {len(returns_df)} ISINs, {len(date_cols)} date columns")

    # Part 1.1: Minimum Variance Portfolio
    logging.info("\nComputing minimum variance portfolio...")
    logging.info(f"returns_df shape: {returns_df.shape}")
    date_range = pd.to_datetime(returns_df.columns[2:],
    errors = 'coerce')
    min_date = date_range.min()
    max_date = date_range.max()
    logging.info(f"returns_df date range: {min_date} to {max_date}")

    try:
        mv_metrics, mv_returns, mv_weights, valid_cols_dict = run_portfolio_optimization(
            returns_df,
            window_size=120
        )

        if not mv_weights or mv_returns.empty or all(np.isnan(v) for v in mv_metrics.values()):
            logging.warning("Portfolio optimization failed: no valid portfolios generated.")
            mv_metrics = {
                'annualized_return': np.nan,
                'annualized_volatility': np.nan,
                'sharpe_ratio': np.nan,
                'min_return': np.nan,
                'max_return': np.nan
            }
            mv_returns = pd.Series(dtype=float)
            mv_weights = {}
            for year in range(2014, 2024):
                rebalance_date = pd.Timestamp(f"{year - 1}-12-31")
                mv_weights[rebalance_date] = pd.Series(
                    1 / len(returns_df['ISIN']), index=returns_df['ISIN']
                )
        else:
            mv_weights_series = {}
            for date, weights in mv_weights.items():
                if isinstance(weights, pd.Series):
                    mv_weights_series[date] = weights.reindex(returns_df['ISIN'], fill_value=0)
                else:
                    valid_cols = valid_cols_dict.get(date, np.arange(len(weights)))
                    valid_isins = returns_df['ISIN'].iloc[valid_cols]
                    mv_weights_series[date] = pd.Series(weights, index=valid_isins).reindex(returns_df['ISIN'],
                                                                                            fill_value=0)
            mv_weights = mv_weights_series

        logging.info("\nMinimum Variance Portfolio Results (P(mv)oos):")
        if all(np.isnan(v) for v in mv_metrics.values()):
            logging.warning("MV portfolio metrics are NaN. Check optimization.")
        else:
            logging.info(f"Annualized Return: {mv_metrics['annualized_return']:.4f}")
            logging.info(f"Annualized Volatility: {mv_metrics['annualized_volatility']:.4f}")
            logging.info(f"Sharpe Ratio: {mv_metrics['sharpe_ratio']:.4f}")
            logging.info(f"Minimum Return: {mv_metrics['min_return']:.4f}")
            logging.info(f"Maximum Return: {mv_metrics['max_return']:.4f}")
            pd.DataFrame([mv_metrics]).to_csv(results_dir / "mv_metrics.csv", index=False)
    except Exception as e:
        logging.error(f"Error in portfolio optimization: {str(e)}.")
        logging.error(traceback.format_exc())
        logging.warning("Using fallback equal weights.")
        mv_metrics = {
            'annualized_return': np.nan,
            'annualized_volatility': np.nan,
            'sharpe_ratio': np.nan,
            'min_return': np.nan,
            'max_return': np.nan
        }
        mv_returns = pd.Series(dtype=float)
        mv_weights = {}
        for year in range(2014, 2024):
            rebalance_date = pd.Timestamp(f"{year - 1}-12-31")
            mv_weights[rebalance_date] = pd.Series(
                1 / len(returns_df['ISIN']), index=returns_df['ISIN']
            )

    # Part 1.2: Value-Weighted Portfolio
    logging.info("\nComputing value-weighted portfolio...")
    market_cap_df = processed_data.get('DS_MV_T_USD_M.csv')

    if market_cap_df is None or market_cap_df.empty:
        logging.error("Market cap data not found.")
        raise ValueError("Market cap data not found.")

    # Unpack the tuple returned by calculate_value_weighted_portfolio
    vw_returns, vw_weights_dict = calculate_value_weighted_portfolio(market_cap_df, returns_df)

    # Compare portfolios
    try:
        comparison_df = compare_portfolio_performance(mv_returns, vw_returns)
        logging.info("\nPortfolio Comparison:")
        if comparison_df is not None:
            logging.info(f"\n{comparison_df}")
            comparison_df.to_csv(results_dir / "portfolio_comparison_part1.csv")
        else:
            logging.error("Could not generate portfolio comparison")
    except Exception as e:
        logging.error(f"Error in portfolio comparison: {str(e)}")
        comparison_df = pd.DataFrame()
        comparison_df.to_csv(results_dir / "portfolio_comparison_part1.csv")

    # Plot cumulative returns
    if mv_returns.empty and vw_returns.empty:
        logging.error("No valid returns for plotting cumulative returns.")
    else:
        plot_cumulative_returns(
            mv_returns,
            vw_returns,
            str(results_dir / "cumulative_returns_part1.png")
        )

    # Save Part 1 results
    mv_returns.to_csv(results_dir / "mv_returns.csv")
    vw_returns.to_csv(results_dir / "vw_returns.csv")
    comparison_df.to_csv(results_dir / "portfolio_comparison_part1.csv")

    # Part 2: Carbon-Aware Portfolio Allocation
    logging.info("\n" + "=" * 80)
    logging.info("PART 2: Asset Allocation with Carbon Emissions Reduction")
    logging.info("=" * 80)

    # Load additional data for carbon analysis
    scope1_df = processed_data.get('Scope_1.csv')
    scope2_df = processed_data.get('Scope_2.csv')
    revenue_df = processed_data.get('DS_REV_USD_Y.csv')
    market_cap_annual_df = processed_data.get('DS_MV_T_USD_Y.csv')

    # Fallback for missing annual market cap data
    if market_cap_annual_df is None or market_cap_annual_df.empty:
        logging.warning("market_cap_annual_df is missing or empty. Attempting to derive from monthly data.")
        market_cap_monthly_df = processed_data.get('DS_MV_T_USD_M.csv')
        if market_cap_monthly_df is not None and not market_cap_monthly_df.empty:
            metadata_cols = ['ISIN', 'NAME'] if 'NAME' in market_cap_monthly_df.columns else ['ISIN']
            date_cols = [col for col in market_cap_monthly_df.columns if col not in metadata_cols]
            market_cap_annual_df = pd.DataFrame({'ISIN': market_cap_monthly_df['ISIN']})
            if 'NAME' in market_cap_monthly_df.columns:
                market_cap_annual_df['NAME'] = market_cap_monthly_df['NAME']

            for year in range(2013, 2024):
                year_cols = [col for col in date_cols if pd.to_datetime(col, errors='coerce').year == year]
                if year_cols:
                    market_cap_annual_df[str(year)] = market_cap_monthly_df[year_cols].mean(axis=1)
            if market_cap_annual_df.drop(
                    columns=['ISIN', 'NAME'] if 'NAME' in market_cap_annual_df.columns else ['ISIN']).empty:
                logging.error("Failed to derive annual market cap data from monthly data.")
                raise ValueError("market_cap_annual_df is missing or empty.")
        else:
            logging.error("Both annual and monthly market cap data are missing.")
            raise ValueError("market_cap_annual_df is missing or empty.")

    # Verify that all required data is available
    for df_name, df in [
        ('scope1_df', scope1_df),
        ('scope2_df', scope2_df),
        ('revenue_df', revenue_df),
        ('market_cap_annual_df', market_cap_annual_df)
    ]:
        if df is None or df.empty:
            logging.error(f"{df_name} is missing or empty. Carbon-aware portfolios cannot be constructed.")
            raise ValueError(f"{df_name} is missing or empty.")

    # Create CarbonAwarePortfolio instance
    try:
        carbon_portfolio = CarbonAwarePortfolio(
            market_cap_annual_df,
            returns_df,
            scope1_df,
            scope2_df,
            revenue_df
        )
    except Exception as e:
        logging.error(f"Failed to create CarbonAwarePortfolio: {str(e)}")
        logging.error(traceback.format_exc())
        raise

    # Part 2.1: Calculate carbon footprint of minimum variance portfolio
    logging.info("\nPart 2.1: Computing carbon footprints...")

    mv_carbon_footprints = {}
    vw_carbon_footprints = {}

    years = range(2014, 2024)
    for year in years:
        rebalance_date = pd.Timestamp(f"{year - 1}-12-31")
        closest_mv_date = min(mv_weights.keys(), key=lambda x: abs(x - rebalance_date)) if mv_weights else None
        mv_weights_year = mv_weights.get(closest_mv_date)
        if mv_weights_year is not None:
            mv_weights_year = carbon_portfolio.fix_invalid_weights(mv_weights_year)
            mv_cf = carbon_portfolio.calculate_portfolio_carbon_footprint(mv_weights_year, year - 1)
            if mv_cf == 0 or np.isnan(mv_cf):
                logging.warning(f"Invalid MV carbon footprint for {year}: {mv_cf}. Using equal weights.")
                mv_weights_year = pd.Series(1 / len(carbon_portfolio.isins), index=carbon_portfolio.isins)
                mv_cf = carbon_portfolio.calculate_portfolio_carbon_footprint(mv_weights_year, year - 1)
            mv_carbon_footprints[year] = mv_cf
            logging.info(f"Year {year}: MV Carbon Footprint = {mv_cf:.2f}")
        else:
            logging.warning(f"No MV weights found for {year}.")
            mv_carbon_footprints[year] = np.nan

        closest_vw_date = min(vw_weights_dict.keys(),
                              key=lambda x: abs(x - rebalance_date)) if vw_weights_dict else None
        vw_weights_year = vw_weights_dict.get(closest_vw_date)
        if vw_weights_year is not None:
            vw_weights_year = carbon_portfolio.fix_invalid_weights(vw_weights_year)
            vw_cf = carbon_portfolio.calculate_portfolio_carbon_footprint(vw_weights_year, year - 1)
            if vw_cf == 0 or np.isnan(vw_cf):
                logging.warning(f"Invalid VW carbon footprint for {year}: {vw_cf}. Using equal weights.")
                vw_weights_year = pd.Series(1 / len(carbon_portfolio.isins), index=carbon_portfolio.isins)
                vw_cf = carbon_portfolio.calculate_portfolio_carbon_footprint(vw_weights_year, year - 1)
            vw_carbon_footprints[year] = vw_cf
            logging.info(f"Year {year}: VW Carbon Footprint = {vw_cf:.2f}")
        else:
            logging.warning(f"No VW weights found for {year}.")
            vw_carbon_footprints[year] = np.nan

    logging.info("\nCarbon Footprints Summary:")
    for year in years:
        mv_cf = mv_carbon_footprints.get(year, np.nan)
        vw_cf = vw_carbon_footprints.get(year, np.nan)
        logging.info(f"Year {year}: MV={mv_cf:.2f}, VW={vw_cf:.2f}")

    # Part 2.2 and 2.3: Carbon-constrained portfolios
    logging.info("\nPart 2.2 & 2.3: Computing carbon-constrained portfolios...")

    # Run carbon-constrained optimization
    try:
        results = carbon_portfolio.run_carbon_constrained_optimization(
            returns_df, mv_weights, vw_weights_dict,
            start_year=2014, end_year=2023, window_size=120
        )

        # Display carbon-constrained portfolio results
        logging.info("\nCarbon-Constrained Portfolio Results:")
        for year in range(2014, 2024):
            logging.info(f"\nYear {year}:")
            for strategy in ['mvc', 'vwc', 'nz']:
                if year in results[strategy]['carbon_footprints']:
                    cf = results[strategy]['carbon_footprints'][year]
                    strategy_label = {
                        'mvc': 'MVC (MV with 50% reduction)',
                        'vwc': 'VWC (VW with 25% reduction)',
                        'nz': 'NZ (Net Zero)'
                    }[strategy]
                    logging.info(f"  {strategy_label}: {cf:.2f}")
                    if cf == 0:
                        logging.warning(
                            f"Zero carbon footprint for {strategy} in {year}. Verify constraints and emissions data.")
    except Exception as e:
        logging.error(f"Error running carbon-constrained optimization: {str(e)}")
        logging.error(traceback.format_exc())
        results = {
            'mv': {'carbon_footprints': mv_carbon_footprints, 'returns': pd.Series(dtype=float), 'weights': {}},
            'vw': {'carbon_footprints': vw_carbon_footprints, 'returns': pd.Series(dtype=float), 'weights': {}},
            'mvc': {'carbon_footprints': {}, 'returns': pd.Series(dtype=float), 'weights': {}},
            'vwc': {'carbon_footprints': {}, 'returns': pd.Series(dtype=float), 'weights': {}},
            'nz': {'carbon_footprints': {}, 'returns': pd.Series(dtype=float), 'weights': {}}
        }

    # Part 3: Net Zero Portfolio
    logging.info("\n" + "=" * 80)
    logging.info("PART 3: Allocation with Net Zero Objective")
    logging.info("=" * 80)

    logging.info("\nNet Zero Carbon Footprints:")
    first_year_vw_cf = results['vw']['carbon_footprints'].get(2014, None)

    for year in range(2014, 2024):
        if year in results['nz']['carbon_footprints']:
            nz_cf = results['nz']['carbon_footprints'][year]
            years_elapsed = year - 2013
            target_reduction = 100 * (1 - (1 - 0.1) ** years_elapsed)

            if first_year_vw_cf is not None and not np.isnan(first_year_vw_cf):
                actual_reduction = 100 * (1 - nz_cf / first_year_vw_cf)
                logging.info(
                    f"Year {year}: CF = {nz_cf:.2f}, Target = {target_reduction:.1f}%, Actual = {actual_reduction:.1f}%")
            else:
                logging.info(f"Year {year}: CF = {nz_cf:.2f}, Target = {target_reduction:.1f}%")

    # Final reporting and visualization
    logging.info("\nGenerating final reports and visualizations...")

    # Generate comprehensive visualizations and performance summary
    try:
        summary_df = carbon_portfolio.plot_comprehensive_results(results, str(results_dir))
    except Exception as e:
        logging.error(f"Error generating comprehensive results: {str(e)}")
        logging.error(traceback.format_exc())
        summary_df = pd.DataFrame()

    # Create final report
    try:
        report_text = carbon_portfolio.generate_project_report(results, summary_df, str(results_dir))
    except Exception as e:
        logging.error(f"Error generating project report: {str(e)}")
        logging.error(traceback.format_exc())

    # Create summary DataFrame for all strategies
    summary_data = []
    for year in range(2014, 2024):
        row = {'Year': year}
        for strategy in ['mv', 'mvc', 'vw', 'vwc', 'nz']:
            if year in results[strategy]['carbon_footprints']:
                row[f'{strategy}_cf'] = results[strategy]['carbon_footprints'][year]
        summary_data.append(row)

    detailed_summary_df = pd.DataFrame(summary_data)
    detailed_summary_df.to_csv(results_dir / 'carbon_footprint_detailed_summary.csv', index=False)

    logging.info("\nProject execution completed successfully!")
    logging.info(f"Results saved to: {results_dir}")
    logging.info("\nGenerated files:")
    logging.info("  - cumulative_returns_part1.png")
    logging.info("  - carbon_footprint_evolution.png")
    logging.info("  - carbon_reduction_percentages.png")
    logging.info("  - cumulative_returns_comparison.png")
    logging.info("  - performance_summary_table.png")
    logging.info("  - performance_summary.csv")
    logging.info("  - carbon_footprint_detailed_summary.csv")
    logging.info("  - project_report.txt")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        traceback.print_exc()