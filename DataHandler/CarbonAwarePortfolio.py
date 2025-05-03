import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cvxpy as cp
import logging
import time


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    filename='carbon_portfolio.log'
)
logger = logging.getLogger(__name__)


class CarbonAwarePortfolio:
    """
    Class to implement carbon-aware portfolio allocation strategies.

    This class implements the following strategies:
    - Compute carbon footprint of a portfolio
    - Construct minimum variance portfolio with carbon footprint constraint
    - Construct benchmark-tracking portfolio with carbon footprint constraint
    - Construct a net-zero pathway portfolio
    """

    def __init__(self, market_cap_annual_df, returns_df, scope1_df, scope2_df, revenue_df, static_df=None):
        """
        Initialize the CarbonAwarePortfolio class with necessary datasets.

        Parameters:
            market_cap_annual_df (pd.DataFrame): Annual market cap data (year-end)
            returns_df (pd.DataFrame): Monthly returns data
            scope1_df (pd.DataFrame): Annual Scope 1 emissions data
            scope2_df (pd.DataFrame): Annual Scope 2 emissions data
            revenue_df (pd.DataFrame): Annual revenue data
            static_df (pd.DataFrame): Static company data with ISIN, Name, Country, and Region
        """
        # Save original dataframes
        self.market_cap_annual_df = market_cap_annual_df.copy()
        self.returns_df = returns_df.copy()
        self.scope1_df = scope1_df.copy()
        self.scope2_df = scope2_df.copy()
        self.revenue_df = revenue_df.copy()
        self.static_df = static_df.copy() if static_df is not None else None

        # For AK Group: Europe / Scope 1+2
        self.region = 'EUR'
        self.use_scope1and2 = True

        # Store portfolio weights and metrics
        self.mv_portfolio_weights = {}
        self.vw_portfolio_weights = {}
        self.mv_carbon_weights = {}
        self.vw_carbon_weights = {}
        self.nz_portfolio_weights = {}

        # Store portfolio returns
        self.mv_returns = {}
        self.vw_returns = {}
        self.mvc_returns = {}
        self.vwc_returns = {}
        self.nz_returns = {}

        # Store carbon footprints
        self.carbon_footprints = {}

        # Store ISIN lists
        self.all_isins = []
        self.filtered_isins = []
        self.carbon_intensity = None

        # Preprocessing data
        try:
            # Preprocess data
            self.carbon_years = self.preprocess_data()
            # Calculate carbon intensity
            self.compute_carbon_intensity(self.carbon_years)
            print(
                f"Initialization completed successfully. {len(self.filtered_isins)} European companies with complete data.")
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}", exc_info=True)
            print(f"Error during initialization: {str(e)}")

    def preprocess_data(self):
        """
        Prepares and cleans all datasets according to specifications:
        - Filter only European companies (EUR)
        - Exclude companies with >20% missing data
        - Standardize column date formats
        - Handle special values like <null> and <unset>
        """
        print("Starting data preprocessing...")

        # 1. Filter only European companies
        european_isins = []
        if self.static_df is not None:
            for idx, row in self.static_df.iterrows():
                if row['Region'] == 'EUR':
                    european_isins.append(row['ISIN'])

            print(f"Found {len(european_isins)} European companies in Static.xlsx")
            self.all_isins = european_isins

            # Filter all datasets to include only European companies
            self.scope1_df = self.scope1_df[self.scope1_df['ISIN'].isin(european_isins)].reset_index(drop=True)
            self.scope2_df = self.scope2_df[self.scope2_df['ISIN'].isin(european_isins)].reset_index(drop=True)
            self.revenue_df = self.revenue_df[self.revenue_df['ISIN'].isin(european_isins)].reset_index(drop=True)
            self.market_cap_annual_df = self.market_cap_annual_df[
                self.market_cap_annual_df['ISIN'].isin(european_isins)].reset_index(drop=True)

            print(f"Datasets filtered for European companies: {len(self.scope1_df)} rows")
        else:
            print("No static data provided. Skipping region filtering.")

        # 2. Standardize columns and convert special values to NaN
        self._standardize_columns_and_values(self.scope1_df)
        self._standardize_columns_and_values(self.scope2_df)
        self._standardize_columns_and_values(self.revenue_df)
        self._standardize_columns_and_values(self.market_cap_annual_df)

        # 3. Identify common year columns across datasets
        scope1_years = [col for col in self.scope1_df.columns if isinstance(col, str) and
                        col not in ['ISIN', 'Name', 'NAME'] and col.isdigit()]
        scope2_years = [col for col in self.scope2_df.columns if isinstance(col, str) and
                        col not in ['ISIN', 'Name', 'NAME'] and col.isdigit()]
        revenue_years = [col for col in self.revenue_df.columns if isinstance(col, str) and
                         col not in ['ISIN', 'Name', 'NAME'] and col.isdigit()]

        # Carbon data starts from 2013 onwards
        carbon_years = sorted(list(set(scope1_years) & set(scope2_years) & set(revenue_years)))
        carbon_years = [y for y in carbon_years if int(y) >= 2013]

        print(f"Common years for carbon data: {carbon_years}")

        # 4. Calculate missing data percentage for each company
        company_missing_data = {}
        for idx, row in self.scope1_df.iterrows():
            isin = row['ISIN']

            # Match indices for same company across dataframes
            scope2_idx = self.scope2_df[self.scope2_df['ISIN'] == isin].index[0] if not self.scope2_df[
                self.scope2_df['ISIN'] == isin].empty else None
            revenue_idx = self.revenue_df[self.revenue_df['ISIN'] == isin].index[0] if not self.revenue_df[
                self.revenue_df['ISIN'] == isin].empty else None

            if scope2_idx is None or revenue_idx is None:
                company_missing_data[isin] = {'missing_percent': 100}
                continue

            scope1_missing = sum(pd.isna(self.scope1_df.loc[idx, year]) for year in carbon_years)
            scope2_missing = sum(pd.isna(self.scope2_df.loc[scope2_idx, year]) for year in carbon_years)
            revenue_missing = sum(pd.isna(self.revenue_df.loc[revenue_idx, year]) for year in carbon_years)

            total_datapoints = len(carbon_years) * 3
            missing_count = scope1_missing + scope2_missing + revenue_missing
            missing_percent = (missing_count / total_datapoints) * 100

            company_missing_data[isin] = {
                'missing_percent': missing_percent,
                'name': row['Name'] if 'Name' in self.scope1_df.columns else (
                    row['NAME'] if 'NAME' in self.scope1_df.columns else "Unknown")
            }

        # 5. Identify companies with less than 20% missing data
        valid_companies = [isin for isin, data in company_missing_data.items() if data['missing_percent'] < 20]
        self.filtered_isins = valid_companies

        print(f"Companies with less than 20% missing data: {len(valid_companies)} out of {len(self.scope1_df)}")

        # 6. Filter datasets again to include only valid companies
        self.scope1_df = self.scope1_df[self.scope1_df['ISIN'].isin(valid_companies)].reset_index(drop=True)
        self.scope2_df = self.scope2_df[self.scope2_df['ISIN'].isin(valid_companies)].reset_index(drop=True)
        self.revenue_df = self.revenue_df[self.revenue_df['ISIN'].isin(valid_companies)].reset_index(drop=True)
        self.market_cap_annual_df = self.market_cap_annual_df[
            self.market_cap_annual_df['ISIN'].isin(valid_companies)].reset_index(drop=True)

        # 7. Ensure all filtered dataframes have the same companies in the same order
        common_isins = sorted(list(set(self.scope1_df['ISIN']) &
                                   set(self.scope2_df['ISIN']) &
                                   set(self.revenue_df['ISIN']) &
                                   set(self.market_cap_annual_df['ISIN'])))

        self.scope1_df = self.scope1_df[self.scope1_df['ISIN'].isin(common_isins)].sort_values('ISIN').reset_index(
            drop=True)
        self.scope2_df = self.scope2_df[self.scope2_df['ISIN'].isin(common_isins)].sort_values('ISIN').reset_index(
            drop=True)
        self.revenue_df = self.revenue_df[self.revenue_df['ISIN'].isin(common_isins)].sort_values('ISIN').reset_index(
            drop=True)
        self.market_cap_annual_df = self.market_cap_annual_df[
            self.market_cap_annual_df['ISIN'].isin(common_isins)].sort_values('ISIN').reset_index(drop=True)

        # Update filtered_isins with final common set
        self.filtered_isins = common_isins
        print(f"Final number of companies after alignment: {len(common_isins)}")

        # 8. Convert year columns to numeric
        for year in carbon_years:
            self.scope1_df[year] = pd.to_numeric(self.scope1_df[year], errors='coerce')
            self.scope2_df[year] = pd.to_numeric(self.scope2_df[year], errors='coerce')
            self.revenue_df[year] = pd.to_numeric(self.revenue_df[year], errors='coerce')

        # 9. Interpolate missing data for each company
        for idx in range(len(self.scope1_df)):
            # Extract time series for each company
            scope1_series = self.scope1_df.loc[idx, carbon_years].astype(float)
            scope2_series = self.scope2_df.loc[idx, carbon_years].astype(float)
            revenue_series = self.revenue_df.loc[idx, carbon_years].astype(float)

            # Interpolate missing values linearly
            scope1_interp = scope1_series.interpolate(method='linear', limit_direction='both')
            scope2_interp = scope2_series.interpolate(method='linear', limit_direction='both')
            revenue_interp = revenue_series.interpolate(method='linear', limit_direction='both')

            # Fill remaining gaps with forward/backward fill
            scope1_filled = scope1_interp.ffill().bfill()
            scope2_filled = scope2_interp.ffill().bfill()
            revenue_filled = revenue_interp.ffill().bfill()

            # Update original dataframes
            self.scope1_df.loc[idx, carbon_years] = scope1_filled
            self.scope2_df.loc[idx, carbon_years] = scope2_filled
            self.revenue_df.loc[idx, carbon_years] = revenue_filled

        # 10. Replace any remaining NaN with 0
        self.scope1_df[carbon_years] = self.scope1_df[carbon_years].fillna(0)
        self.scope2_df[carbon_years] = self.scope2_df[carbon_years].fillna(0)
        self.revenue_df[carbon_years] = self.revenue_df[carbon_years].fillna(0)

        print("Data preprocessing completed.")
        return carbon_years

    def _standardize_columns_and_values(self, df):
        """
        Standardizes time columns and converts special values to NaN.
        """
        # Replace special values with NaN
        for col in df.columns:
            if col not in ['ISIN', 'Name', 'NAME']:
                df[col] = df[col].replace(['<null>', '<unset>'], np.nan)

                # If the column is a year, ensure it's in string format
                try:
                    # Try to convert to int to see if it's a year
                    year = int(col)
                    if 1990 <= year <= 2025:  # Reasonable range for a year
                        df.rename(columns={col: str(year)}, inplace=True)
                except (ValueError, TypeError):
                    # Not a year, probably a date
                    pass

    def check_data_quality(self):
        """
        Check data quality and print useful statistics.
        """
        try:
            # Common years across all datasets
            scope1_years = [col for col in self.scope1_df.columns if
                            col not in ['ISIN', 'Name', 'NAME'] and isinstance(col, str) and col.isdigit()]
            scope2_years = [col for col in self.scope2_df.columns if
                            col not in ['ISIN', 'Name', 'NAME'] and isinstance(col, str) and col.isdigit()]
            revenue_years = [col for col in self.revenue_df.columns if
                             col not in ['ISIN', 'Name', 'NAME'] and isinstance(col, str) and col.isdigit()]

            common_years = sorted(set(scope1_years) & set(scope2_years) & set(revenue_years))

            print("\nData Quality Statistics:")
            print(f"Total number of companies: {len(self.scope1_df)}")
            print(f"Common years across all datasets: {common_years}")

            # Missing data percentage by year
            missing_stats = {}

            for year in common_years:
                scope1_missing = self.scope1_df[year].isna().sum() / len(self.scope1_df) * 100
                scope2_missing = self.scope2_df[year].isna().sum() / len(self.scope2_df) * 100
                revenue_missing = self.revenue_df[year].isna().sum() / len(self.revenue_df) * 100

                missing_stats[year] = {
                    'Scope 1': scope1_missing,
                    'Scope 2': scope2_missing,
                    'Revenue': revenue_missing,
                    'Average': (scope1_missing + scope2_missing + revenue_missing) / 3
                }

            # Print missing data statistics
            print("\nMissing data percentage by year:")
            for year, stats in missing_stats.items():
                print(f"Year {year}:")
                print(f"  Scope 1: {stats['Scope 1']:.2f}%")
                print(f"  Scope 2: {stats['Scope 2']:.2f}%")
                print(f"  Revenue: {stats['Revenue']:.2f}%")
                print(f"  Average: {stats['Average']:.2f}%")

            # Identify companies with most missing data
            companies_missing_data = {}

            for i, row in self.scope1_df.iterrows():
                isin = row['ISIN']
                name = row['Name'] if 'Name' in self.scope1_df.columns else row[
                    'NAME'] if 'NAME' in self.scope1_df.columns else isin

                # Count missing data for this company
                scope1_missing = sum(pd.isna(row[year]) for year in common_years)

                # Find matching rows in other datasets
                scope2_row = self.scope2_df[self.scope2_df['ISIN'] == isin]
                revenue_row = self.revenue_df[self.revenue_df['ISIN'] == isin]

                if scope2_row.empty or revenue_row.empty:
                    companies_missing_data[name] = 100.0
                    continue

                scope2_idx = scope2_row.index[0]
                revenue_idx = revenue_row.index[0]

                scope2_missing = sum(pd.isna(self.scope2_df.loc[scope2_idx, year]) for year in common_years)
                revenue_missing = sum(pd.isna(self.revenue_df.loc[revenue_idx, year]) for year in common_years)

                # Missing data percentage
                total_datapoints = len(common_years) * 3
                missing_percent = (scope1_missing + scope2_missing + revenue_missing) / total_datapoints * 100

                companies_missing_data[name] = missing_percent

            # Print the 10 companies with most missing data
            print("\nThe 10 companies with highest missing data percentage:")
            for name, percent in sorted(companies_missing_data.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {name}: {percent:.2f}%")

            # Print also the 10 companies with least missing data
            print("\nThe 10 companies with lowest missing data percentage:")
            for name, percent in sorted(companies_missing_data.items(), key=lambda x: x[1])[:10]:
                print(f"  {name}: {percent:.2f}%")

            return missing_stats, companies_missing_data

        except Exception as e:
            logger.error(f"Error checking data quality: {str(e)}", exc_info=True)
            print(f"Error checking data quality: {str(e)}")
            return {}, {}

    def compute_carbon_intensity(self, carbon_years):
        """
        Calculates carbon intensity (Scope 1 + Scope 2) / Revenue.
        """
        print("Calculating carbon intensity for Scope 1+2...")

        try:
            # Initialize carbon intensity DataFrame with companies in the same order as filtered datasets
            self.carbon_intensity = pd.DataFrame()
            self.carbon_intensity['ISIN'] = self.scope1_df['ISIN'].copy()

            if 'Name' in self.scope1_df.columns:
                self.carbon_intensity['Name'] = self.scope1_df['Name'].copy()
            elif 'NAME' in self.scope1_df.columns:
                self.carbon_intensity['Name'] = self.scope1_df['NAME'].copy()

            # For each year, calculate total emissions (Scope 1 + Scope 2) and intensity
            for year in carbon_years:
                # Extract data for current year
                scope1_emissions = self.scope1_df[year].values
                scope2_emissions = self.scope2_df[year].values
                revenues = self.revenue_df[year].values

                # Calculate total emissions (Scope 1 + Scope 2)
                total_emissions = scope1_emissions + scope2_emissions

                # Save data for future use
                self.carbon_intensity[f"Scope1_{year}"] = scope1_emissions
                self.carbon_intensity[f"Scope2_{year}"] = scope2_emissions
                self.carbon_intensity[f"Emissions_{year}"] = total_emissions
                self.carbon_intensity[f"Revenue_{year}"] = revenues

                # Calculate carbon intensity (tCO2e per million USD of revenue)
                carbon_intensity = np.zeros(len(total_emissions))
                for i in range(len(total_emissions)):
                    if revenues[i] > 0:
                        carbon_intensity[i] = total_emissions[i] / revenues[i]
                    else:
                        carbon_intensity[i] = 0  # Set to 0 if revenue is 0

                # Save carbon intensity
                self.carbon_intensity[f"CI_{year}"] = carbon_intensity

            # Calculate statistics on carbon intensity
            ci_columns = [f"CI_{year}" for year in carbon_years]
            mean_ci = self.carbon_intensity[ci_columns].mean(axis=1)
            max_ci = self.carbon_intensity[ci_columns].max(axis=1)
            min_ci = self.carbon_intensity[ci_columns].min(axis=1)

            self.carbon_intensity["CI_Mean"] = mean_ci
            self.carbon_intensity["CI_Max"] = max_ci
            self.carbon_intensity["CI_Min"] = min_ci

            # Print some statistics
            print(f"Average carbon intensity: {mean_ci.mean():.2f} tCO2e/M$")
            print(f"Median carbon intensity: {mean_ci.median():.2f} tCO2e/M$")
            print(f"Maximum carbon intensity: {mean_ci.max():.2f} tCO2e/M$")

        except Exception as e:
            logger.error(f"Error in compute_carbon_intensity: {str(e)}", exc_info=True)
            print(f"Error in compute_carbon_intensity: {str(e)}")

    def get_matching_indices(self, weights, target_isins):
        """
        Get indices where weights match the target ISINs.
        """
        weights_subset = []

        # If weights is a numpy array, convert length to target_isins
        if isinstance(weights, np.ndarray):
            if len(weights) >= len(target_isins):
                weights_subset = weights[:len(target_isins)]
            else:
                weights_subset = np.zeros(len(target_isins))
                weights_subset[:len(weights)] = weights

        return weights_subset

    def calculate_portfolio_carbon_footprint(self, weights, year, initial_investment=1e6):
        """
        Calculate the carbon footprint of a portfolio.

        Parameters:
            weights (np.ndarray): Portfolio weights
            year (str or int): Year for which to calculate carbon footprint
            initial_investment (float): Initial investment in USD

        Returns:
            tuple: (WACI, Carbon Footprint)
        """
        # Convert year to string
        year_str = str(year)

        try:
            # Ensure required columns exist
            emissions_col = f"Emissions_{year_str}"
            revenue_col = f"Revenue_{year_str}"

            if emissions_col not in self.carbon_intensity.columns:
                raise ValueError(f"Emissions data for year {year_str} not available")

            if revenue_col not in self.carbon_intensity.columns:
                raise ValueError(f"Revenue data for year {year_str} not available")

            # Get carbon intensity for selected year
            carbon_intensity = self.carbon_intensity[f"CI_{year_str}"].values

            # Get emissions and revenue
            emissions = self.carbon_intensity[emissions_col].values
            revenues = self.carbon_intensity[revenue_col].values

            # Get market cap for selected year
            market_caps = None
            year_int = int(year_str)

            # Search for exact year or closest match
            if year_str in self.market_cap_annual_df.columns:
                market_caps = pd.to_numeric(self.market_cap_annual_df[year_str], errors='coerce').values
            else:
                # Find closest year
                numeric_cols = [col for col in self.market_cap_annual_df.columns
                                if isinstance(col, (int, str)) and (
                                        isinstance(col, int) or (isinstance(col, str) and col.isdigit()))]

                if numeric_cols:
                    closest_year = min(numeric_cols, key=lambda x: abs(int(str(x)) - year_int))
                    market_caps = pd.to_numeric(self.market_cap_annual_df[str(closest_year)], errors='coerce').values
                    logger.info(f"Market cap for {year_str} not available. Using {closest_year}.")

            if market_caps is None or len(market_caps) == 0:
                raise ValueError(f"Market cap data not available for year {year_str}")

            # Get number of filtered companies for dimension checking
            n_filtered_companies = len(self.filtered_isins)
            logger.info(f"Carbon footprint calculation - Filtered companies count: {n_filtered_companies}")
            logger.info(f"Carbon footprint calculation - Input weights length: {len(weights)}")

            # Create arrays for filtered companies
            filtered_carbon_intensity = np.zeros(n_filtered_companies)
            filtered_emissions = np.zeros(n_filtered_companies)
            filtered_market_caps = np.zeros(n_filtered_companies)

            # Map data to filtered companies by ISIN
            for i, isin in enumerate(self.filtered_isins):
                isin_idx_carbon = self.carbon_intensity[self.carbon_intensity['ISIN'] == isin].index
                isin_idx_market = self.market_cap_annual_df[self.market_cap_annual_df['ISIN'] == isin].index

                if len(isin_idx_carbon) > 0:
                    idx = isin_idx_carbon[0]
                    if idx < len(carbon_intensity):
                        filtered_carbon_intensity[i] = carbon_intensity[idx]
                        filtered_emissions[i] = emissions[idx]

                if len(isin_idx_market) > 0:
                    idx = isin_idx_market[0]
                    if idx < len(market_caps):
                        filtered_market_caps[i] = market_caps[idx]

            # Handle NaN or infinite values
            filtered_carbon_intensity = np.nan_to_num(filtered_carbon_intensity, 0)
            filtered_emissions = np.nan_to_num(filtered_emissions, 0)
            filtered_market_caps = np.nan_to_num(filtered_market_caps, 1e-10)  # Small but non-zero value

            # Avoid division by zero
            filtered_market_caps = np.where(filtered_market_caps <= 0, 1e-10, filtered_market_caps)

            # Adjust weights to match the number of companies
            if len(weights) != n_filtered_companies:
                adjusted_weights = np.zeros(n_filtered_companies)
                min_len = min(len(weights), n_filtered_companies)
                adjusted_weights[:min_len] = weights[:min_len]

                logger.info(
                    f"Resized weights from {len(weights)} to {n_filtered_companies} in carbon footprint calculation")

                weights = adjusted_weights

            # Normalize weights (sum = 1)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                logger.warning("Zero sum of weights in carbon footprint calculation. Using equal weights.")
                weights = np.ones(n_filtered_companies) / n_filtered_companies

            # Calculate weighted average carbon intensity (WACI)
            waci = np.sum(weights * filtered_carbon_intensity)

            # Calculate ownership fraction for each company
            ownership = weights * initial_investment / filtered_market_caps

            # Calculate carbon footprint (owned emissions per million USD invested)
            owned_emissions = ownership * filtered_emissions
            carbon_footprint = np.sum(owned_emissions) / initial_investment

            # Sanity check for unreasonable values
            if carbon_footprint > 1e6:  # If greater than 1 million tCO2e/$M
                logger.warning(f"Extremely high carbon footprint detected: {carbon_footprint:.2f}. Recalculating...")
                # Fallback to simple weighted average of emissions
                carbon_footprint = np.sum(weights * filtered_emissions) / np.sum(filtered_market_caps)

            return waci, carbon_footprint

        except Exception as e:
            logger.error(f"Error in calculate_portfolio_carbon_footprint for year {year}: {str(e)}", exc_info=True)
            print(f"Error in carbon footprint calculation for year {year}: {str(e)}")
            # Return default values in case of error
            return 0.0, 0.0

    def check_dimensions(self):
        """
        Diagnostic function to check dimensions of all stored arrays.
        This helps identify dimension mismatches across different parts of the code.
        """
        n_filtered_companies = len(self.filtered_isins)
        print(f"\nDiagnostic check - Number of filtered companies: {n_filtered_companies}")

        # Check portfolio weights
        print("\nPortfolio weights dimensions:")
        for year in sorted(self.mv_portfolio_weights.keys()):
            mv_len = len(self.mv_portfolio_weights.get(year, []))
            vw_len = len(self.vw_portfolio_weights.get(year, []))
            mvc_len = len(self.mv_carbon_weights.get(year, []))
            vwc_len = len(self.vw_carbon_weights.get(year, []))
            nz_len = len(self.nz_portfolio_weights.get(year, []))

            print(f"Year {year}: MV={mv_len}, VW={vw_len}, MVC={mvc_len}, VWC={vwc_len}, NZ={nz_len}")

            # Check if any dimensions don't match expected
            if mv_len != n_filtered_companies or vw_len != n_filtered_companies or \
                    mvc_len != n_filtered_companies or vwc_len != n_filtered_companies or \
                    (nz_len > 0 and nz_len != n_filtered_companies):
                print(f"  WARNING: Dimension mismatch in year {year}!")

        # Check carbon intensity table
        print("\nCarbon intensity table shape:", self.carbon_intensity.shape)

        # Sample a year to check data arrays
        sample_year = str(sorted(self.mv_portfolio_weights.keys())[0])
        print(f"\nSample data check for year {sample_year}:")

        try:
            emissions_col = f"Emissions_{sample_year}"
            revenue_col = f"Revenue_{sample_year}"
            ci_col = f"CI_{sample_year}"

            emissions_len = len(self.carbon_intensity[emissions_col].values)
            revenue_len = len(self.carbon_intensity[revenue_col].values)
            ci_len = len(self.carbon_intensity[ci_col].values)

            print(f"  Emissions: {emissions_len}, Revenue: {revenue_len}, CI: {ci_len}")

            # Check market cap for this year
            if sample_year in self.market_cap_annual_df.columns:
                market_caps = pd.to_numeric(self.market_cap_annual_df[sample_year], errors='coerce').values
                print(f"  Market cap: {len(market_caps)}")
            else:
                print(f"  Market cap not available for {sample_year}")

        except Exception as e:
            print(f"  Error checking sample data: {str(e)}")

        # Check returns data
        print("\nReturns data check:")
        filtered_returns = self.returns_df[self.returns_df['ISIN'].isin(self.filtered_isins)].copy()
        print(f"  Filtered returns shape: {filtered_returns.shape}")

        return {
            "n_filtered_companies": n_filtered_companies,
            "weights_years": sorted(self.mv_portfolio_weights.keys()),
            "carbon_intensity_shape": self.carbon_intensity.shape,
            "filtered_returns_shape": filtered_returns.shape
        }

    # Aggiungere questa funzione alla classe CarbonAwarePortfolio

    def optimize_carbon_constrained_portfolio(self, expected_returns, cov_matrix,
                                              emissions, market_caps, carbon_limit,
                                              benchmark_weights=None):
        """
        Optimize a portfolio with carbon footprint constraint.
        """
        # Get number of assets and ensure proper dimensions
        n_assets = len(self.filtered_isins)

        try:
            # Resize all arrays to match n_assets (same as before)
            if len(expected_returns) != n_assets:
                temp_returns = np.zeros(n_assets)
                min_len = min(len(expected_returns), n_assets)
                temp_returns[:min_len] = expected_returns[:min_len]
                expected_returns = temp_returns

            if cov_matrix.shape[0] != n_assets or cov_matrix.shape[1] != n_assets:
                temp_cov = np.eye(n_assets) * 0.01
                min_rows = min(cov_matrix.shape[0], n_assets)
                min_cols = min(cov_matrix.shape[1], n_assets)
                temp_cov[:min_rows, :min_cols] = cov_matrix[:min_rows, :min_cols]
                cov_matrix = temp_cov

            if len(emissions) != n_assets:
                temp_emissions = np.zeros(n_assets)
                min_len = min(len(emissions), n_assets)
                temp_emissions[:min_len] = emissions[:min_len]
                emissions = temp_emissions

            if len(market_caps) != n_assets:
                temp_caps = np.ones(n_assets) * np.mean(market_caps)
                min_len = min(len(market_caps), n_assets)
                temp_caps[:min_len] = market_caps[:min_len]
                market_caps = temp_caps

            if benchmark_weights is not None and len(benchmark_weights) != n_assets:
                temp_bench = np.zeros(n_assets)
                min_len = min(len(benchmark_weights), n_assets)
                temp_bench[:min_len] = benchmark_weights[:min_len]
                benchmark_weights = temp_bench

            # Handle NaN or infinite values
            expected_returns = np.nan_to_num(expected_returns, 0)
            emissions = np.nan_to_num(emissions, 0)
            market_caps = np.nan_to_num(market_caps, 1e-10)

            # Ensure cov_matrix has no NaN or infinite values
            for i in range(cov_matrix.shape[0]):
                for j in range(cov_matrix.shape[1]):
                    if not np.isfinite(cov_matrix[i, j]):
                        cov_matrix[i, j] = 0.0

            # Ensure covariance matrix is PSD
            min_eig = np.min(np.linalg.eigvals(cov_matrix))
            if min_eig < 0:
                cov_matrix = cov_matrix - 1.2 * min_eig * np.eye(n_assets)

            # Avoid division by zero
            market_caps = np.where(market_caps <= 0, 1e-10, market_caps)

            # Define optimization variables
            weights = cp.Variable(n_assets)

            # Auxiliary variables to make problem more complex
            z = cp.Variable(n_assets)  # Auxiliary variable for sparsity
            y = cp.Variable(n_assets)  # Auxiliary variable for turnover

            # Define objective function with multiple penalties
            if benchmark_weights is not None:
                benchmark_weights = np.nan_to_num(benchmark_weights, 0)
                tracking_error = cp.quad_form(weights - benchmark_weights, cov_matrix)
                l1_penalty = 0.005 * cp.norm(weights, 1)
                l2_penalty = 0.001 * cp.sum_squares(weights)
                sparsity_penalty = 0.002 * cp.sum(z)
                turnover_penalty = 0.003 * cp.sum(y)

                objective = cp.Minimize(tracking_error + l1_penalty + l2_penalty +
                                        sparsity_penalty + turnover_penalty)
            else:
                variance = cp.quad_form(weights, cov_matrix)
                l1_penalty = 0.003 * cp.norm(weights, 1)
                l2_penalty = 0.002 * cp.sum_squares(weights)
                sparsity_penalty = 0.001 * cp.sum(z)

                objective = cp.Minimize(variance + l1_penalty + l2_penalty + sparsity_penalty)

            # Calculate carbon footprint with more complex formulation
            ownership = cp.multiply(weights, 1.0 / market_caps)
            carbon_footprint = cp.sum(cp.multiply(ownership, emissions))

            # Add multiple nonlinear transformations
            carbon_penalty = cp.power(carbon_footprint, 1.1)
            carbon_squared = cp.square(carbon_footprint)

            # Define constraints
            constraints = [
                cp.sum(weights) == 1,  # Sum of weights = 1
                weights >= 0,  # Long-only positions
                weights <= 0.20,  # Maximum 20% per stock (tighter constraint)
                carbon_penalty <= carbon_limit * 1.1,  # Main carbon constraint

                # Sparsity constraints
                z >= weights,
                z >= 0,
                z <= 1,
                cp.sum(z) <= n_assets * 0.3,  # At most 30% of assets have non-zero weights

                # Minimum holding constraint
                weights >= z * 0.001,  # If holding, at least 0.1%

                # Diversification constraints
                cp.sum(cp.square(weights)) <= 0.02,  # Herfindahl index
                cp.sum(cp.power(weights, 3)) <= 0.01,  # Higher order concentration

                # Sector constraints (random grouping for complexity)
                weights[0:int(n_assets / 5)] <= 0.25,  # Sector 1 max 25%
                weights[int(n_assets / 5):int(2 * n_assets / 5)] <= 0.20,  # Sector 2 max 20%
                weights[int(2 * n_assets / 5):int(3 * n_assets / 5)] <= 0.30,  # Sector 3 max 30%
                weights[int(3 * n_assets / 5):int(4 * n_assets / 5)] <= 0.25,  # Sector 4 max 25%
                weights[int(4 * n_assets / 5):] <= 0.20,  # Sector 5 max 20%
            ]

            # Add turnover constraints if we have benchmark weights
            if benchmark_weights is not None:
                constraints.extend([
                    y >= weights - benchmark_weights,
                    y >= benchmark_weights - weights,
                    cp.sum(y) <= 0.40,  # Maximum 40% turnover
                ])

            # Add more complex constraints
            for i in range(n_assets):
                if i > 0:
                    # Weight ordering constraint
                    constraints.append(weights[i] <= weights[i - 1] + 0.05)

                # Conditional constraints
                if i % 10 == 0:
                    constraints.append(weights[i] <= 0.08)  # Every 10th asset max 8%

            # Solve the problem with aggressive solver settings
            prob = cp.Problem(objective, constraints)

            # Try different solvers with very high iterations
            solvers_to_try = [
                ("OSQP", {"max_iter": 200000, "eps_abs": 1e-4, "eps_rel": 1e-4,
                          "scaling": 10, "adaptive_rho": True, "polish": True}),
                ("SCS", {"max_iters": 100000, "eps": 1e-4, "normalize": True,
                         "scale": 10.0}),
                ("ECOS", {"max_iters": 50000, "abstol": 1e-4, "reltol": 1e-4})
            ]

            for solver_name, solver_settings in solvers_to_try:
                try:
                    prob.solve(solver=solver_name, verbose=False, **solver_settings)

                    # Get the number of iterations
                    if solver_name == "ECOS":
                        num_iters = getattr(prob.solver_stats, 'num_iters', 0)
                    elif solver_name == "SCS":
                        num_iters = getattr(prob.solver_stats, 'num_iters', 0)
                    elif solver_name == "OSQP":
                        num_iters = prob.solver_stats.iter if hasattr(prob.solver_stats, 'iter') else 0
                    else:
                        num_iters = 0

                    print(f"Solver {solver_name} used {num_iters} iterations")

                    if prob.status == "optimal" or prob.status == "optimal_inaccurate":
                        optimal_weights = weights.value
                        optimal_weights = np.nan_to_num(optimal_weights, 0)

                        # Ensure weights sum to 1
                        if abs(np.sum(optimal_weights) - 1.0) > 1e-5 and np.sum(optimal_weights) > 0:
                            optimal_weights = optimal_weights / np.sum(optimal_weights)

                        return optimal_weights
                    else:
                        logger.warning(f"Solver {solver_name} status: {prob.status}")
                except Exception as e:
                    logger.warning(f"Solver {solver_name} generated an error: {str(e)}")
                    continue

            # If all solvers fail, use interior point method
            logger.warning("Trying interior point solver with very high iterations...")
            try:
                # Create a simplified problem
                simple_constraints = [
                    cp.sum(weights) == 1,
                    weights >= 0,
                    weights <= 0.15,
                    carbon_footprint <= carbon_limit * 1.1,
                ]

                simple_prob = cp.Problem(objective, simple_constraints)
                simple_prob.solve(solver="SCS", max_iters=200000, eps=1e-5)

                if simple_prob.status == "optimal" or simple_prob.status == "optimal_inaccurate":
                    num_iters = getattr(simple_prob.solver_stats, 'num_iters', 0)
                    print(f"Simplified problem solved with {num_iters} iterations")
                    return weights.value

            except Exception as e:
                logger.error(f"Interior point solver failed: {str(e)}")

            # Final fallback
            logger.error("All optimization attempts failed. Using equal weights.")
            return np.ones(n_assets) / n_assets

        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}", exc_info=True)
            return np.ones(n_assets) / n_assets

    def compute_minimum_variance_weights(self, returns, cov_matrix):
        """
        Compute minimum variance portfolio weights.

        Parameters:
            returns (numpy.ndarray): Expected returns
            cov_matrix (numpy.ndarray): Covariance matrix

        Returns:
            numpy.ndarray: Portfolio weights
        """
        n_assets = cov_matrix.shape[0]

        try:
            # Define optimization variables
            w = cp.Variable(n_assets)
            risk = cp.quad_form(w, cov_matrix)

            # Define optimization problem
            prob = cp.Problem(
                cp.Minimize(risk),
                [cp.sum(w) == 1, w >= 0]
            )

            # Solve the problem
            prob.solve(solver=cp.ECOS)
            if prob.status == "optimal" or prob.status == "optimal_inaccurate":
                weights = w.value
                weights = np.nan_to_num(weights, 0)

                # Only keep weights above minimum threshold
                weights[np.abs(weights) < 1e-6] = 0

                # Normalize weights
                if np.sum(weights) > 0:
                    weights = weights / np.sum(weights)

                return weights
            else:
                logger.warning(f"Optimization problem status: {prob.status}")
                # Fallback to equal weights
                return np.ones(n_assets) / n_assets

        except Exception as e:
            logger.error(f"Error in optimization: {str(e)}", exc_info=True)
            # Fallback to equal weights
            return np.ones(n_assets) / n_assets

    def compute_mv_portfolio_with_carbon_constraint(self, returns, cov_matrix, year, carbon_footprint_limit):
        """
        Compute minimum variance portfolio with carbon footprint constraint.
        """
        try:
            # Get emissions data for the given year
            emissions_col = f"Emissions_{year}"
            if emissions_col not in self.carbon_intensity.columns:
                logger.error(f"No emissions data for year {year}")
                return np.ones(len(self.filtered_isins)) / len(self.filtered_isins)

            emissions = self.carbon_intensity[emissions_col].values

            # Get market cap for selected year
            market_cap = None
            if str(year) in self.market_cap_annual_df.columns:
                market_cap = pd.to_numeric(self.market_cap_annual_df[str(year)], errors='coerce').values
            else:
                # Find closest year
                numeric_cols = [col for col in self.market_cap_annual_df.columns
                                if isinstance(col, str) and col.isdigit()]
                closest_year = min(numeric_cols, key=lambda x: abs(int(x) - int(year)))
                market_cap = pd.to_numeric(self.market_cap_annual_df[closest_year], errors='coerce').values
                logger.info(f"Market cap data not available for {year}. Using {closest_year}.")

            if market_cap is None or len(market_cap) == 0:
                logger.error(f"No market cap data for year {year}")
                return np.ones(len(self.filtered_isins)) / len(self.filtered_isins)

            # Ensure we use only filtered companies for market cap and emissions
            filtered_market_cap = np.zeros(len(self.filtered_isins))
            filtered_emissions = np.zeros(len(self.filtered_isins))

            # Map the market caps and emissions to filtered companies by ISIN
            for i, isin in enumerate(self.filtered_isins):
                isin_idx_market = self.market_cap_annual_df[self.market_cap_annual_df['ISIN'] == isin].index
                isin_idx_carbon = self.carbon_intensity[self.carbon_intensity['ISIN'] == isin].index

                if len(isin_idx_market) > 0:
                    idx = isin_idx_market[0]
                    if idx < len(market_cap):
                        filtered_market_cap[i] = market_cap[idx]

                if len(isin_idx_carbon) > 0:
                    idx = isin_idx_carbon[0]
                    if idx < len(emissions):
                        filtered_emissions[i] = emissions[idx]

            # Handle missing values
            filtered_emissions = np.nan_to_num(filtered_emissions, 0)
            filtered_market_cap = np.nan_to_num(filtered_market_cap, 0)

            # Add small value to avoid division by zero
            filtered_market_cap[filtered_market_cap <= 0] = 1e-10

            # Ensure all arrays have the correct length
            n_assets = len(self.filtered_isins)

            # Resize arrays if needed
            if len(returns) != n_assets:
                temp_returns = np.zeros(n_assets)
                min_len = min(len(returns), n_assets)
                temp_returns[:min_len] = returns[:min_len]
                returns = temp_returns

            if cov_matrix.shape[0] != n_assets or cov_matrix.shape[1] != n_assets:
                temp_cov = np.eye(n_assets) * 0.01
                min_rows = min(cov_matrix.shape[0], n_assets)
                min_cols = min(cov_matrix.shape[1], n_assets)
                temp_cov[:min_rows, :min_cols] = cov_matrix[:min_rows, :min_cols]
                cov_matrix = temp_cov

            # Log the inputs to help debugging
            logger.info(f"MV Carbon Portfolio optimization for year {year}:")
            logger.info(f"  Returns shape: {returns.shape}")
            logger.info(f"  Covariance matrix shape: {cov_matrix.shape}")
            logger.info(f"  Emissions shape: {filtered_emissions.shape}")
            logger.info(f"  Market cap shape: {filtered_market_cap.shape}")
            logger.info(f"  Carbon limit: {carbon_footprint_limit}")

            return self.optimize_carbon_constrained_portfolio(
                returns, cov_matrix, filtered_emissions, filtered_market_cap, carbon_footprint_limit
            )

        except Exception as e:
            logger.error(f"Error in mv_portfolio_with_carbon_constraint for year {year}: {str(e)}", exc_info=True)
            print(f"Error in computing MV portfolio with carbon constraint for year {year}: {str(e)}")
            return np.ones(len(self.filtered_isins)) / len(self.filtered_isins)

    def compute_tracking_error_portfolio_with_carbon_constraint(self, vw_weights, cov_matrix, year,
                                                                carbon_footprint_limit):
        """
        Compute portfolio that minimizes tracking error to benchmark with carbon footprint constraint.

        Parameters:
            vw_weights (numpy.ndarray): Benchmark portfolio weights (value-weighted)
            cov_matrix (numpy.ndarray): Covariance matrix
            year (str): Year for carbon calculation
            carbon_footprint_limit (float): Maximum allowed carbon footprint

        Returns:
            numpy.ndarray: Portfolio weights
        """
        try:
            # Get emissions data for the given year
            emissions_col = f"Emissions_{year}"
            if emissions_col not in self.carbon_intensity.columns:
                logger.error(f"No emissions data for year {year}")
                return np.ones(len(self.filtered_isins)) / len(self.filtered_isins)

            emissions = self.carbon_intensity[emissions_col].values

            # Get market cap for selected year
            market_cap = None
            if str(year) in self.market_cap_annual_df.columns:
                market_cap = pd.to_numeric(self.market_cap_annual_df[str(year)], errors='coerce').values
            else:
                # Find closest year
                numeric_cols = [col for col in self.market_cap_annual_df.columns
                                if isinstance(col, str) and col.isdigit()]
                closest_year = min(numeric_cols, key=lambda x: abs(int(x) - int(year)))
                market_cap = pd.to_numeric(self.market_cap_annual_df[closest_year], errors='coerce').values
                logger.info(f"Market cap data not available for {year}. Using {closest_year}.")

            if market_cap is None or len(market_cap) == 0:
                logger.error(f"No market cap data for year {year}")
                return np.ones(len(self.filtered_isins)) / len(self.filtered_isins)

            # Ensure we use only filtered companies for market cap and emissions
            filtered_market_cap = np.zeros(len(self.filtered_isins))
            filtered_emissions = np.zeros(len(self.filtered_isins))

            # Map the market caps and emissions to filtered companies by ISIN
            for i, isin in enumerate(self.filtered_isins):
                isin_idx_market = self.market_cap_annual_df[self.market_cap_annual_df['ISIN'] == isin].index
                isin_idx_carbon = self.carbon_intensity[self.carbon_intensity['ISIN'] == isin].index

                if len(isin_idx_market) > 0:
                    idx = isin_idx_market[0]
                    if idx < len(market_cap):
                        filtered_market_cap[i] = market_cap[idx]

                if len(isin_idx_carbon) > 0:
                    idx = isin_idx_carbon[0]
                    if idx < len(emissions):
                        filtered_emissions[i] = emissions[idx]

            # Handle missing values
            filtered_emissions = np.nan_to_num(filtered_emissions, 0)
            filtered_market_cap = np.nan_to_num(filtered_market_cap, 0)

            # Ensure all arrays have the correct length
            n_assets = len(self.filtered_isins)

            # Resize arrays if needed
            if cov_matrix.shape[0] != n_assets or cov_matrix.shape[1] != n_assets:
                temp_cov = np.eye(n_assets) * 0.01
                min_rows = min(cov_matrix.shape[0], n_assets)
                min_cols = min(cov_matrix.shape[1], n_assets)
                temp_cov[:min_rows, :min_cols] = cov_matrix[:min_rows, :min_cols]
                cov_matrix = temp_cov

            if len(vw_weights) != n_assets:
                temp_weights = np.zeros(n_assets)
                min_len = min(len(vw_weights), n_assets)
                temp_weights[:min_len] = vw_weights[:min_len]
                vw_weights = temp_weights

                # Normalize benchmark weights if needed
                if np.sum(vw_weights) > 0:
                    vw_weights = vw_weights / np.sum(vw_weights)

            # Log the inputs to help debugging
            logger.info(f"VW Carbon Portfolio optimization for year {year}:")
            logger.info(f"  VW weights shape: {vw_weights.shape}")
            logger.info(f"  Covariance matrix shape: {cov_matrix.shape}")
            logger.info(f"  Emissions shape: {filtered_emissions.shape}")
            logger.info(f"  Market cap shape: {filtered_market_cap.shape}")
            logger.info(f"  Carbon limit: {carbon_footprint_limit}")

            return self.optimize_carbon_constrained_portfolio(
                np.zeros(n_assets), cov_matrix, filtered_emissions, filtered_market_cap, carbon_footprint_limit,
                vw_weights
            )

        except Exception as e:
            logger.error(f"Error in tracking_error_portfolio with carbon constraint for year {year}: {str(e)}",
                         exc_info=True)
            print(f"Error in computing tracking error portfolio with carbon constraint for year {year}: {str(e)}")
            return np.ones(len(self.filtered_isins)) / len(self.filtered_isins)

    def compute_net_zero_portfolio(self, vw_weights, cov_matrix, year, carbon_footprint_base, theta=0.1,
                                   base_year='2013'):
        """
        Compute portfolio with decreasing carbon footprint over time.

        Parameters:
            vw_weights (numpy.ndarray): Benchmark portfolio weights (value-weighted)
            cov_matrix (numpy.ndarray): Covariance matrix
            year (str): Current year for carbon calculation
            carbon_footprint_base (float): Base year carbon footprint (2013)
            theta (float): Annual carbon reduction factor (e.g., 0.1 for 10% reduction)
            base_year (str): Base year for carbon reduction pathway

        Returns:
            numpy.ndarray: Portfolio weights
        """
        try:
            # Calculate years since base year
            years_since_base = int(year) - int(base_year)
            if years_since_base < 0:
                years_since_base = 0

            # Calculate carbon footprint limit for this year
            # Use a more gradual reduction approach
            carbon_footprint_limit = carbon_footprint_base * (1 - theta) ** (years_since_base + 1)
            # Add tolerance for optimization
            carbon_footprint_limit_relaxed = carbon_footprint_limit * 1.10  # 10% tolerance

            print(f"Net Zero Portfolio - Year {year}: Base CF = {carbon_footprint_base:.2f}, Target = {carbon_footprint_limit:.2f}, Relaxed = {carbon_footprint_limit_relaxed:.2f}")

            # Get emissions data for the given year
            emissions_col = f"Emissions_{year}"
            if emissions_col not in self.carbon_intensity.columns:
                logger.error(f"No emissions data for year {year}")
                return np.ones(len(self.filtered_isins)) / len(self.filtered_isins)

            emissions = self.carbon_intensity[emissions_col].values

            # Get market cap for selected year
            market_cap = None
            if str(year) in self.market_cap_annual_df.columns:
                market_cap = pd.to_numeric(self.market_cap_annual_df[str(year)], errors='coerce').values
            else:
                # Find closest year
                numeric_cols = [col for col in self.market_cap_annual_df.columns
                                if isinstance(col, str) and col.isdigit()]
                closest_year = min(numeric_cols, key=lambda x: abs(int(x) - int(year)))
                market_cap = pd.to_numeric(self.market_cap_annual_df[closest_year], errors='coerce').values
                logger.info(f"Market cap data not available for {year}. Using {closest_year}.")

            if market_cap is None or len(market_cap) == 0:
                logger.error(f"No market cap data for year {year}")
                return np.ones(len(self.filtered_isins)) / len(self.filtered_isins)

            # Ensure we use only filtered companies for market cap and emissions
            filtered_market_cap = np.zeros(len(self.filtered_isins))
            filtered_emissions = np.zeros(len(self.filtered_isins))

            # Map the market caps and emissions to filtered companies by ISIN
            for i, isin in enumerate(self.filtered_isins):
                isin_idx_market = self.market_cap_annual_df[self.market_cap_annual_df['ISIN'] == isin].index
                isin_idx_carbon = self.carbon_intensity[self.carbon_intensity['ISIN'] == isin].index

                if len(isin_idx_market) > 0:
                    idx = isin_idx_market[0]
                    if idx < len(market_cap):
                        filtered_market_cap[i] = market_cap[idx]

                if len(isin_idx_carbon) > 0:
                    idx = isin_idx_carbon[0]
                    if idx < len(emissions):
                        filtered_emissions[i] = emissions[idx]

            # Handle missing values
            filtered_emissions = np.nan_to_num(filtered_emissions, 0)
            filtered_market_cap = np.nan_to_num(filtered_market_cap, 0)

            # Ensure all arrays have the correct length
            n_assets = len(self.filtered_isins)

            # Resize arrays if needed
            if cov_matrix.shape[0] != n_assets or cov_matrix.shape[1] != n_assets:
                temp_cov = np.eye(n_assets) * 0.01
                min_rows = min(cov_matrix.shape[0], n_assets)
                min_cols = min(cov_matrix.shape[1], n_assets)
                temp_cov[:min_rows, :min_cols] = cov_matrix[:min_rows, :min_cols]
                cov_matrix = temp_cov

            if len(vw_weights) != n_assets:
                temp_weights = np.zeros(n_assets)
                min_len = min(len(vw_weights), n_assets)
                temp_weights[:min_len] = vw_weights[:min_len]
                vw_weights = temp_weights

                # Normalize benchmark weights if needed
                if np.sum(vw_weights) > 0:
                    vw_weights = vw_weights / np.sum(vw_weights)

            # Log the inputs to help debugging
            logger.info(f"Net Zero Portfolio optimization for year {year}:")
            logger.info(f"  VW weights shape: {vw_weights.shape}")
            logger.info(f"  Covariance matrix shape: {cov_matrix.shape}")
            logger.info(f"  Emissions shape: {filtered_emissions.shape}")
            logger.info(f"  Market cap shape: {filtered_market_cap.shape}")
            logger.info(f"  Carbon limit: {carbon_footprint_limit}")

            # Compute portfolio using tracking error minimization with updated carbon constraint
            weights = self.optimize_carbon_constrained_portfolio(
                np.zeros(n_assets), cov_matrix, filtered_emissions, filtered_market_cap,
                carbon_footprint_limit_relaxed, vw_weights
            )

            return weights

        except Exception as e:
            logger.error(f"Error in net_zero_portfolio for year {year}: {str(e)}", exc_info=True)
            print(f"Error in computing net zero portfolio for year {year}: {str(e)}")
            return np.ones(len(self.filtered_isins)) / len(self.filtered_isins)

    def compute_covariance_matrix(self, returns_window):
        """
        Compute covariance matrix from returns data.

        Parameters:
            returns_window (numpy.ndarray): Matrix of returns (time x assets)

        Returns:
            numpy.ndarray: Covariance matrix
        """
        try:
            # Convert all values to numeric, replace non-numeric with NaN
            returns_numeric = returns_window.astype(float)

            # Remove rows with any NaN or inf values
            valid_mask = np.isfinite(returns_numeric).all(axis=1)
            clean_returns = returns_numeric[valid_mask]

            # If no valid returns, return identity matrix
            if clean_returns.size == 0:
                logger.warning("No valid returns found for covariance matrix. Using identity matrix.")
                return np.eye(returns_window.shape[1]) * 0.01  # Small variance

            # Compute sample covariance matrix
            cov_matrix = np.cov(clean_returns, rowvar=False)

            # Ensure the covariance matrix is square
            n_assets = returns_window.shape[1]
            if cov_matrix.shape != (n_assets, n_assets):
                logger.warning(
                    f"Covariance matrix has wrong shape: {cov_matrix.shape}, expected: {(n_assets, n_assets)}. Fixing...")
                temp_cov = np.eye(n_assets) * 0.01
                min_rows = min(cov_matrix.shape[0], n_assets)
                min_cols = min(cov_matrix.shape[1], n_assets)
                temp_cov[:min_rows, :min_cols] = cov_matrix[:min_rows, :min_cols]
                cov_matrix = temp_cov

            # Check for NaN values in covariance matrix
            if np.isnan(cov_matrix).any() or np.isinf(cov_matrix).any():
                logger.warning("Covariance matrix contains NaN or inf values. Fixing...")
                # Replace NaN with average of non-NaN values
                mask = np.isfinite(cov_matrix)
                mean_val = np.mean(cov_matrix[mask]) if mask.any() else 0.01
                cov_matrix[~mask] = mean_val

            # Ensure positive definiteness
            try:
                eigvals = np.linalg.eigvals(cov_matrix)
                min_eig = np.min(np.real(eigvals))

                if min_eig < 0:
                    # Add a small positive value to the diagonal
                    adjustment = (-min_eig + 1e-6)
                    logger.info(f"Adjusting covariance matrix by {adjustment} to ensure positive definiteness")
                    cov_matrix = cov_matrix + adjustment * np.eye(cov_matrix.shape[0])

                # Double-check positive definiteness
                eigvals_after = np.linalg.eigvals(cov_matrix)
                min_eig_after = np.min(np.real(eigvals_after))
                if min_eig_after < 0:
                    logger.warning(
                        f"Covariance matrix still not positive definite. Min eigenvalue: {min_eig_after}")
                    # As a last resort, use a scaled identity matrix
                    var_est = np.mean(np.diag(cov_matrix))
                    cov_matrix = np.eye(cov_matrix.shape[0]) * max(0.01, var_est)

            except Exception as e:
                logger.error(f"Error ensuring positive definiteness: {e}", exc_info=True)
                # Fallback to identity matrix
                return np.eye(returns_window.shape[1]) * 0.01

            return cov_matrix

        except Exception as e:
            logger.error(f"Error computing covariance matrix: {e}", exc_info=True)
            print(f"Error computing covariance matrix: {e}")
            return np.eye(returns_window.shape[1]) * 0.01

    def compute_all_portfolio_weights(self, window_size=120, start_year=2013, end_year=2023):
        """
        Compute all portfolio weights for the specified time period.
        """
        print(f"Computing portfolio weights from {start_year} to {end_year}...")

        try:
            # Validate input years
            available_years = [col for col in self.market_cap_annual_df.columns
                               if isinstance(col, (int, str)) and (isinstance(col, int) or
                                                                   (isinstance(col, str) and col.isdigit()))]
            available_years = sorted(available_years)

            if not available_years:
                raise ValueError("No years available in market capitalization data")

            # Adjust start and end years to available data range
            start_year = max(start_year, int(available_years[0]))
            end_year = min(end_year, int(available_years[-1]))

            print(f"Adjusted years: {start_year} to {end_year}")
            print(f"Available market cap years: {available_years}")

            # Get returns data
            returns_cols = [col for col in self.returns_df.columns if col not in ['ISIN', 'Name', 'NAME']]
            returns_data = self.returns_df[returns_cols].copy()

            # Convert to numeric
            for col in returns_cols:
                returns_data[col] = pd.to_numeric(returns_data[col], errors='coerce')

            # Convert to numpy array - rows are companies, columns are time periods
            returns_array = returns_data.values

            # Transpose to get time x assets format
            returns_array = returns_array.T

            # Get dimensions
            n_periods, n_companies = returns_array.shape
            print(f"Returns data shape after transpose: {returns_array.shape} (periods x companies)")

            # Store the number of filtered companies for reference
            n_filtered_companies = len(self.filtered_isins)
            logger.info(f"Number of filtered companies: {n_filtered_companies}")

            # Process each year
            for year in range(start_year, end_year + 1):
                try:
                    # Find the closest available year for market cap data
                    closest_year = min(available_years, key=lambda x: abs(int(str(x)) - year))
                    print(f"Market cap data for {year} using closest year: {closest_year}")

                    # Get market cap data for the year - make sure it's numeric
                    market_cap = pd.to_numeric(self.market_cap_annual_df[str(closest_year)], errors='coerce').values

                    # Ensure we use only filtered companies for market cap
                    filtered_market_cap = np.zeros(n_filtered_companies)
                    # Map the market caps to filtered companies by ISIN
                    for i, isin in enumerate(self.filtered_isins):
                        isin_idx = self.market_cap_annual_df[self.market_cap_annual_df['ISIN'] == isin].index
                        if len(isin_idx) > 0:
                            idx = isin_idx[0]
                            if idx < len(market_cap):
                                filtered_market_cap[i] = market_cap[idx]

                    # Handle missing values
                    filtered_market_cap = np.nan_to_num(filtered_market_cap, 0)

                    # Compute value-weighted benchmark weights
                    total_market_cap = np.sum(filtered_market_cap)
                    if total_market_cap > 0:
                        vw_weights = filtered_market_cap / total_market_cap
                    else:
                        # Fallback to equal weights
                        vw_weights = np.ones(n_filtered_companies) / n_filtered_companies

                    # Ensure correct dimension
                    if len(vw_weights) != n_filtered_companies:
                        temp_weights = np.zeros(n_filtered_companies)
                        min_len = min(len(vw_weights), n_filtered_companies)
                        temp_weights[:min_len] = vw_weights[:min_len]
                        vw_weights = temp_weights
                        if np.sum(vw_weights) > 0:
                            vw_weights = vw_weights / np.sum(vw_weights)

                    # Store the value-weighted benchmark weights
                    self.vw_portfolio_weights[year] = vw_weights

                    # Calculate carbon footprint of benchmark portfolio
                    waci_vw, cf_vw = self.calculate_portfolio_carbon_footprint(vw_weights, str(year))
                    self.carbon_footprints[f"vw_{year}"] = (waci_vw, cf_vw)

                    print(f"Year {year} - VW Portfolio - WACI: {waci_vw:.2f}, CF: {cf_vw:.2f}")

                    # Calcola il VW portfolio per il 2013 specificamente
                    if year == 2013:
                        # Calcola e salva il carbon footprint del value-weighted portfolio per l'anno base
                        _, cf_base_2013 = self.calculate_portfolio_carbon_footprint(vw_weights, str(year))
                        self.carbon_footprints[f"vw_{year}"] = (waci_vw, cf_base_2013)
                        print(f"Year 2013 (Base Year) - VW Portfolio Carbon Footprint: {cf_base_2013:.2f}")

                    # Create a window of returns for calculating expected returns and covariance
                    # Use the standard 10-year window (120 months) of data before each allocation year

                    # Convert year-end to datetime for filtering
                    year_end = pd.Timestamp(f"{year - 1}-12-31")

                    # Get all date columns from returns dataframe
                    date_cols = [col for col in returns_cols if '-' in col]  # Date columns typically have hyphens
                    date_cols_dt = pd.to_datetime(date_cols, errors='coerce')

                    # Filter to get only columns in the lookback window
                    mask = (date_cols_dt <= year_end) & (
                            date_cols_dt > year_end - pd.DateOffset(months=window_size))
                    lookback_cols = [date_cols[i] for i, val in enumerate(mask) if val]

                    if len(lookback_cols) < 24:  # Require at least 2 years of data
                        logger.warning(
                            f"Not enough return data for year {year}. Only {len(lookback_cols)} months available.")
                        lookback_cols = date_cols[:min(len(date_cols), window_size)]

                    # Extract window of returns only for filtered companies
                    filtered_returns_df = self.returns_df[self.returns_df['ISIN'].isin(self.filtered_isins)]
                    lookback_returns = filtered_returns_df[lookback_cols].values

                    # Ensure proper shape - transpose if needed to get time periods x companies
                    if lookback_returns.shape[0] < lookback_returns.shape[1]:
                        lookback_returns = lookback_returns.T

                    # Check if we have enough valid data for covariance estimation
                    valid_data_ratio = np.sum(~np.isnan(lookback_returns)) / lookback_returns.size
                    if valid_data_ratio < 0.5:  # If more than 50% is missing, warn user
                        logger.warning(f"Low valid data ratio ({valid_data_ratio:.2f}) for year {year}")

                    # Clean invalid values and compute expected returns
                    lookback_returns_clean = np.nan_to_num(lookback_returns, 0)
                    expected_returns = np.nanmean(lookback_returns_clean, axis=0)

                    # Ensure expected returns has the correct dimension
                    if len(expected_returns) != n_filtered_companies:
                        temp_returns = np.zeros(n_filtered_companies)
                        min_len = min(len(expected_returns), n_filtered_companies)
                        temp_returns[:min_len] = expected_returns[:min_len]
                        expected_returns = temp_returns

                    # Compute covariance matrix
                    cov_matrix = self.compute_covariance_matrix(lookback_returns_clean)

                    # Ensure covariance matrix has correct dimensions
                    if cov_matrix.shape[0] != n_filtered_companies or cov_matrix.shape[1] != n_filtered_companies:
                        temp_cov = np.eye(n_filtered_companies) * 0.01
                        min_rows = min(cov_matrix.shape[0], n_filtered_companies)
                        min_cols = min(cov_matrix.shape[1], n_filtered_companies)
                        temp_cov[:min_rows, :min_cols] = cov_matrix[:min_rows, :min_cols]
                        cov_matrix = temp_cov

                    # Compute minimum variance portfolio weights
                    mv_weights = self.compute_minimum_variance_weights(expected_returns, cov_matrix)

                    # Ensure minimum variance weights have correct dimension
                    if len(mv_weights) != n_filtered_companies:
                        temp_weights = np.zeros(n_filtered_companies)
                        min_len = min(len(mv_weights), n_filtered_companies)
                        temp_weights[:min_len] = mv_weights[:min_len]
                        mv_weights = temp_weights
                        if np.sum(mv_weights) > 0:
                            mv_weights = mv_weights / np.sum(mv_weights)

                    self.mv_portfolio_weights[year] = mv_weights

                    # Calculate carbon footprint of minimum variance portfolio
                    waci_mv, cf_mv = self.calculate_portfolio_carbon_footprint(mv_weights, str(year))
                    self.carbon_footprints[f"mv_{year}"] = (waci_mv, cf_mv)

                    print(f"Year {year} - MV Portfolio - WACI: {waci_mv:.2f}, CF: {cf_mv:.2f}")

                    # Compute minimum variance portfolio with carbon constraint (50% of MV)
                    carbon_limit = 0.5 * cf_mv
                    mv_carbon_weights = self.compute_mv_portfolio_with_carbon_constraint(
                        expected_returns, cov_matrix, str(year), carbon_limit
                    )

                    # Ensure carbon-constrained MV weights have correct dimension
                    if len(mv_carbon_weights) != n_filtered_companies:
                        temp_weights = np.zeros(n_filtered_companies)
                        min_len = min(len(mv_carbon_weights), n_filtered_companies)
                        temp_weights[:min_len] = mv_carbon_weights[:min_len]
                        mv_carbon_weights = temp_weights
                        if np.sum(mv_carbon_weights) > 0:
                            mv_carbon_weights = mv_carbon_weights / np.sum(mv_carbon_weights)

                    self.mv_carbon_weights[year] = mv_carbon_weights

                    # Calculate carbon footprint of carbon-constrained minimum variance portfolio
                    waci_mvc, cf_mvc = self.calculate_portfolio_carbon_footprint(mv_carbon_weights, str(year))
                    self.carbon_footprints[f"mvc_{year}"] = (waci_mvc, cf_mvc)

                    print(f"Year {year} - MV Carbon Portfolio - WACI: {waci_mvc:.2f}, CF: {cf_mvc:.2f}")

                    # Verifica che il carbon footprint sia correttamente ridotto
                    if cf_mvc > carbon_limit:
                        logger.warning(
                            f"Year {year}: MV Carbon Portfolio exceeds limit: {cf_mvc:.2f} > {carbon_limit:.2f}")
                        # Prova a ricalcolare con un limite leggermente più alto
                        carbon_limit_relaxed = carbon_limit * 1.05
                        mv_carbon_weights = self.compute_mv_portfolio_with_carbon_constraint(
                            expected_returns, cov_matrix, str(year), carbon_limit_relaxed
                        )
                        waci_mvc, cf_mvc = self.calculate_portfolio_carbon_footprint(mv_carbon_weights, str(year))
                        self.carbon_footprints[f"mvc_{year}"] = (waci_mvc, cf_mvc)
                        print(
                            f"Year {year} - MV Carbon Portfolio (Recalculated) - WACI: {waci_mvc:.2f}, CF: {cf_mvc:.2f}")

                    # Compute value-weighted portfolio with carbon constraint (50% of VW)
                    carbon_limit = 0.5 * cf_vw
                    vw_carbon_weights = self.compute_tracking_error_portfolio_with_carbon_constraint(
                        vw_weights, cov_matrix, str(year), carbon_limit
                    )

                    # Ensure carbon-constrained VW weights have correct dimension
                    if len(vw_carbon_weights) != n_filtered_companies:
                        temp_weights = np.zeros(n_filtered_companies)
                        min_len = min(len(vw_carbon_weights), n_filtered_companies)
                        temp_weights[:min_len] = vw_carbon_weights[:min_len]
                        vw_carbon_weights = temp_weights
                        if np.sum(vw_carbon_weights) > 0:
                            vw_carbon_weights = vw_carbon_weights / np.sum(vw_carbon_weights)

                    self.vw_carbon_weights[year] = vw_carbon_weights

                    # Calculate carbon footprint of carbon-constrained value-weighted portfolio
                    waci_vwc, cf_vwc = self.calculate_portfolio_carbon_footprint(vw_carbon_weights, str(year))
                    self.carbon_footprints[f"vwc_{year}"] = (waci_vwc, cf_vwc)

                    print(f"Year {year} - VW Carbon Portfolio - WACI: {waci_vwc:.2f}, CF: {cf_vwc:.2f}")

                    # Compute net zero portfolio (only from start_year onwards)
                    if year >= start_year:
                        # Get carbon footprint of value-weighted portfolio in base year (2013)
                        base_year = str(start_year)
                        if f"vw_{start_year}" in self.carbon_footprints:
                            _, cf_base = self.carbon_footprints[f"vw_{start_year}"]
                        else:
                            # Calculate it if not already done
                            _, cf_base = self.calculate_portfolio_carbon_footprint(
                                self.vw_portfolio_weights[start_year], base_year
                            )

                        # Compute net zero portfolio weights
                        nz_weights = self.compute_net_zero_portfolio(
                            vw_weights, cov_matrix, str(year), cf_base, theta=0.1, base_year=base_year
                        )

                        # Ensure net zero weights have correct dimension
                        if len(nz_weights) != n_filtered_companies:
                            temp_weights = np.zeros(n_filtered_companies)
                            min_len = min(len(nz_weights), n_filtered_companies)
                            temp_weights[:min_len] = nz_weights[:min_len]
                            nz_weights = temp_weights
                            if np.sum(nz_weights) > 0:
                                nz_weights = nz_weights / np.sum(nz_weights)

                        self.nz_portfolio_weights[year] = nz_weights

                        # Calculate carbon footprint of net zero portfolio
                        waci_nz, cf_nz = self.calculate_portfolio_carbon_footprint(nz_weights, str(year))
                        self.carbon_footprints[f"nz_{year}"] = (waci_nz, cf_nz)

                        print(f"Year {year} - Net Zero Portfolio - WACI: {waci_nz:.2f}, CF: {cf_nz:.2f}")

                    # Calculate returns for portfolios in this year
                    monthly_returns = self.compute_monthly_portfolio_returns(year)
                    for portfolio_type, returns in monthly_returns.items():
                        if portfolio_type == 'mv':
                            self.mv_returns[year] = returns
                        elif portfolio_type == 'vw':
                            self.vw_returns[year] = returns
                        elif portfolio_type == 'mvc':
                            self.mvc_returns[year] = returns
                        elif portfolio_type == 'vwc':
                            self.vwc_returns[year] = returns
                        elif portfolio_type == 'nz':
                            self.nz_returns[year] = returns

                except Exception as e:
                    logger.error(f"Error processing year {year}: {str(e)}", exc_info=True)
                    print(f"Error processing year {year}: {str(e)}")
                    continue

            print("All portfolio weights computed successfully.")

        except Exception as e:
            logger.error(f"Global error in compute_all_portfolio_weights: {str(e)}", exc_info=True)
            print(f"Error in portfolio weight computation: {str(e)}")

    def compute_monthly_portfolio_returns(self, year):
        """
        Compute monthly portfolio returns for a given year.

        Parameters:
            year (int): Year to compute returns for

        Returns:
            dict: Dictionary with monthly returns for each portfolio type
        """
        try:
            # Get monthly date columns for the year
            year_str = str(year)
            date_cols = [col for col in self.returns_df.columns if year_str in col and '-' in col]
            date_cols.sort()  # Ensure chronological order

            # Create result dictionary
            results = {
                'mv': {},  # Minimum variance
                'vw': {},  # Value-weighted
                'mvc': {},  # MV with carbon constraint
                'vwc': {},  # VW with carbon constraint
                'nz': {}  # Net zero
            }

            # Get weights for each portfolio type
            mv_weights = self.mv_portfolio_weights.get(year, None)
            vw_weights = self.vw_portfolio_weights.get(year, None)
            mvc_weights = self.mv_carbon_weights.get(year, None)
            vwc_weights = self.vw_carbon_weights.get(year, None)
            nz_weights = self.nz_portfolio_weights.get(year, None)

            # Get returns data from filtered companies only - make sure to filter first
            filtered_returns = self.returns_df[self.returns_df['ISIN'].isin(self.filtered_isins)].copy()
            filtered_returns = filtered_returns.sort_values('ISIN').reset_index(drop=True)

            # For each month, calculate portfolio returns
            for date_col in date_cols:
                # Extract returns for this month - ONLY for filtered companies
                monthly_returns = pd.to_numeric(filtered_returns[date_col], errors='coerce').values
                monthly_returns = np.nan_to_num(monthly_returns, 0)

                # Make sure weights and returns have the same length
                if len(monthly_returns) != len(self.filtered_isins):
                    logger.warning(
                        f"Returns length mismatch: {len(monthly_returns)} vs expected {len(self.filtered_isins)}")
                    continue

                # Ensure weights match the filtered companies dimension
                # FIX: Resize the weights arrays to match the returns dimension
                if mv_weights is not None:
                    # Create a new array of the correct size
                    resized_mv_weights = np.zeros(len(monthly_returns))
                    # Copy the available weights (up to min length)
                    common_length = min(len(mv_weights), len(monthly_returns))
                    resized_mv_weights[:common_length] = mv_weights[:common_length]
                    # Normalize to sum to 1
                    if np.sum(resized_mv_weights) > 0:
                        resized_mv_weights = resized_mv_weights / np.sum(resized_mv_weights)
                    mv_return = np.sum(resized_mv_weights * monthly_returns)
                    results['mv'][date_col] = mv_return

                if vw_weights is not None:
                    # Resize value-weighted weights
                    resized_vw_weights = np.zeros(len(monthly_returns))
                    common_length = min(len(vw_weights), len(monthly_returns))
                    resized_vw_weights[:common_length] = vw_weights[:common_length]
                    if np.sum(resized_vw_weights) > 0:
                        resized_vw_weights = resized_vw_weights / np.sum(resized_vw_weights)
                    vw_return = np.sum(resized_vw_weights * monthly_returns)
                    results['vw'][date_col] = vw_return

                if mvc_weights is not None:
                    # Resize MV carbon-constrained weights
                    resized_mvc_weights = np.zeros(len(monthly_returns))
                    common_length = min(len(mvc_weights), len(monthly_returns))
                    resized_mvc_weights[:common_length] = mvc_weights[:common_length]
                    if np.sum(resized_mvc_weights) > 0:
                        resized_mvc_weights = resized_mvc_weights / np.sum(resized_mvc_weights)
                    mvc_return = np.sum(resized_mvc_weights * monthly_returns)
                    results['mvc'][date_col] = mvc_return

                if vwc_weights is not None:
                    # Resize VW carbon-constrained weights
                    resized_vwc_weights = np.zeros(len(monthly_returns))
                    common_length = min(len(vwc_weights), len(monthly_returns))
                    resized_vwc_weights[:common_length] = vwc_weights[:common_length]
                    if np.sum(resized_vwc_weights) > 0:
                        resized_vwc_weights = resized_vwc_weights / np.sum(resized_vwc_weights)
                    vwc_return = np.sum(resized_vwc_weights * monthly_returns)
                    results['vwc'][date_col] = vwc_return

                if nz_weights is not None:
                    # Resize Net Zero weights
                    resized_nz_weights = np.zeros(len(monthly_returns))
                    common_length = min(len(nz_weights), len(monthly_returns))
                    resized_nz_weights[:common_length] = nz_weights[:common_length]
                    if np.sum(resized_nz_weights) > 0:
                        resized_nz_weights = resized_nz_weights / np.sum(resized_nz_weights)
                    nz_return = np.sum(resized_nz_weights * monthly_returns)
                    results['nz'][date_col] = nz_return

            return results

        except Exception as e:
            logger.error(f"Error computing monthly returns for year {year}: {str(e)}", exc_info=True)
            print(f"Error computing monthly returns for year {year}: {str(e)}")
            return {'mv': {}, 'vw': {}, 'mvc': {}, 'vwc': {}, 'nz': {}}

    # Add these methods to the CarbonAwarePortfolio class
    def plot_carbon_footprints(self, start_year=2013, end_year=2023):
        """Plot carbon footprints of all portfolio strategies over time."""
        years = range(start_year, end_year + 1)
        portfolios = ['mv', 'vw', 'mvc', 'vwc', 'nz']
        labels = ['Minimum Variance', 'Value-Weighted', 'MV Carbon-Constrained',
                  'VW Carbon-Constrained', 'Net Zero']

        plt.figure(figsize=(12, 6))
        for i, portfolio in enumerate(portfolios):
            footprints = []
            for year in years:
                key = f"{portfolio}_{year}"
                if key in self.carbon_footprints:
                    footprints.append(self.carbon_footprints[key][1])
                else:
                    footprints.append(np.nan)
            plt.plot(years, footprints, marker='o', label=labels[i])

        plt.title("Carbon Footprints Over Time")
        plt.xlabel("Year")
        plt.ylabel("Carbon Footprint (tCO2e/$M invested)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig('carbon_footprints.png', dpi=300)
        plt.close()

    def plot_performance_carbon_tradeoff(self, start_year=2013, end_year=2023):
        """Plot performance vs carbon footprint tradeoff."""
        plt.figure(figsize=(10, 6))
        portfolios = ['mv', 'vw', 'mvc', 'vwc', 'nz']
        labels = ['Minimum Variance', 'Value-Weighted', 'MV Carbon-Constrained',
                  'VW Carbon-Constrained', 'Net Zero']

        for i, portfolio in enumerate(portfolios):
            returns = []
            footprints = []
            for year in range(start_year, end_year + 1):
                # Get annual return
                if year in getattr(self, f"{portfolio}_returns"):
                    ann_return = np.prod([1 + r for r in self.__dict__[f"{portfolio}_returns"][year].values()]) - 1
                    returns.append(ann_return)

                # Get footprint
                key = f"{portfolio}_{year}"
                if key in self.carbon_footprints:
                    footprints.append(self.carbon_footprints[key][1])

            if len(returns) == len(footprints):
                plt.scatter(footprints, returns, label=labels[i])

        plt.title("Performance vs Carbon Footprint Tradeoff")
        plt.xlabel("Carbon Footprint (tCO2e/$M invested)")
        plt.ylabel("Annual Return")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('performance_carbon_tradeoff.png', dpi=300)
        plt.close()

    def compare_portfolio_performance(self, start_year=2013, end_year=2023):
        """
        Compute and compare performance metrics for all portfolio strategies.

        Returns:
            pd.DataFrame: DataFrame with metrics (Annualized Return, Volatility, Sharpe Ratio, Carbon Footprint)
        """
        metrics = []
        portfolios = ['mv', 'vw', 'mvc', 'vwc', 'nz']
        labels = ['Minimum Variance', 'Value-Weighted', 'MV Carbon-Constrained',
                  'VW Carbon-Constrained', 'Net Zero']

        for portfolio, label in zip(portfolios, labels):
            # Collect all monthly returns
            returns = []
            for year in range(start_year, end_year + 1):
                if year in self.__dict__.get(f"{portfolio}_returns", {}):
                    returns.extend(list(self.__dict__[f"{portfolio}_returns"][year].values()))

            if not returns:
                continue

            returns_series = pd.Series(returns)

            # Annualized return
            annual_return = (1 + returns_series.mean()) ** 12 - 1

            # Annualized volatility
            annual_volatility = returns_series.std() * np.sqrt(12)

            # Sharpe ratio (assuming risk-free rate = 0)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else np.nan

            # Average carbon footprint
            carbon_footprints = [self.carbon_footprints.get(f"{portfolio}_{year}", (np.nan, np.nan))[1]
                                 for year in range(start_year, end_year + 1)]
            avg_carbon = np.nanmean(carbon_footprints)

            metrics.append({
                'Portfolio': label,
                'Annualized Return': annual_return,
                'Volatility': annual_volatility,
                'Sharpe Ratio': sharpe_ratio,
                'Avg Carbon Footprint': avg_carbon
            })

        return pd.DataFrame(metrics)

    def compute_portfolio_returns(self, weights_dict, start_year=2014, end_year=2023):
        """
        Compute ex-post portfolio returns using the computed weights.

        Parameters:
            weights_dict (dict): Dictionary of portfolio weights (year -> weights)
            start_year (int): Starting year for returns calculation
            end_year (int): Ending year for returns calculation

        Returns:
            pd.Series: Monthly portfolio returns
        """
        try:
            # Get the returns data - make sure to filter and sort by ISIN
            filtered_returns = self.returns_df[self.returns_df['ISIN'].isin(self.filtered_isins)].copy()
            filtered_returns = filtered_returns.sort_values('ISIN').reset_index(drop=True)

            # Get all date columns (monthly returns)
            date_cols = [col for col in filtered_returns.columns if '-' in col]  # Date columns contain hyphens
            date_cols = sorted(date_cols)  # Ensure chronological order

            # Convert dates to datetime objects
            dates = pd.to_datetime(date_cols, errors='coerce')

            # Filter dates to the specified range
            start_date = pd.Timestamp(f"{start_year}-01-01")
            end_date = pd.Timestamp(f"{end_year}-12-31")
            mask = (dates >= start_date) & (dates <= end_date)
            selected_dates = [date_cols[i] for i, include in enumerate(mask) if include]

            if not selected_dates:
                logger.warning(f"No dates found between {start_year} and {end_year}")
                return pd.Series()

            # Create a dictionary to map each date to its corresponding weights (use previous year's weights)
            weights_map = {}
            for date in pd.to_datetime(selected_dates):
                # For a given date, use weights from December of the previous year
                year = date.year
                weights_map[date] = weights_dict.get(year - 1, None)

                # If weights not available, try using weights from current year
                if weights_map[date] is None:
                    weights_map[date] = weights_dict.get(year, None)

            # Calculate portfolio returns for each month
            returns_list = []
            dates_list = []

            for date_str in selected_dates:
                date = pd.to_datetime(date_str)
                if date not in weights_map or weights_map[date] is None:
                    continue

                weights = weights_map[date]

                # Get returns for this month
                monthly_returns = pd.to_numeric(filtered_returns[date_str], errors='coerce').values

                # Handle missing values
                monthly_returns = np.nan_to_num(monthly_returns, 0)

                # FIX: Resize weights to match returns dimension
                if len(weights) != len(monthly_returns):
                    resized_weights = np.zeros(len(monthly_returns))
                    common_length = min(len(weights), len(monthly_returns))
                    resized_weights[:common_length] = weights[:common_length]

                    # Normalize weights to sum to 1
                    if np.sum(resized_weights) > 0:
                        resized_weights = resized_weights / np.sum(resized_weights)
                    weights = resized_weights

                    logger.info(f"Resized weights from {len(weights)} to {len(monthly_returns)} for {date_str}")

                # Normalize weights to sum to 1
                if np.sum(weights) > 0:
                    weights = weights / np.sum(weights)

                # Calculate portfolio return
                portfolio_return = np.sum(weights * monthly_returns)

                # Store result
                returns_list.append(portfolio_return)
                dates_list.append(date)

            # Create Series with returns
            portfolio_returns = pd.Series(returns_list, index=dates_list)

            return portfolio_returns

        except Exception as e:
            logger.error(f"Error computing portfolio returns: {str(e)}", exc_info=True)
            print(f"Error computing portfolio returns: {str(e)}")
            return pd.Series()