import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import warnings
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass


class DataProcessor:
    """Handles data processing with robust error handling and validation"""

    def __init__(self, max_missing: float = 0.2):
        self.max_missing = max_missing
        self.carbon_start_year = 2013
        self.carbon_end_year = 2023
        self.price_start_date = '2004-01-31'
        self.price_end_date = '2023-12-31'
        self.valid_regions = ['EUR', 'NA', 'ASIA', 'EMERG']
        self.acceptable_return_max = 1.0  # 100% maximum return
        self.acceptable_return_min = -0.9  # -90% minimum return

    def extract_european_companies(self, static_file: str) -> Dict[str, List]:
        """
        Extracts European companies from static file with error handling

        Args:
            static_file: Path to the static companies file

        Returns:
            Dictionary containing company information

        Raises:
            DataValidationError: If file structure is invalid
        """
        try:
            static_df = pd.read_excel(static_file)
            logger.info(f"Loaded static file with {len(static_df)} companies")

            # Handle different column naming conventions
            column_mappings = {
                'Name': ['Name', 'NAME', 'Company Name', 'Company_Name'],
                'ISIN': ['ISIN', 'Isin', 'ISIN Code', 'ISIN_Code'],
                'Country': ['Country', 'COUNTRY', 'Nation'],
                'Region': ['Region', 'REGION', 'Area']
            }

            # Standardize column names
            for standard_name, variants in column_mappings.items():
                for variant in variants:
                    if variant in static_df.columns and standard_name not in static_df.columns:
                        static_df = static_df.rename(columns={variant: standard_name})
                        break

            # Validate required columns
            required_columns = ['ISIN', 'Name', 'Region']
            missing_columns = [col for col in required_columns if col not in static_df.columns]
            if missing_columns:
                raise DataValidationError(f"Missing required columns: {missing_columns}")

            # Check for duplicate ISINs
            duplicate_isins = static_df[static_df.duplicated(subset=['ISIN'], keep=False)]
            if not duplicate_isins.empty:
                logger.warning(
                    f"Found {len(duplicate_isins)} duplicate ISINs in static file. Keeping first occurrence.")
                static_df = static_df.drop_duplicates(subset=['ISIN'], keep='first')

            # Validate region values
            invalid_regions = static_df[~static_df['Region'].isin(self.valid_regions)]['Region'].unique()
            if len(invalid_regions) > 0:
                logger.warning(f"Found invalid regions: {invalid_regions}")

            # Filter European companies
            europe_df = static_df[static_df['Region'] == 'EUR'].copy()

            if europe_df.empty:
                raise DataValidationError("No European companies found in static file")

            logger.info(f"Found {len(europe_df)} European companies")
            return {
                "ISIN": europe_df["ISIN"].tolist(),
                "Name": europe_df["Name"].tolist(),
                "Country": europe_df["Country"].tolist() if "Country" in europe_df.columns else ["Unknown"] * len(
                    europe_df),
                "Region": europe_df["Region"].tolist()
            }

        except Exception as e:
            logger.error(f"Error processing static file: {e}")
            raise

    def validate_data_structure(self, df: pd.DataFrame, expected_type: str) -> Tuple[bool, List[str]]:
        """
        Validates the structure of input data based on expected type

        Args:
            df: DataFrame to validate
            expected_type: Type of data ('carbon', 'market_cap', 'revenue', 'price')

        Returns:
            Tuple of (is_valid, valid_columns)
        """
        if df is None or df.empty:
            logger.error(f"Empty or None DataFrame for {expected_type} validation")
            return False, []

        # Log columns for debugging
        logger.info(f"Columns in {expected_type} data: {list(df.columns)}")

        # Check for required identifier columns
        if 'ISIN' not in df.columns:
            logger.error(f"Missing ISIN column in {expected_type} data")
            return False, []

        valid_columns = []

        if expected_type == 'carbon':
            # For carbon data, check if we have years in the expected range
            year_cols = []
            for col in df.columns:
                try:
                    if isinstance(col, (int, float)):
                        year = int(col)
                        if self.carbon_start_year <= year <= self.carbon_end_year:
                            year_cols.append(str(year))
                            valid_columns.append(str(year))
                    elif isinstance(col, str) and col.isdigit():
                        year = int(col)
                        if self.carbon_start_year <= year <= self.carbon_end_year:
                            year_cols.append(col)
                            valid_columns.append(col)
                except:
                    continue

            if not year_cols:
                logger.warning(f"No carbon data found for years {self.carbon_start_year}-{self.carbon_end_year}")
                return False, []
            else:
                logger.info(f"Found carbon data for years: {sorted(year_cols)}")
                return True, valid_columns

        elif expected_type == 'market_cap' or expected_type == 'price':
            # Check for date columns within the specified range
            date_cols = [col for col in df.columns if col not in ['ISIN', 'NAME', 'Name']]

            if expected_type == 'price':
                # For monthly price data, check for date format columns
                valid_date_cols = []
                for col in date_cols:
                    try:
                        if isinstance(col, str) and ('-' in col or '/' in col):
                            date = pd.to_datetime(col)
                            if pd.Timestamp(self.price_start_date) <= date <= pd.Timestamp(self.price_end_date):
                                valid_date_cols.append(col)
                                valid_columns.append(col)
                    except:
                        continue

                if not valid_date_cols:
                    logger.warning(
                        f"No valid date columns found in {expected_type} data for {self.price_start_date}-{self.price_end_date}")
                    return False, []
                else:
                    logger.info(f"Found {len(valid_date_cols)} valid date columns for {expected_type} data")
                    return True, valid_columns
            else:
                # For yearly market cap data, check for year columns
                year_cols = []
                for col in date_cols:
                    try:
                        if isinstance(col, (int, float)):
                            year = int(col)
                            if 1999 <= year <= 2024:  # Market cap data range from project spec
                                year_cols.append(str(year))
                                valid_columns.append(str(year))
                        elif isinstance(col, str) and col.isdigit():
                            year = int(col)
                            if 1999 <= year <= 2024:
                                year_cols.append(col)
                                valid_columns.append(col)
                        elif isinstance(col, str):
                            # Try parsing as date and extract year
                            date = pd.to_datetime(col, errors='coerce')
                            if pd.notna(date) and 1999 <= date.year <= 2024:
                                year_cols.append(str(date.year))
                                valid_columns.append(str(date.year))
                    except:
                        continue

                if not year_cols:
                    logger.warning(f"No valid year columns found in {expected_type} data")
                    return False, []
                else:
                    logger.info(f"Found {len(year_cols)} valid year columns for {expected_type} data")
                    return True, valid_columns

        elif expected_type == 'revenue':
            # Check for annual revenue data
            year_cols = []
            for col in df.columns:
                try:
                    if isinstance(col, (int, float)):
                        year = int(col)
                        if 1999 <= year <= 2024:  # Revenue data range from project spec
                            year_cols.append(str(year))
                            valid_columns.append(str(year))
                    elif isinstance(col, str) and col.isdigit():
                        year = int(col)
                        if 1999 <= year <= 2024:
                            year_cols.append(col)
                            valid_columns.append(col)
                    elif isinstance(col, str):
                        # Try parsing as date and extract year
                        date = pd.to_datetime(col, errors='coerce')
                        if pd.notna(date) and 1999 <= date.year <= 2024:
                            year_cols.append(str(date.year))
                            valid_columns.append(str(date.year))
                except:
                    continue

            if len(year_cols) < 10:  # Expect at least 10 years of data
                logger.warning(f"Limited revenue data: only {len(year_cols)} years")
                return False if len(year_cols) == 0 else True, valid_columns
            else:
                logger.info(f"Found revenue data for {len(year_cols)} years")
                return True, valid_columns

        return True, valid_columns

    def forward_fill_carbon_data(self, df: pd.DataFrame, year_cols: List[str]) -> pd.DataFrame:
        """
        Implements the forward-fill logic for carbon data as specified in the project:
        - Fill missing values between available years with previous year's value
        - Fill missing values at the end with previous year's value
        - Leave missing values at the beginning (company not investable until data is available)

        Args:
            df: DataFrame with carbon data
            year_cols: List of year columns in chronological order

        Returns:
            DataFrame with forward-filled values
        """
        logger.info(f"Forward-filling carbon data for {len(df)} companies across {len(year_cols)} years")

        # Make a copy to avoid modifying the original
        filled_df = df.copy()

        # Convert to numeric
        for col in year_cols:
            filled_df[col] = pd.to_numeric(filled_df[col], errors='coerce')

        # Log companies that will be excluded due to missing first year data
        excluded_companies = filled_df[pd.isna(filled_df[year_cols[0]])]['ISIN'].tolist()
        if excluded_companies:
            logger.warning(
                f"{len(excluded_companies)} companies excluded due to missing {year_cols[0]} data: {excluded_companies}")

        # Forward fill each company's data
        for idx in filled_df.index:
            prev_valid_value = None
            # First pass: identify last valid value for each missing data point
            for i, col in enumerate(year_cols):
                current_value = filled_df.loc[idx, col]
                if pd.notna(current_value):
                    prev_valid_value = current_value
                elif i > 0 and prev_valid_value is not None:
                    # Fill with previous valid value if not the first year
                    filled_df.loc[idx, col] = prev_valid_value

        # Count companies with missing first year data
        missing_first_year = filled_df[pd.isna(filled_df[year_cols[0]])].shape[0]
        if missing_first_year > 0:
            logger.warning(f"{missing_first_year} companies have missing data for the first year ({year_cols[0]})")
            logger.info("These companies will not be investable until data becomes available")

        return filled_df

    def handle_missing_data(self, df: pd.DataFrame, data_type: str, max_missing: float = None,
                            start_year: int = None, end_year: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Handles missing data according to project specifications with improved validation
        """
        if max_missing is None:
            max_missing = self.max_missing
        if start_year is None:
            start_year = self.carbon_start_year
        if end_year is None:
            end_year = self.carbon_end_year

        try:
            # Initialize outputs
            valid_df = pd.DataFrame()
            failed_df = pd.DataFrame()

            if df is None or df.empty or 'ISIN' not in df.columns:
                logger.warning(f"Invalid input data for {data_type}: DataFrame is empty or missing ISIN column.")
                return pd.DataFrame(), pd.DataFrame()  # Return empty DataFrames

            if data_type in ['carbon', 'revenue']:
                # For annual data (carbon, revenue), identify relevant year columns
                year_cols = []
                available_cols = []

                for col in df.columns:
                    try:
                        if isinstance(col, (int, float)):
                            year = int(col)
                        elif isinstance(col, str) and col.isdigit():
                            year = int(col)
                        else:
                            continue

                        if data_type == 'carbon' and start_year <= year <= end_year:
                            year_cols.append(str(year))
                            available_cols.append(col)
                        elif data_type == 'revenue' and 1999 <= year <= 2024:
                            year_cols.append(str(year))
                            available_cols.append(col)
                    except:
                        continue

                if not available_cols:
                    logger.warning(f"No valid year columns found for {data_type} data")
                    return pd.DataFrame(), df  # Return empty valid_df and all data as failed_df

                # Sort year columns chronologically
                year_cols = sorted(year_cols, key=lambda x: int(x))
                available_cols = sorted(available_cols,
                                        key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else int(x))

                logger.info(f"Processing {data_type} data with years: {year_cols}")

                # Create a copy with numeric columns for analysis
                data_subset = df[['ISIN', 'NAME'] + available_cols].copy() if 'NAME' in df.columns else df[
                    ['ISIN'] + available_cols].copy()
                for col in available_cols:
                    data_subset[col] = pd.to_numeric(data_subset[col], errors='coerce')

                # Calculate missing percentage
                missing_count = data_subset[available_cols].isnull().sum(axis=1)
                total_count = len(available_cols)

                if total_count == 0:
                    logger.warning(f"No valid columns found for {data_type} data")
                    return pd.DataFrame(), df

                missing_percent = missing_count / total_count

                # Log missing data statistics
                logger.info(f"Missing data statistics for {len(df)} companies:")
                logger.info(f"  Companies with 0% missing: {(missing_percent == 0).sum()}")
                logger.info(f"  Companies with >0% and <={max_missing * 100}% missing: "
                            f"{((missing_percent > 0) & (missing_percent <= max_missing)).sum()}")
                logger.info(f"  Companies with >{max_missing * 100}% missing: {(missing_percent > max_missing).sum()}")

                # Split into valid and failed based on missing percentage
                valid_mask = missing_percent <= max_missing
                valid_df = df[valid_mask].copy()
                failed_df = df[~valid_mask].copy()

                logger.info(f"After filtering: {len(valid_df)} valid, {len(failed_df)} failed")

                # For carbon data, apply specific forward-fill rules
                if data_type == 'carbon' and not valid_df.empty:
                    valid_df = self.forward_fill_carbon_data(valid_df, year_cols)

                # For revenue data, apply simpler forward-fill (can use previous year for all missing)
                elif data_type == 'revenue' and not valid_df.empty:
                    valid_df[available_cols] = valid_df[available_cols].fillna(method='ffill', axis=1)

                return valid_df, failed_df

            elif data_type == 'price':
                # For price data, work with date columns
                date_cols = [col for col in df.columns if col not in ['ISIN', 'NAME', 'Name']]

                # Convert string dates to datetime objects
                date_objects = []
                valid_cols = []

                for col in date_cols:
                    try:
                        date = pd.to_datetime(col)
                        if pd.Timestamp(self.price_start_date) <= date <= pd.Timestamp(self.price_end_date):
                            date_objects.append(date)
                            valid_cols.append(col)
                    except:
                        continue

                if not valid_cols:
                    logger.warning(f"No valid date columns found for price data")
                    return pd.DataFrame(), df

                # Sort date columns chronologically
                sorted_indices = np.argsort(date_objects)
                valid_cols = [valid_cols[i] for i in sorted_indices]

                logger.info(f"Processing price data with {len(valid_cols)} valid dates")

                # Create a subset for analysis
                data_subset = df[['ISIN', 'NAME'] + valid_cols].copy() if 'NAME' in df.columns else df[
                    ['ISIN'] + valid_cols].copy()

                # Convert to numeric and handle non-positive prices
                for col in valid_cols:
                    prices = pd.to_numeric(data_subset[col], errors='coerce')
                    non_positive = (prices <= 0).sum()
                    if non_positive > 0:
                        logger.warning(f"Found {non_positive} non-positive prices in column {col}. Treating as NaN.")
                        prices = prices.where(prices > 0, np.nan)
                    data_subset[col] = prices

                # Calculate missing percentage
                missing_count = data_subset[valid_cols].isnull().sum(axis=1)
                total_count = len(valid_cols)
                missing_percent = missing_count / total_count

                # Log missing data statistics
                logger.info(f"Missing price data statistics for {len(df)} companies:")
                logger.info(f"  Companies with 0% missing: {(missing_percent == 0).sum()}")
                logger.info(f"  Companies with >0% and <={max_missing * 100}% missing: "
                            f"{((missing_percent > 0) & (missing_percent <= max_missing)).sum()}")
                logger.info(f"  Companies with >{max_missing * 100}% missing: {(missing_percent > max_missing).sum()}")

                # Split into valid and failed
                valid_mask = missing_percent <= max_missing
                valid_df = df[valid_mask].copy()
                failed_df = df[~valid_mask].copy()

                # Clean the valid data
                if not valid_df.empty:
                    for col in valid_cols:
                        prices = pd.to_numeric(valid_df[col], errors='coerce')
                        valid_df[col] = prices.where(prices > 0, np.nan)

                    # Interpolate missing values
                    numeric_cols = [col for col in valid_cols if col in valid_df.columns]
                    valid_df[numeric_cols] = valid_df[numeric_cols].interpolate(method='linear', axis=1,
                                                                                limit_direction='both')

                logger.info(f"After price data cleaning: {len(valid_df)} valid, {len(failed_df)} failed")
                return valid_df, failed_df

            return pd.DataFrame(), df  # Default fallback

        except Exception as e:
            logger.error(f"Error in handle_missing_data for {data_type}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return empty DataFrames on error
            return pd.DataFrame(), df

    def standardize_date_columns(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Standardize date columns with robust error handling

        Args:
            df: DataFrame to standardize
            data_type: Type of data ('monthly' or 'annual')
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame passed to standardize_date_columns")
            return df

        try:
            new_columns = []

            for col in df.columns:
                if col in ['ISIN', 'NAME', 'Name']:
                    new_columns.append(col)
                else:
                    try:
                        if isinstance(col, (int, float)):
                            new_columns.append(str(int(col)))
                        elif isinstance(col, str):
                            if col.isdigit() and len(col) == 4:
                                new_columns.append(col)
                            elif '-' in col or '/' in col:
                                try:
                                    dt = pd.to_datetime(col)
                                    if data_type == 'monthly':
                                        new_columns.append(dt.strftime('%Y-%m-%d'))
                                    else:
                                        new_columns.append(str(dt.year))
                                except:
                                    new_columns.append(str(col))
                            else:
                                new_columns.append(str(col))
                        else:
                            new_columns.append(str(col))
                    except:
                        logger.warning(f"Could not parse column '{col}', keeping as string")
                        new_columns.append(str(col))

            df.columns = new_columns
            return df

        except Exception as e:
            logger.error(f"Error in standardize_date_columns: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return df

    def calculate_simple_returns(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate simple returns from price data with improved error handling
        and extreme return filtering
        """
        if price_df is None or price_df.empty:
            logger.warning("Empty price DataFrame passed to calculate_simple_returns")
            return pd.DataFrame()

        try:
            # Extract metadata and price columns
            metadata_cols = ['ISIN', 'NAME'] if 'NAME' in price_df.columns else ['ISIN', 'Name']
            metadata = price_df[metadata_cols].copy()

            # Identify price columns (columns that aren't metadata)
            price_cols = [col for col in price_df.columns if col not in metadata_cols]

            if not price_cols:
                logger.error("No price columns found in DataFrame")
                return pd.DataFrame()

            # Convert prices to numeric
            prices = price_df[price_cols].apply(pd.to_numeric, errors='coerce')

            # Check for non-positive prices
            non_positive_mask = (prices <= 0) | prices.isna()
            non_positive_count = non_positive_mask.sum().sum()

            if non_positive_count > 0:
                logger.warning(f"Found {non_positive_count} non-positive or NaN prices. Interpolating.")
                prices = prices.interpolate(method='linear', axis=1, limit_direction='both')

                # Recheck for remaining non-positive prices
                non_positive_mask = (prices <= 0) | prices.isna()
                if non_positive_mask.sum().sum() > 0:
                    logger.warning("Remaining non-positive prices after interpolation. Setting to column mean.")
                    for col in price_cols:
                        col_data = prices[col]
                        mean_val = col_data[col_data > 0].mean()
                        if pd.isna(mean_val):
                            mean_val = 1.0  # Default if no positive values
                        prices[col] = col_data.where(col_data > 0, mean_val)

            # Calculate simple returns
            returns = prices.pct_change(axis=1)

            # Handle extreme returns - Winsorize instead of setting to NaN
            extreme_returns_mask = (returns > self.acceptable_return_max) | (returns < self.acceptable_return_min)
            extreme_returns_count = extreme_returns_mask.sum().sum()

            if extreme_returns_count > 0:
                logger.warning(f"Found {extreme_returns_count} extreme returns. Winsorizing.")
                returns = returns.clip(lower=self.acceptable_return_min, upper=self.acceptable_return_max)

            # Interpolate missing returns
            returns = returns.interpolate(method='linear', axis=1, limit_direction='both', limit=5)

            # Drop first column which would be NaN after pct_change()
            returns = returns.iloc[:, 1:]

            # Combine metadata and returns
            returns_df = pd.concat([metadata, returns], axis=1)

            logger.info(
                f"Calculated returns for {len(returns_df)} companies across {len(returns.columns)} time periods")
            return returns_df

        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def validate_returns(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform additional validation and cleaning on returns
        """
        if returns_df is None or returns_df.empty:
            return returns_df

        try:
            # Extract metadata and return columns
            metadata_cols = ['ISIN', 'NAME'] if 'NAME' in returns_df.columns else ['ISIN', 'Name']
            returns_cols = [col for col in returns_df.columns if col not in metadata_cols]

            if not returns_cols:
                return returns_df

            # Check for companies with too many missing or invalid returns
            returns_data = returns_df[returns_cols].apply(pd.to_numeric, errors='coerce')
            missing_pct = returns_data.isnull().sum(axis=1) / len(returns_cols)

            valid_mask = missing_pct <= self.max_missing
            invalid_companies = (~valid_mask).sum()

            if invalid_companies > 0:
                logger.warning(f"Found {invalid_companies} companies with >{self.max_missing * 100}% missing returns")
                returns_df = returns_df[valid_mask].copy()

            # Identify any anomalies in return distributions
            mean_returns = returns_data.mean(axis=1)
            std_returns = returns_data.std(axis=1)

            # Flag companies with unusually high or low mean returns or volatility
            if not mean_returns.empty and not std_returns.empty:
                extreme_mean_mask = np.abs(mean_returns) > 0.1  # 10% monthly average return threshold
                extreme_std_mask = std_returns > 0.5  # 50% monthly std dev threshold

                extreme_companies = (extreme_mean_mask | extreme_std_mask).sum()
                if extreme_companies > 0:
                    logger.warning(f"Found {extreme_companies} companies with extreme return characteristics")
                    # Don't remove these companies, just log the warning

            return returns_df

        except Exception as e:
            logger.error(f"Error in validate_returns: {e}")
            return returns_df


def create_annual_market_cap_from_monthly(monthly_df):
    """
    Create annual market cap data from monthly data

    Args:
        monthly_df: DataFrame with monthly market cap data

    Returns:
        DataFrame with annual market cap data
    """
    if monthly_df is None or monthly_df.empty:
        logger.error("Empty monthly market cap dataframe provided")
        return None

    logger.info("Creating annual market cap data from monthly data...")

    # Extract ISIN and name columns
    metadata_cols = ['ISIN', 'NAME'] if 'NAME' in monthly_df.columns else ['ISIN', 'Name']
    metadata = monthly_df[metadata_cols].copy()

    # Get date columns (all columns except metadata)
    date_cols = [col for col in monthly_df.columns if col not in metadata_cols]

    # Convert columns to datetime
    valid_dates = []
    valid_cols = []

    for col in date_cols:
        try:
            date = pd.to_datetime(col)
            if pd.notna(date):
                valid_dates.append(date)
                valid_cols.append(col)
        except:
            continue

    # Sort dates chronologically
    sorted_indices = np.argsort(valid_dates)
    valid_dates = [valid_dates[i] for i in sorted_indices]
    valid_cols = [valid_cols[i] for i in sorted_indices]

    # Filter for December dates (end of year)
    annual_dates = []
    annual_cols = []

    for date, col in zip(valid_dates, valid_cols):
        if date.month == 12:
            annual_dates.append(date)
            annual_cols.append(col)

    if not annual_cols:
        logger.warning("No December dates found. Using last month of each year.")

        # Use last month of each year
        years = sorted(set(date.year for date in valid_dates))
        for year in years:
            year_dates = [date for date in valid_dates if date.year == year]
            if year_dates:
                last_date = max(year_dates)
                idx = valid_dates.index(last_date)
                annual_dates.append(last_date)
                annual_cols.append(valid_cols[idx])

    # Create annual data frame with year columns
    annual_df = metadata.copy()

    for col, date in zip(annual_cols, annual_dates):
        # Convert to numeric to handle invalid values
        col_data = pd.to_numeric(monthly_df[col], errors='coerce')
        annual_df[str(date.year)] = col_data

    # Ensure all years from 1999 to 2024 exist
    years_range = list(range(1999, 2025))
    existing_years = [date.year for date in annual_dates]

    # Add missing years through interpolation
    for year in years_range:
        if year not in existing_years:
            # Find closest available years
            prev_year = max([y for y in existing_years if y < year], default=None)
            next_year = min([y for y in existing_years if y > year], default=None)

            if prev_year and next_year:
                # Linear interpolation
                prev_values = pd.to_numeric(annual_df[str(prev_year)], errors='coerce')
                next_values = pd.to_numeric(annual_df[str(next_year)], errors='coerce')
                weight = (year - prev_year) / (next_year - prev_year)
                annual_df[str(year)] = prev_values + (next_values - prev_values) * weight
            elif prev_year:
                # Forward fill
                annual_df[str(year)] = annual_df[str(prev_year)]
            elif next_year:
                # Backward fill
                annual_df[str(year)] = annual_df[str(next_year)]

    # Ensure carbon years (2013-2023) have valid data
    critical_years = list(range(2013, 2024))
    for year in critical_years:
        year_str = str(year)

        if year_str not in annual_df.columns:
            logger.warning(f"Critical year {year} missing from annual data")
            continue

        # Check for and handle missing values
        col_data = pd.to_numeric(annual_df[year_str], errors='coerce')
        if col_data.isna().any():
            # Fill NaNs with column mean for non-zero values
            mean_val = col_data[col_data > 0].mean()
            if pd.isna(mean_val):
                mean_val = 1.0  # Default if no valid values
            annual_df[year_str] = col_data.fillna(mean_val)

        # Check for non-positive values
        if (col_data <= 0).any():
            # Replace with mean of positive values
            mean_val = col_data[col_data > 0].mean()
            if pd.isna(mean_val):
                mean_val = 1.0  # Default if no valid values
            annual_df[year_str] = col_data.where(col_data > 0, mean_val)

    # Validate final result
    carbon_years = [str(y) for y in range(2013, 2024)]
    logger.info(
        f"Created annual market cap data with {len(annual_df)} companies and {len(annual_df.columns) - len(metadata_cols)} years")
    logger.info(
        f"Valid year columns: {sorted([col for col in annual_df.columns if col.isdigit() and 2013 <= int(col) <= 2023])}")

    return annual_df


def fix_standard_asset_allocation(returns_df):
    """
    Fix issues in the Standard_Asset_Allocation.py implementation

    This function corrects common issues:
    1. Ensures date columns are correctly parsed
    2. Handles missing values robustly
    3. Improves the optimization process

    Args:
        returns_df: DataFrame with returns data

    Returns:
        Fixed returns DataFrame
    """
    if returns_df is None or returns_df.empty:
        logger.warning("Empty returns DataFrame provided")
        return returns_df

    try:
        # Ensure returns_df has expected structure
        if 'ISIN' not in returns_df.columns:
            logger.error("Missing ISIN column in returns_df")
            return returns_df

        # Convert date columns to datetime
        date_cols = [col for col in returns_df.columns if col not in ['ISIN', 'NAME', 'Name']]
        valid_dates = []

        for col in date_cols:
            try:
                # Skip columns that can't be parsed as dates
                date = pd.to_datetime(col)
                if pd.notna(date):
                    valid_dates.append(col)
            except:
                continue

        if not valid_dates:
            logger.error("No valid date columns found in returns_df")
            return returns_df

        # Create a copy with only valid date columns
        fixed_df = returns_df[['ISIN'] + (['NAME'] if 'NAME' in returns_df.columns else ['Name']) + valid_dates].copy()

        # Convert returns to numeric, handling invalid values
        for col in valid_dates:
            fixed_df[col] = pd.to_numeric(fixed_df[col], errors='coerce')

        # Handle extreme returns
        for col in valid_dates:
            returns = fixed_df[col]
            extreme_mask = (returns > 1.0) | (returns < -0.9)
            extreme_count = extreme_mask.sum()

            if extreme_count > 0:
                logger.warning(f"Found {extreme_count} extreme returns in column {col}. Winsorizing.")
                # Winsorize extreme returns at 100% and -90%
                fixed_df[col] = returns.where(~extreme_mask, returns.clip(lower=-0.9, upper=1.0))

        # Interpolate missing values for each company (row)
        fixed_df[valid_dates] = fixed_df[valid_dates].interpolate(axis=1, method='linear', limit_direction='both',
                                                                  limit=5)

        logger.info(f"Fixed returns DataFrame with {len(fixed_df)} companies and {len(valid_dates)} time periods")
        return fixed_df

    except Exception as e:
        logger.error(f"Error fixing returns DataFrame: {str(e)}")
        return returns_df


def process_all_data(static_file: str, original_data_folder: str,
                     filtered_data_folder: str, failed_data_folder: str,
                     max_missing: float = 0.2) -> Dict[str, pd.DataFrame]:
    """
    Process all data files with improved error handling and data cleaning
    according to project specifications
    """
    processor = DataProcessor(max_missing)

    try:
        # Create output directories
        for folder in [filtered_data_folder, failed_data_folder]:
            os.makedirs(folder, exist_ok=True)

        # Extract European companies
        european_companies = processor.extract_european_companies(static_file)

        # Define files to process
        files_to_process = {
            'Scope_1.xlsx': 'carbon',
            'Scope_2.xlsx': 'carbon',
            'DS_REV_USD_Y.xlsx': 'revenue',
            'DS_MV_T_USD_Y.xlsx': 'market_cap',
            'DS_MV_T_USD_M.xlsx': 'price',
            'DS_RI_T_USD_Y.xlsx': 'price',
            'DS_RI_T_USD_M.xlsx': 'price'
        }

        # Initialize dictionaries for processed data
        processed_data = {}
        failed_data = {}

        # Track missing files
        missing_files = []

        # Process each file
        for file, data_type in files_to_process.items():
            input_path = os.path.join(original_data_folder, file)
            if not os.path.exists(input_path):
                logger.error(f"Required file not found: {input_path}")
                missing_files.append(file)
                continue

            try:
                logger.info(f"Processing {file}...")

                # Read file
                df = pd.read_excel(input_path)
                if df.empty:
                    logger.warning(f"Empty DataFrame from {file}")
                    continue

                # Standardize column names
                data_format = 'monthly' if '_M.' in file else 'annual'
                df = processor.standardize_date_columns(df, data_format)

                # Validate data structure
                is_valid, valid_columns = processor.validate_data_structure(df, data_type)
                if not is_valid:
                    logger.warning(f"Invalid data structure in {file}")
                    continue

                # Filter for European companies
                filtered_df = df[df['ISIN'].isin(european_companies['ISIN'])]
                if filtered_df.empty:
                    logger.warning(f"No European companies found in {file}")
                    continue

                logger.info(f"Filtered {file} to {len(filtered_df)} European companies")

                # Handle missing data
                valid_df, failed_df = processor.handle_missing_data(filtered_df, data_type)

                processed_data[file] = valid_df
                failed_data[file] = failed_df

                logger.info(f"Processed {file}: {len(valid_df)} valid, {len(failed_df)} failed")

            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue

        # Check required files
        essential_files = ['Scope_1.xlsx', 'Scope_2.xlsx', 'DS_REV_USD_Y.xlsx', 'DS_RI_T_USD_M.xlsx']
        missing_essential = [file for file in essential_files if
                             file not in processed_data or processed_data[file].empty]

        if missing_essential:
            logger.error(f"Missing essential files: {missing_essential}")
            raise DataValidationError(f"Cannot proceed without essential files: {missing_essential}")

        # Calculate simple returns
        price_file = 'DS_RI_T_USD_M.xlsx'
        if price_file in processed_data and not processed_data[price_file].empty:
            price_df = processed_data[price_file]
            returns_df = processor.calculate_simple_returns(price_df)

            if not returns_df.empty:
                # Validate returns
                returns_df = processor.validate_returns(returns_df)
                processed_data["Simple_Returns.csv"] = returns_df
                logger.info(f"Created Simple_Returns.csv with {len(returns_df)} companies")
            else:
                logger.error("Failed to calculate simple returns")
        else:
            logger.error(f"Price data {price_file} not available for returns calculation")

        # Apply fixes to specific issues
        # 1. Fix annual market cap data
        if 'DS_MV_T_USD_M.xlsx' in processed_data and 'DS_MV_T_USD_Y.xlsx' not in processed_data:
            logger.info("Generating annual market cap data from monthly data")
            monthly_df = processed_data['DS_MV_T_USD_M.xlsx']
            annual_df = create_annual_market_cap_from_monthly(monthly_df)

            if annual_df is not None and not annual_df.empty:
                # Save to CSV
                annual_path = os.path.join(filtered_data_folder, "DS_MV_T_USD_Y.csv")
                annual_df.to_csv(annual_path, index=False)
                processed_data['DS_MV_T_USD_Y.csv'] = annual_df
                logger.info(f"Created and saved annual market cap data to {annual_path}")

        # 2. Fix returns data for optimization
        if 'Simple_Returns.csv' in processed_data:
            logger.info("Fixing returns data for optimization")
            returns_df = processed_data['Simple_Returns.csv']
            fixed_returns = fix_standard_asset_allocation(returns_df)

            # Save fixed returns
            if fixed_returns is not None and not fixed_returns.equals(returns_df):
                fixed_path = os.path.join(filtered_data_folder, "Fixed_Returns.csv")
                fixed_returns.to_csv(fixed_path, index=False)
                processed_data['Fixed_Returns.csv'] = fixed_returns
                logger.info(f"Created fixed returns data at {fixed_path}")

        # Identify companies valid across all required datasets
        valid_scope1 = set(processed_data['Scope_1.xlsx']['ISIN']) if 'Scope_1.xlsx' in processed_data else set()
        valid_scope2 = set(processed_data['Scope_2.xlsx']['ISIN']) if 'Scope_2.xlsx' in processed_data else set()
        valid_revenue = set(
            processed_data['DS_REV_USD_Y.xlsx']['ISIN']) if 'DS_REV_USD_Y.xlsx' in processed_data else set()
        valid_returns = set(
            processed_data['Simple_Returns.csv']['ISIN']) if 'Simple_Returns.csv' in processed_data else set()

        # Find common valid ISINs
        valid_isins = valid_scope1.intersection(valid_scope2, valid_revenue)

        # Add returns data if available
        if valid_returns:
            valid_isins = valid_isins.intersection(valid_returns)

        logger.info(f"Initial intersection of valid ISINs across datasets: {len(valid_isins)}")

        # Additional check: ensure companies have sufficient emissions data
        # This follows project specifications for handling missing carbon data
        years = [str(y) for y in range(processor.carbon_start_year, processor.carbon_end_year + 1)]
        min_years = len(years) * 0.8  # 80% threshold

        final_valid_isins = set()
        for isin in valid_isins:
            # Check Scope 1 data completeness
            scope1_data = processed_data['Scope_1.xlsx']
            scope1_rows = scope1_data[scope1_data['ISIN'] == isin]

            if scope1_rows.empty:
                continue

            scope1_row = scope1_rows.iloc[0]

            # Check Scope 2 data completeness
            scope2_data = processed_data['Scope_2.xlsx']
            scope2_rows = scope2_data[scope2_data['ISIN'] == isin]

            if scope2_rows.empty:
                continue

            scope2_row = scope2_rows.iloc[0]

            # Count years with valid data
            valid_years = 0
            for year in years:
                # Check if either Scope 1 or Scope 2 has valid data for this year
                # This is more lenient than requiring both to have data
                scope1_valid = year in scope1_row.index and pd.notna(scope1_row[year])
                scope2_valid = year in scope2_row.index and pd.notna(scope2_row[year])

                if scope1_valid or scope2_valid:
                    valid_years += 1

            if valid_years >= min_years:
                final_valid_isins.add(isin)
            else:
                logger.warning(f"ISIN {isin} has insufficient emissions data ({valid_years}/{len(years)} years)")

        valid_isins = final_valid_isins
        logger.info(f"Companies valid across all datasets with sufficient emissions data: {len(valid_isins)}")

        if len(valid_isins) < 10:  # Arbitrary threshold for a reasonable dataset
            logger.error(
                f"Total {len(valid_isins)} companies valid across all datasets. This is too few for meaningful analysis.")
            raise DataValidationError("Insufficient valid companies after filtering")

        # Create failure report
        failure_report = []
        for isin in european_companies['ISIN']:
            if isin not in valid_isins:
                idx = european_companies['ISIN'].index(isin)
                failure_entry = {
                    'ISIN': isin,
                    'Name': european_companies['Name'][idx],
                    'Country': european_companies['Country'][idx],
                    'Region': european_companies['Region'][idx],
                    'Reason': []
                }

                for file in files_to_process.keys():
                    if file in failed_data and file in failed_data and isin in failed_data[file]['ISIN'].values:
                        failure_entry['Reason'].append(f"{file}: >{max_missing * 100}% missing data")
                    elif file in processed_data and isin not in processed_data[file]['ISIN'].values:
                        failure_entry['Reason'].append(f"{file}: Not found")

                failure_entry['Reason'] = '; '.join(failure_entry['Reason'])
                failure_report.append(failure_entry)

        failure_df = pd.DataFrame(failure_report)
        if not failure_df.empty:
            failure_path = os.path.join(failed_data_folder, "Company_Failure_Report.csv")
            failure_df.to_csv(failure_path, index=False)
            logger.info(f"Created failure report with {len(failure_report)} companies at {failure_path}")

        # Process each file, keeping only valid companies
        final_processed_data = {}

        for file in files_to_process.keys():
            input_path = os.path.join(original_data_folder, file)
            output_path = os.path.join(filtered_data_folder, file.replace('.xlsx', '.csv'))

            if not os.path.exists(input_path):
                logger.warning(f"File not found: {input_path}")
                continue

            try:
                if file not in processed_data or processed_data[file].empty:
                    logger.warning(f"No processed data for {file}")
                    continue

                df = processed_data[file]
                filtered_df = df[df['ISIN'].isin(valid_isins)]

                if filtered_df.empty:
                    logger.warning(f"No valid companies found in {file} after final filtering")
                    continue

                # Save to CSV
                filtered_df.to_csv(output_path, index=False)
                final_processed_data[file.replace('.xlsx', '.csv')] = filtered_df
                logger.info(f"Processed and saved: {output_path} ({len(filtered_df)} rows)")

            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
                continue

        # Ensure Simple_Returns.csv is included in final output
        if 'Simple_Returns.csv' in processed_data and not processed_data['Simple_Returns.csv'].empty:
            returns_df = processed_data['Simple_Returns.csv']
            filtered_returns = returns_df[returns_df['ISIN'].isin(valid_isins)]

            if not filtered_returns.empty:
                returns_path = os.path.join(filtered_data_folder, "Simple_Returns.csv")
                filtered_returns.to_csv(returns_path, index=False)
                final_processed_data["Simple_Returns.csv"] = filtered_returns
                logger.info(f"Saved final Simple_Returns.csv with {len(filtered_returns)} companies")

        # Ensure Fixed_Returns.csv is included if it exists
        if 'Fixed_Returns.csv' in processed_data and not processed_data['Fixed_Returns.csv'].empty:
            fixed_returns_df = processed_data['Fixed_Returns.csv']
            filtered_fixed_returns = fixed_returns_df[fixed_returns_df['ISIN'].isin(valid_isins)]

            if not filtered_fixed_returns.empty:
                fixed_returns_path = os.path.join(filtered_data_folder, "Fixed_Returns.csv")
                filtered_fixed_returns.to_csv(fixed_returns_path, index=False)
                final_processed_data["Fixed_Returns.csv"] = filtered_fixed_returns
                logger.info(f"Saved final Fixed_Returns.csv with {len(filtered_fixed_returns)} companies")

        # Ensure DS_MV_T_USD_Y.csv is included if it was generated
        if 'DS_MV_T_USD_Y.csv' in processed_data and not processed_data['DS_MV_T_USD_Y.csv'].empty:
            market_cap_df = processed_data['DS_MV_T_USD_Y.csv']
            filtered_market_cap = market_cap_df[market_cap_df['ISIN'].isin(valid_isins)]

            if not filtered_market_cap.empty:
                market_cap_path = os.path.join(filtered_data_folder, "DS_MV_T_USD_Y.csv")
                filtered_market_cap.to_csv(market_cap_path, index=False)
                final_processed_data["DS_MV_T_USD_Y.csv"] = filtered_market_cap
                logger.info(f"Saved final DS_MV_T_USD_Y.csv with {len(filtered_market_cap)} companies")

        # Log final data sizes
        logger.info("\nFinal processed data summary:")
        for file, df in final_processed_data.items():
            logger.info(f"  {file}: {len(df)} companies")

        # Final validation check
        if not all(file.replace('.xlsx', '.csv') in final_processed_data for file in essential_files):
            missing = [file.replace('.xlsx', '.csv') for file in essential_files
                       if file.replace('.xlsx', '.csv') not in final_processed_data]
            logger.error(f"Final validation failed: Missing essential files in output: {missing}")
        else:
            logger.info("Final validation passed: All essential files processed successfully")

        return final_processed_data
    except Exception as e:
        logger.error(f"Critical error in data processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def Initializer(static_file, original_data_folder, filtered_data_folder,
                failed_data_folder, max_missing=0.2):
    """Main initialization routine with improved error handling and data validation"""
    try:
        logger.info("=" * 80)
        logger.info("Starting data initialization for Sustainability Aware Asset Management")
        logger.info(f"Max missing data threshold: {max_missing * 100}%")
        logger.info("=" * 80)

        # Process all data files
        processed_data = process_all_data(
            static_file, original_data_folder, filtered_data_folder,
            failed_data_folder, max_missing
        )

        if not processed_data:
            logger.error("Data initialization failed: No processed data returned")
            raise DataValidationError("Data initialization failed")

        # Perform additional validation on returns data
        if "Simple_Returns.csv" in processed_data:
            returns_df = processed_data["Simple_Returns.csv"]

            # Get return columns (excluding metadata)
            metadata_cols = ['ISIN', 'NAME'] if 'NAME' in returns_df.columns else ['ISIN', 'Name']
            return_cols = [col for col in returns_df.columns if col not in metadata_cols]

            if return_cols:
                # Convert to numeric
                returns = returns_df[return_cols].apply(pd.to_numeric, errors='coerce')

                # Check for extreme values
                max_return = returns.max().max()
                min_return = returns.min().min()
                mean_return = returns.mean().mean()
                median_return = returns.median().median()
                std_return = returns.std().mean()

                logger.info(f"Returns validation:")
                logger.info(f"  Max return: {max_return:.4f}")
                logger.info(f"  Min return: {min_return:.4f}")
                logger.info(f"  Mean return: {mean_return:.4f}")
                logger.info(f"  Median return: {median_return:.4f}")
                logger.info(f"  Average std dev: {std_return:.4f}")

                # Warn about potential issues
                if max_return > 1.0:
                    logger.warning(f"Maximum return exceeds 100%: {max_return * 100:.2f}%")
                if min_return < -0.5:
                    logger.warning(f"Minimum return below -50%: {min_return * 100:.2f}%")

        logger.info("=" * 80)
        logger.info("Data initialization complete successfully")
        logger.info("=" * 80)
        return processed_data

    except Exception as e:
        logger.error(f"Initialization failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())

        # Return empty dict as fallback
        return {}