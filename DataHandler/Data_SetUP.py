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

            # Validate region values
            invalid_regions = static_df[~static_df['Region'].isin(self.valid_regions)]['Region'].unique()
            if len(invalid_regions) > 0:
                logger.warning(f"Found invalid regions: {invalid_regions}")

            # Filter European companies
            europe_df = static_df[static_df['Region'] == 'EUR'].copy()

            if europe_df.empty:
                raise DataValidationError("No European companies found in static file")

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

    def validate_data_structure(self, df: pd.DataFrame, expected_type: str) -> None:
        """
        Validates the structure of input data based on expected type

        Args:
            df: DataFrame to validate
            expected_type: Type of data ('carbon', 'market_cap', 'revenue', 'price')
        """
        if expected_type == 'carbon':
            # For carbon data, check if we have years in the expected range
            year_cols = []
            for col in df.columns:
                try:
                    if isinstance(col, (int, float)):
                        year = int(col)
                        if self.carbon_start_year <= year <= self.carbon_end_year:
                            year_cols.append(str(year))
                    elif isinstance(col, str) and col.isdigit():
                        year = int(col)
                        if self.carbon_start_year <= year <= self.carbon_end_year:
                            year_cols.append(col)
                except:
                    continue

            if not year_cols:
                logger.warning(f"No carbon data found for years {self.carbon_start_year}-{self.carbon_end_year}")
            else:
                logger.info(f"Found carbon data for years: {sorted(year_cols)}")

        elif expected_type == 'market_cap' or expected_type == 'price':
            # Check for date columns within the specified range
            date_cols = [col for col in df.columns if col not in ['ISIN', 'NAME', 'Name']]
            date_cols = [col for col in date_cols if self.price_start_date <= col <= self.price_end_date]
            if not date_cols:
                raise DataValidationError(f"No date columns found in {expected_type} data for 2004-2023")

        elif expected_type == 'revenue':
            # Check for annual revenue data
            year_cols = []
            for col in df.columns:
                try:
                    if isinstance(col, (int, float)):
                        year_cols.append(str(int(col)))
                    elif isinstance(col, str) and col.isdigit() and len(col) == 4:
                        year_cols.append(col)
                except:
                    continue

            if len(year_cols) < 10:  # Expect at least 10 years of data
                logger.warning(f"Limited revenue data: only {len(year_cols)} years")
            else:
                logger.info(f"Found revenue data for {len(year_cols)} years")

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
            if data_type in ['carbon', 'revenue']:
                # For annual data (carbon, revenue), use previous year's value
                year_cols = []
                available_cols = []

                for col in df.columns:
                    try:
                        if isinstance(col, (int, float)):
                            year = int(col)
                        else:
                            year = int(str(col))
                        if start_year <= year <= end_year:
                            year_cols.append(str(year))
                            available_cols.append(col)
                    except:
                        continue

                if not available_cols:
                    logger.warning("No valid year columns found for missing data analysis")
                    return df, df[0:0]

                data_subset = df[available_cols].copy()
                for col in available_cols:
                    data_subset[col] = pd.to_numeric(data_subset[col], errors='coerce')

                missing_count = data_subset.isnull().sum(axis=1)
                total_count = len(available_cols)

                if total_count == 0:
                    return df, df[0:0]

                missing_percent = missing_count / total_count

                logger.info(f"Missing data statistics for {len(df)} companies:")
                logger.info(f"  Companies with 0% missing: {(missing_percent == 0).sum()}")
                logger.info(f"  Companies with >0% and <={max_missing * 100}% missing: "
                            f"{((missing_percent > 0) & (missing_percent <= max_missing)).sum()}")
                logger.info(f"  Companies with >{max_missing * 100}% missing: {(missing_percent > max_missing).sum()}")

                valid_mask = missing_percent <= max_missing
                valid_df = df[valid_mask].copy()
                failed_df = df[~valid_mask].copy()

                for idx in valid_df.index:
                    for i, col in enumerate(available_cols):
                        if pd.isna(valid_df.loc[idx, col]):
                            if i > 0:
                                prev_value = valid_df.loc[idx, available_cols[i - 1]]
                                if pd.notna(prev_value):
                                    valid_df.loc[idx, col] = prev_value
                                else:
                                    for j in range(i - 1, -1, -1):
                                        prev_val = valid_df.loc[idx, available_cols[j]]
                                        if pd.notna(prev_val):
                                            valid_df.loc[idx, col] = prev_val
                                            break
                            # First year missing: leave as NaN (company not investable)

                return valid_df, failed_df

            elif data_type == 'price':
                # For monthly price data, interpolate missing values
                date_cols = [col for col in df.columns if col not in ['ISIN', 'NAME', 'Name']]
                date_cols = [col for col in date_cols if self.price_start_date <= col <= self.price_end_date]

                if not date_cols:
                    logger.warning("No valid date columns found for price data in 2004-2023")
                    return df, df[0:0]

                data_subset = df[date_cols].copy()
                for col in date_cols:
                    data_subset[col] = pd.to_numeric(data_subset[col], errors='coerce')

                missing_count = data_subset.isnull().sum(axis=1)
                total_count = len(date_cols)

                if total_count == 0:
                    return df, df[0:0]

                missing_percent = missing_count / total_count

                logger.info(f"Missing price data statistics for {len(df)} companies:")
                logger.info(f"  Companies with 0% missing: {(missing_percent == 0).sum()}")
                logger.info(f"  Companies with >0% and <={max_missing * 100}% missing: "
                            f"{((missing_percent > 0) & (missing_percent <= max_missing)).sum()}")
                logger.info(f"  Companies with >{max_missing * 100}% missing: {(missing_percent > max_missing).sum()}")

                valid_mask = missing_percent <= max_missing
                valid_df = df[valid_mask].copy()
                failed_df = df[~valid_mask].copy()

                # Interpolate missing values for valid companies
                valid_df[date_cols] = valid_df[date_cols].interpolate(method='linear', axis=1, limit_direction='both')

                return valid_df, failed_df

        except Exception as e:
            logger.error(f"Error in handle_missing_data: {e}")
            raise

    def standardize_date_columns(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Standardize date columns with robust error handling

        Args:
            df: DataFrame to standardize
            data_type: Type of data ('monthly' or 'annual')
        """
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
                                    new_columns.append(col)
                            else:
                                new_columns.append(col)
                        else:
                            new_columns.append(str(col))
                    except:
                        logger.warning(f"Could not parse column '{col}', keeping as string")
                        new_columns.append(str(col))

            df.columns = new_columns
            return df

        except Exception as e:
            logger.error(f"Error in standardize_date_columns: {e}")
            raise

    def calculate_simple_returns(self, price_df: pd.DataFrame) -> pd.DataFrame:
        try:
            metadata = price_df.iloc[:, :2]
            prices = price_df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')

            # Handle non-positive prices
            negative_prices = (prices <= 0).sum().sum()
            if negative_prices > 0:
                logger.warning(f"Found {negative_prices} non-positive prices. Treated as NaN.")
                prices[prices <= 0] = np.nan

            # Calculate returns
            returns = prices.pct_change(axis=1, fill_method=None)

            # Handle extreme returns (>100% or <-90%)
            extreme_returns = ((returns > 1.0) | (returns < -0.9)).sum().sum()
            if extreme_returns > 0:
                logger.warning(f"Found {extreme_returns} extreme returns (>100% or <-90%). Treated as NaN.")
                returns = returns.where((returns <= 1.0) & (returns >= -0.9), np.nan)

            # Interpolate missing returns
            returns = returns.interpolate(method='linear', axis=1, limit_direction='both')

            returns = returns.iloc[:, 1:]
            returns_df = pd.concat([metadata, returns], axis=1)
            return returns_df
        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            raise

def process_all_data(static_file: str, original_data_folder: str,
                     filtered_data_folder: str, failed_data_folder: str,
                     max_missing: float = 0.2) -> Dict[str, pd.DataFrame]:
    processor = DataProcessor(max_missing)

    try:
        for folder in [filtered_data_folder, failed_data_folder]:
            os.makedirs(folder, exist_ok=True)

        european_companies = processor.extract_european_companies(static_file)
        logger.info(f"Found {len(european_companies['ISIN'])} European companies")

        files_to_process = {
            'Scope_1.xlsx': 'carbon',
            'Scope_2.xlsx': 'carbon',
            'DS_REV_USD_Y.xlsx': 'revenue'
        }

        processed_data = {}
        failed_data = {}

        for file, data_type in files_to_process.items():
            input_path = os.path.join(original_data_folder, file)
            if not os.path.exists(input_path):
                logger.error(f"Required file not found: {input_path}")
                continue

            try:
                df = pd.read_excel(input_path)
                df = processor.standardize_date_columns(df, 'annual')
                processor.validate_data_structure(df, data_type)
                filtered_df = df[df['ISIN'].isin(european_companies['ISIN'])]
                valid_df, failed_df = processor.handle_missing_data(filtered_df, data_type)
                processed_data[file] = valid_df
                failed_data[file] = failed_df
                logger.info(f"Processed {file}: {len(valid_df)} valid, {len(failed_df)} failed")
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
                continue

        # Process price data with specific handling
        price_file = os.path.join(original_data_folder, "DS_RI_T_USD_M.xlsx")
        if os.path.exists(price_file):
            price_df = pd.read_excel(price_file)
            price_df = processor.standardize_date_columns(price_df, 'monthly')
            processor.validate_data_structure(price_df, 'price')
            filtered_price_df = price_df[price_df['ISIN'].isin(european_companies['ISIN'])]
            valid_price_df, failed_price_df = processor.handle_missing_data(filtered_price_df, 'price')
            returns_df = processor.calculate_simple_returns(valid_price_df)
            processed_data["Simple_Returns.csv"] = returns_df
            if not failed_price_df.empty:
                logger.warning(f"Found invalid price data for {len(failed_price_df)} companies.")

        # Identify companies valid across all required datasets
        if all(file in processed_data for file in files_to_process.keys()):
            valid_isins = set(processed_data['Scope_1.xlsx']['ISIN'])
            valid_isins &= set(processed_data['Scope_2.xlsx']['ISIN'])
            valid_isins &= set(processed_data['DS_REV_USD_Y.xlsx']['ISIN'])

            # Additional check: ensure companies have emissions data for at least 80% of years
            years = [str(y) for y in range(2013, 2024)]
            min_years = len(years) * 0.8
            final_valid_isins = set()
            for isin in valid_isins:
                scope1_data = processed_data['Scope_1.xlsx']
                scope2_data = processed_data['Scope_2.xlsx']
                valid_years = sum(
                    1 for y in years
                    if
                    isin in scope1_data['ISIN'].values and pd.notna(scope1_data[scope1_data['ISIN'] == isin][y]).any()
                    and isin in scope2_data['ISIN'].values and pd.notna(
                        scope2_data[scope2_data['ISIN'] == isin][y]).any()
                )
                if valid_years >= min_years:
                    final_valid_isins.add(isin)
                else:
                    logger.warning(f"ISIN {isin} has insufficient emissions data ({valid_years}/{len(years)} years)")

            valid_isins = final_valid_isins
            logger.info(f"Companies valid across all datasets with sufficient emissions data: {len(valid_isins)}")
        else:
            logger.error("Could not process all required files")
            return {}

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
                    if file in failed_data and isin in failed_data[file]['ISIN'].values:
                        failure_entry['Reason'].append(f"{file}: >20% missing data")
                    elif file in processed_data and isin not in processed_data[file]['ISIN'].values:
                        failure_entry['Reason'].append(f"{file}: Not found")

                failure_entry['Reason'] = '; '.join(failure_entry['Reason'])
                failure_report.append(failure_entry)

        failure_df = pd.DataFrame(failure_report)
        failure_df.to_csv(os.path.join(failed_data_folder, "Company_Failure_Report.csv"), index=False)
        logger.info(f"Created failure report with {len(failure_report)} companies")

        # Process remaining files with only valid companies
        all_files = [
            "DS_MV_T_USD_M.xlsx", "DS_MV_T_USD_Y.xlsx", "DS_RI_T_USD_M.xlsx",
            "DS_RI_T_USD_Y.xlsx", "Scope_1.xlsx", "Scope_2.xlsx", "DS_REV_USD_Y.xlsx"
        ]

        final_processed_data = {}

        for file in all_files:
            input_path = os.path.join(original_data_folder, file)
            output_path = os.path.join(filtered_data_folder, file.replace('.xlsx', '.csv'))

            if not os.path.exists(input_path):
                logger.warning(f"File not found: {input_path}")
                continue

            try:
                df = pd.read_excel(input_path)
                filtered_df = df[df['ISIN'].isin(valid_isins)]

                if 'USD_M' in file:
                    data_type = 'monthly'
                elif 'USD_Y' in file or 'Scope' in file:
                    data_type = 'annual'
                else:
                    data_type = 'annual'

                filtered_df = processor.standardize_date_columns(filtered_df, data_type)
                filtered_df.to_csv(output_path, index=False)
                final_processed_data[file.replace('.xlsx', '.csv')] = filtered_df

                logger.info(f"Processed and saved: {output_path} ({len(filtered_df)} rows)")

            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
                continue

        # Calculate returns if price data is available
        price_file = os.path.join(filtered_data_folder, "DS_RI_T_USD_M.csv")
        if os.path.exists(price_file):
            try:
                price_df = pd.read_csv(price_file)
                returns_df = processor.calculate_simple_returns(price_df)
                returns_path = os.path.join(filtered_data_folder, "Simple_Returns.csv")
                returns_df.to_csv(returns_path, index=False)
                final_processed_data["Simple_Returns.csv"] = returns_df
                logger.info("Simple returns calculated and saved")
            except Exception as e:
                logger.error(f"Error calculating returns: {e}")

        return final_processed_data
    except Exception as e:
        logger.error(f"Critical error in data processing: {e}")
        raise

def Initializer(static_file, original_data_folder, filtered_data_folder,
                failed_data_folder, max_missing=0.2):
    """Main initialization routine with improved error handling"""
    try:
        logger.info("Starting data initialization...")
        processed_data = process_all_data(
            static_file, original_data_folder, filtered_data_folder,
            failed_data_folder, max_missing
        )
        logger.info("Data initialization complete")
        return processed_data
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        raise