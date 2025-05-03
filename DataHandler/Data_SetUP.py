import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def extract_european_companies_dict(static_file="Data/Static.xlsx"):
    """
    Extracts European companies from static file
    Returns dictionary with ISIN, Name, Country, Region
    """
    static_df = pd.read_excel(static_file)
    europe_df = static_df[static_df['Region'] == 'EUR']
    return {
        "ISIN": europe_df["ISIN"].tolist(),
        "Name": europe_df["Name"].tolist(),
        "Country": europe_df["Country"].tolist(),
        "Region": europe_df["Region"].tolist()
    }


def handle_missing_data(df, max_missing=0.2):
    """
    Advanced missing data handling with imputation
    Returns cleaned DataFrame and failed companies DataFrame
    """
    # Calculate missing percentage
    missing_percent = df.iloc[:, 2:].isnull().mean(axis=1)

    # Split valid/invalid
    valid_mask = missing_percent <= max_missing
    valid_df = df[valid_mask].copy()
    failed_df = df[~valid_mask].copy()

    # Imputation pipeline
    if not valid_df.empty:
        # Forward/backward fill
        valid_df.iloc[:, 2:] = valid_df.iloc[:, 2:].ffill(axis=1).bfill(axis=1)

        # Linear interpolation
        valid_df.iloc[:, 2:] = valid_df.iloc[:, 2:].interpolate(
            method='linear',
            axis=1,
            limit_direction='both'
        )

        # Final fill with mean
        valid_df.iloc[:, 2:] = valid_df.iloc[:, 2:].apply(
            lambda row: row.fillna(row.mean()),
            axis=1
        )

    return valid_df, failed_df


def filter_european_data(dataset, european_companies):
    """
    Filters dataset to include only European companies
    Handles different naming conventions
    """
    name_col = 'NAME' if 'NAME' in dataset.columns else 'Name'
    static_names_upper = [n.upper() for n in european_companies["Name"]]

    isin_filter = dataset['ISIN'].isin(european_companies["ISIN"])
    name_filter = dataset[name_col].str.upper().isin(static_names_upper)

    return dataset[isin_filter & name_filter]


def filter_companies_with_emissions_data(scope1_file, scope2_file, european_companies,
                                         max_missing=0.2, start_year=2013, end_year=2023):
    """
    Filters companies with complete emissions data
    Returns filtered companies and failure report
    """
    # Load emissions data
    scope1_df = Load(scope1_file)
    scope2_df = Load(scope2_file)

    # Process emissions data
    scope1_clean, scope1_failed = handle_missing_data(scope1_df, max_missing)
    scope2_clean, scope2_failed = handle_missing_data(scope2_df, max_missing)

    # Find valid companies
    valid_isins = set(scope1_clean['ISIN']).intersection(set(scope2_clean['ISIN']))

    # Build failure report
    failure_report = {
        "ISIN": [], "Name": [], "Country": [], "Region": [], "Reason": []
    }

    # Classify failures
    for idx, isin in enumerate(european_companies["ISIN"]):
        name = european_companies["Name"][idx]
        country = european_companies["Country"][idx]
        region = european_companies["Region"][idx]

        if isin in valid_isins:
            continue

        failure_report["ISIN"].append(isin)
        failure_report["Name"].append(name)
        failure_report["Country"].append(country)
        failure_report["Region"].append(region)

        if isin in scope1_failed['ISIN'].values:
            failure_report["Reason"].append("Scope1: Missing >20% data")
        elif isin in scope2_failed['ISIN'].values:
            failure_report["Reason"].append("Scope2: Missing >20% data")
        else:
            failure_report["Reason"].append("No emissions data")

    # Build filtered companies
    filtered_companies = {
        k: [v[idx] for idx, isin in enumerate(european_companies["ISIN"]) if isin in valid_isins]
        for k, v in european_companies.items()
    }

    return filtered_companies, pd.DataFrame(failure_report)


def filter_all_datasets(original_data_folder="Data/Original Data",
                        filtered_data_folder="Data/Filtered Data",
                        failed_data_folder="Data/Failed Companies",
                        european_companies=None,
                        max_missing=0.2):
    """
    Processes all datasets with advanced filtering
    """
    # Create directories if needed
    os.makedirs(filtered_data_folder, exist_ok=True)
    os.makedirs(failed_data_folder, exist_ok=True)

    target_files = [
        "DS_MV_T_USD_M.xlsx", "DS_MV_T_USD_Y.xlsx", "DS_REV_USD_Y.xlsx",
        "DS_RI_T_USD_M.xlsx", "DS_RI_T_USD_Y.xlsx", "Scope_1.xlsx", "Scope_2.xlsx"
    ]

    for file in target_files:
        input_path = os.path.join(original_data_folder, file)
        output_path = os.path.join(filtered_data_folder, file)
        failed_path = os.path.join(failed_data_folder, f"failed_{file}")

        if os.path.exists(output_path):
            continue

        df = Load(input_path)

        # Handle missing data
        cleaned_df, failed_df = handle_missing_data(df, max_missing)
        failed_df.to_excel(failed_path, index=False)

        # Filter European companies
        filtered_df = filter_european_data(cleaned_df, european_companies)

        # Format dates
        time_cols = filtered_df.columns[2:]
        new_cols = []

        if "_M" in file:  # Monthly data
            for col in time_cols:
                try:
                    new_col = pd.to_datetime(col).strftime('%Y-%m-%d')
                except:
                    new_col = col
                new_cols.append(new_col)
        else:  # Yearly data
            new_cols = [str(pd.to_datetime(col).year) if pd.notnull(pd.to_datetime(col)) else col for col in time_cols]

        filtered_df.columns = list(filtered_df.columns[:2]) + new_cols

        # Save results
        if file.endswith('.csv'):
            filtered_df.to_csv(output_path, index=False)
        else:
            filtered_df.to_excel(output_path, index=False)

    return None


def Load(filename, show_head=False):
    """Universal loader for CSV/Excel files"""
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)
    elif filename.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(filename)
    else:
        raise ValueError("Unsupported file format")

    if show_head:
        print(f"\n{filename} head:")
        print(df.head())

    return df


def Data_Analyzer(dataset, produce_visuals=False):
    """
    Enhanced data analyzer with missing data diagnostics
    """
    # Basic stats
    print(f"\nTotal observations: {len(dataset)}")
    print(f"Total missing values: {dataset.isna().sum().sum()}")

    # Time period analysis
    time_cols = dataset.columns[2:]
    try:
        dates = pd.to_datetime(time_cols)
        freq = 'Monthly' if len(dates) > 120 else 'Yearly'
        print(f"Data frequency: {freq}")
        print(f"Date range: {dates.min().date()} - {dates.max().date()}")
    except:
        print("Could not determine date frequency")

    # Numeric analysis
    numeric_data = dataset.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
    print("\nDescriptive statistics:")
    print(numeric_data.describe())

    # Visualization
    if produce_visuals:
        plt.figure(figsize=(12, 6))
        numeric_data.mean().plot(title="Average Values Over Time")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(12, 6))
        numeric_data.stack().hist(bins=50)
        plt.title("Value Distribution")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()


def calculate_simple_returns(filtered_file_path,
                             init_start="2004-01-01",
                             init_end="2013-12-31",
                             max_missing=0.05):
    """
    Calculates returns with enhanced validation
    """
    # Load price data
    df = Load(filtered_file_path)
    meta = df.iloc[:, :2]
    prices = df.iloc[:, 2:]

    # Sort columns chronologically
    dates = pd.to_datetime(prices.columns)
    prices = prices[prices.columns[dates.argsort()]]

    # Calculate returns
    returns = prices.div(prices.shift(axis=1)) - 1
    returns = returns.iloc[:, 1:]  # Remove first column

    # Validate returns quality
    missing_returns = returns.isna().mean(axis=1)
    valid_companies = missing_returns[missing_returns <= max_missing].index
    returns = returns.loc[valid_companies]

    # Split initialization period
    init_cols = returns.columns[
        (pd.to_datetime(returns.columns) >= pd.to_datetime(init_start)) &
        (pd.to_datetime(returns.columns) <= pd.to_datetime(init_end))
        ]

    # Create DataFrames
    full_returns = pd.concat([meta, returns], axis=1)
    init_returns = pd.concat([meta, returns[init_cols]], axis=1)

    # Save results
    full_returns.to_excel("Data/Simple_Returns.xlsx", index=False)
    init_returns.to_excel("Data/Initialization_Returns.xlsx", index=False)

    return full_returns, init_returns


def Initializer(static_file="Data/Static.xlsx",
                original_data_folder="Data/Original Data",
                filtered_data_folder="Data/Filtered Data",
                failed_data_folder="Data/Failed Companies",
                produce_visuals=False,
                max_missing=0.2):
    """
    Main initialization routine with enhanced features
    """
    # Extract European companies
    european_companies = extract_european_companies_dict(static_file)

    # Filter companies with emissions data
    scope1_path = os.path.join(original_data_folder, "Scope_1.xlsx")
    scope2_path = os.path.join(original_data_folder, "Scope_2.xlsx")
    filtered_companies, failure_report = filter_companies_with_emissions_data(
        scope1_path, scope2_path, european_companies, max_missing
    )

    os.makedirs(failed_data_folder, exist_ok=True)

    # Save failure report
    failure_report.to_excel(
        os.path.join(failed_data_folder, "Failure_Report.xlsx"),
        index=False
    )

    # Process all datasets
    filter_all_datasets(
        original_data_folder=original_data_folder,
        filtered_data_folder=filtered_data_folder,
        failed_data_folder=failed_data_folder,
        european_companies=filtered_companies,
        max_missing=max_missing
    )

    # Calculate returns
    price_file = os.path.join(filtered_data_folder, "DS_RI_T_USD_M.xlsx")
    returns, init_returns = calculate_simple_returns(price_file)

    # Load all processed data
    processed_data = {}
    for file in os.listdir(filtered_data_folder):
        if file.endswith(('.xlsx', '.csv')):
            processed_data[file] = Load(os.path.join(filtered_data_folder, file))

    # Add returns data
    processed_data.update({
        "Simple_Returns.xlsx": returns,
        "Initialization_Returns.xlsx": init_returns
    })

    # Analyze datasets
    for name, data in processed_data.items():
        print(f"\nAnalyzing {name}:")
        Data_Analyzer(data, produce_visuals)

    return processed_data