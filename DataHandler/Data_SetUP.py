import os
import pandas as pd
import matplotlib.pyplot as plt

"""
This file handles data loading and building our group Investment datasets (filtered datasets for Europe / Scope 1+2 and returns).
It also runs some data analysis.
"""

def extract_european_companies_dict(static_file="Data/Static.xlsx"):
    """
    Loads the static file, extracts companies from the EUR region,
    and returns a dictionary with keys: 'ISIN', 'Name', 'Country', and 'Region'.
    """
    static_df = pd.read_excel(static_file)
    europe_df = static_df[static_df['Region'] == 'EUR']
    european_companies = {
        "ISIN": europe_df["ISIN"].tolist(),
        "Name": europe_df["Name"].tolist(),
        "Country": europe_df["Country"].tolist(),
        "Region": europe_df["Region"].tolist()
    }
    return european_companies

def filter_european_data(dataset, european_companies):
    """
    Filters a given dataset to include only rows that belong to companies in the EUR region.
    Accounts for the fact that the original data files use "NAME" (uppercase) for the company name,
    while the static file uses "Name".
    """
    if 'Name' in dataset.columns:
        name_col = 'Name'
    elif 'NAME' in dataset.columns:
        name_col = 'NAME'
    else:
        raise ValueError("Dataset must contain a company name column labeled 'Name' or 'NAME'.")
    
    static_names_upper = [n.upper() for n in european_companies["Name"]]
    isin_filtered = dataset['ISIN'].isin(european_companies["ISIN"])
    name_filtered = dataset[name_col].str.upper().isin(static_names_upper)
    
    return dataset[isin_filtered & name_filtered]

def filter_all_datasets(original_data_folder="Data/Original Data", 
                        filtered_data_folder="Data/Filtered Data", 
                        european_companies=None):
    """
    Filters all datasets in the 'Original Data' folder and saves the cleaned versions in 'Filtered Data'.
    The files to filter include:
       DS_MV_T_USD_M.xlsx, DS_MV_T_USD_Y.xlsx, DS_REV_USD_Y.xlsx,
       DS_RI_T_USD_M.xlsx, DS_RI_T_USD_Y.xlsx, Scope_1.xlsx, Scope_2.xlsx.
       
    Additionally, it standardizes the column names of time-indexed data:
      - For monthly data (columns in 'dd/mm/yyyy' format), it converts and reformats them to ISO format (YYYY-MM-DD)
      - For yearly data (columns in 'YYYY' format), it ensures they remain as a simple year string.
    """
    if european_companies is None:
        raise ValueError("European companies dictionary must be provided.")
    
    if not os.path.exists(filtered_data_folder):
        os.makedirs(filtered_data_folder)
    
    target_files = ["DS_MV_T_USD_M.xlsx", "DS_MV_T_USD_Y.xlsx", "DS_REV_USD_Y.xlsx", 
                    "DS_RI_T_USD_M.xlsx", "DS_RI_T_USD_Y.xlsx", "Scope_1.xlsx", "Scope_2.xlsx"]
    
    for file in os.listdir(original_data_folder):
        if file in target_files:
            file_path = os.path.join(original_data_folder, file)
            filtered_file_path = os.path.join(filtered_data_folder, file)
            
            if os.path.isfile(filtered_file_path):
                print(f"File '{filtered_file_path}' already generated.")
                continue
            
            df = Load(file_path)
            if 'ISIN' in df.columns and ('Name' in df.columns or 'NAME' in df.columns):
                filtered_df = filter_european_data(df, european_companies)
                
                # Standardize time-indexed column names (from the 3rd column onward)
                time_cols = list(filtered_df.columns[2:])
                new_time_cols = []
                if "_M" in file:
                    # Convert dd/mm/yyyy to ISO format YYYY-MM-DD
                    for col in time_cols:
                        try:
                            new_col = pd.to_datetime(col, format='%d/%m/%Y').strftime('%Y-%m-%d')
                        except Exception:
                            new_col = col
                        new_time_cols.append(new_col)
                else:
                    # For yearly data, ensure column names are just year strings.
                    for col in time_cols:
                        try:
                            new_col = pd.to_datetime(col, format='%Y').strftime('%Y')
                        except Exception:
                            new_col = col
                        new_time_cols.append(new_col)
                filtered_df.columns = list(filtered_df.columns[:2]) + new_time_cols
                
                if file.endswith(".csv"):
                    filtered_df.to_csv(filtered_file_path, index=False)
                else:
                    filtered_df.to_excel(filtered_file_path, index=False)
                print(f"Filtered data saved: {filtered_file_path}")
    return None

def Load(filename, show_head=False):
    """
    Loads a dataset and optionally prints the first few rows.
    Supports both CSV and Excel files.
    """
    if filename.endswith(".csv"):
        data = pd.read_csv(filename)
    elif filename.endswith(".xlsx"):
        data = pd.read_excel(filename)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")
    
    if show_head:
        print(data.head())
    
    return data

def Data_Analyzer(dataset, produce_visuals=False):
    """
    Analyzes the dataset by printing the total number of observations, missing values, determining the periodicity of the time-indexed data,
    and printing descriptive statistics. Optionally, it produces graphs showing key information about moments,
    distribution, and evolution through time.
    
    Expected structure: first two columns are metadata and the remaining columns are time periods.
    """
    total_obs = dataset.shape[0]
    print(f"Total number of observations (rows): {total_obs}")
    
    missing_values = dataset.isnull().sum().sum()
    print(f"Missing values: {missing_values}")
    
    time_cols = list(dataset.columns[2:])
    
    time_dates_month = pd.to_datetime(time_cols, format='%Y-%m-%d', errors='coerce')
    if time_dates_month.isnull().sum() == 0:
        frequency = "Monthly"
        start_date = time_dates_month.min().strftime("%Y-%m-%d")
        end_date = time_dates_month.max().strftime("%Y-%m-%d")
    else:
        time_dates_year = pd.to_datetime(time_cols, format='%Y', errors='coerce')
        if time_dates_year.isnull().sum() == 0:
            frequency = "Yearly"
            start_date = str(time_dates_year.min().year)
            end_date = str(time_dates_year.max().year)
        else:
            frequency = "Unknown"
            start_date = "Unknown"
            end_date = "Unknown"
    
    print(f"Data periodicity: {frequency}")
    print(f"Time window: {start_date} to {end_date}")
    
    numeric_data = dataset.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
    desc_stats = numeric_data.describe()
    print("\nDescriptive Statistics (aggregated over companies for each time period):")
    print(desc_stats)
    
    if produce_visuals:
        avg_time_series = numeric_data.mean(axis=0)
        if frequency == "Monthly":
            x_values = pd.to_datetime(time_cols, format='%Y-%m-%d')
        elif frequency == "Yearly":
            x_values = [int(col) for col in time_cols]
        else:
            x_values = list(range(len(time_cols)))
        
        plt.figure(figsize=(10, 5))
        plt.plot(x_values, avg_time_series, marker='o')
        plt.title("Evolution of Average Values Over Time")
        plt.xlabel("Time")
        plt.ylabel("Average Value")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(10, 5))
        plt.hist(numeric_data.values.flatten(), bins=30, edgecolor='black')
        plt.title("Histogram of All Time Series Values")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def calculate_simple_returns(filtered_file_path, init_start="2004-01-01", init_end="2013-12-31"):
    """
    Calculates simple returns from monthly price data.
    
    The function loads the filtered price data from DS_RI_T_USD_M.xlsx (which is expected to have:
      - The first two columns as metadata (e.g., 'Name' and 'ISIN')
      - Subsequent columns corresponding to monthly prices in ISO format (YYYY-MM-DD)
    ).
    
    It then:
      1. Checks if the output files (for full returns and initialization returns) exist in the Data folder.
         If they do, the function loads and returns them immediately.
      2. Otherwise, it sorts the time-indexed columns in chronological order.
      3. Computes simple returns as: Return_t = (Price_t / Price_{t-1}) - 1 for each firm,
         and then drops the first return column (which is NaN).
      4. Extracts the initialization period data (from init_start to init_end).
      5. Saves the new DataFrames to the Data folder.
    """
    import os
    import pandas as pd
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(base_dir, "..", "Data")
    
    simple_returns_file = os.path.join(data_folder, "Simple_Returns.xlsx")
    init_returns_file = os.path.join(data_folder, "Initialization_Returns.xlsx")
    
    if os.path.exists(simple_returns_file) and os.path.exists(init_returns_file):
        print(f"Output files already exist in '{data_folder}'. Loading them...")
        returns_df = pd.read_excel(simple_returns_file)
        init_returns_df = pd.read_excel(init_returns_file)
        return returns_df, init_returns_df
    
    df = pd.read_excel(filtered_file_path)
    meta = df.iloc[:, :2]
    price_data = df.iloc[:, 2:]
    
    try:
        time_cols = pd.to_datetime(price_data.columns, format='%Y-%m-%d', errors='raise')
    except Exception as e:
        raise ValueError("Time-indexed columns must be in ISO format (YYYY-MM-DD).") from e
    
    sorted_idx = time_cols.argsort()
    sorted_time_cols = price_data.columns[sorted_idx]
    price_data = price_data[sorted_time_cols]
    
    # Calculate simple returns: first column will be NaN. Drop that column.
    returns_data = price_data.div(price_data.shift(axis=1)) - 1
    returns_data = returns_data.iloc[:, 1:]  # eliminate the first column since it's NaN
    
    # Combine metadata with the returns data.
    returns_df = pd.concat([meta, returns_data], axis=1)
    
    # Update dt_cols to reflect the new returns columns (after dropping the first time period).
    dt_cols = pd.to_datetime(returns_data.columns, format='%Y-%m-%d')
    init_mask = (dt_cols >= pd.to_datetime(init_start)) & (dt_cols <= pd.to_datetime(init_end))
    init_time_cols = returns_data.columns[init_mask]
    init_returns_df = pd.concat([meta, returns_data[init_time_cols]], axis=1)
    
    print("Simple returns calculated.")
    print(f"Initialization period returns cover {len(init_time_cols)} months from {init_start} to {init_end}.")
    
    returns_df.to_excel(simple_returns_file, index=False)
    print(f"Simple returns saved to: {simple_returns_file}")
    init_returns_df.to_excel(init_returns_file, index=False)
    print(f"Initialization returns saved to: {init_returns_file}")
    
    return returns_df, init_returns_df

def filter_companies_with_emissions_data(scope1_file, scope2_file, european_companies):
    """
    Filters companies to include only those that have both Scope 1 and 2 emissions data.
    Returns a dictionary with the same structure as european_companies but only for companies
    with complete emissions data.
    """
    # Load emissions data
    scope1_df = Load(scope1_file)
    scope2_df = Load(scope2_file)
    
    # Get companies with non-null emissions data
    scope1_companies = scope1_df[scope1_df.iloc[:, 2:].notna().any(axis=1)]['ISIN'].tolist()
    scope2_companies = scope2_df[scope2_df.iloc[:, 2:].notna().any(axis=1)]['ISIN'].tolist()
    
    # Find intersection of companies with both types of emissions data
    companies_with_emissions = list(set(scope1_companies) & set(scope2_companies))
    
    # Filter european_companies dictionary to include only companies with emissions data
    filtered_companies = {
        "ISIN": [],
        "Name": [],
        "Country": [],
        "Region": []
    }
    
    for i, isin in enumerate(european_companies["ISIN"]):
        if isin in companies_with_emissions:
            filtered_companies["ISIN"].append(isin)
            filtered_companies["Name"].append(european_companies["Name"][i])
            filtered_companies["Country"].append(european_companies["Country"][i])
            filtered_companies["Region"].append(european_companies["Region"][i])
    
    return filtered_companies

def Initializer(static_file="Data/Static.xlsx",   
                original_data_folder="Data/Original Data", 
                filtered_data_folder="Data/Filtered Data", 
                produce_visuals=False):
    """
    Main initialization function that:
    1. Extracts European companies from the static file
    2. Filters companies to include only those with Scope 1 and 2 emissions data
    3. Filters all datasets to include only these companies
    4. Calculates simple returns
    5. Returns a dictionary containing all filtered datasets and returns
    """
    # Extract European companies
    european_companies = extract_european_companies_dict(static_file)
    
    # Filter for companies with emissions data
    scope1_file = os.path.join(original_data_folder, "Scope_1.xlsx")
    scope2_file = os.path.join(original_data_folder, "Scope_2.xlsx")
    filtered_companies = filter_companies_with_emissions_data(scope1_file, scope2_file, european_companies)
    
    print(f"\nTotal European companies: {len(european_companies['ISIN'])}")
    print(f"Companies with Scope 1 and 2 data: {len(filtered_companies['ISIN'])}")
    
    # Filter all datasets using the companies with emissions data
    filter_all_datasets(original_data_folder, filtered_data_folder, filtered_companies)
    
    # Calculate simple returns
    filtered_file_path = os.path.join(filtered_data_folder, "DS_RI_T_USD_M.xlsx")
    returns_df, init_returns_df = calculate_simple_returns(filtered_file_path)
    
    # Load all filtered datasets
    filtered_datasets = {}
    for file in os.listdir(filtered_data_folder):
        if file.endswith(('.xlsx', '.csv')):
            file_path = os.path.join(filtered_data_folder, file)
            filtered_datasets[file] = Load(file_path)
    
    # Add returns datasets to the dictionary
    filtered_datasets["Simple_Returns.xlsx"] = returns_df
    filtered_datasets["Initialization_Returns.xlsx"] = init_returns_df
    
    # Analyze each dataset
    for file, dataset in filtered_datasets.items():
        print(f"\nAnalyzing {file}:")
        Data_Analyzer(dataset, produce_visuals)
    
    return filtered_datasets
