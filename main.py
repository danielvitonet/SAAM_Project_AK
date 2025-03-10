
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

# Set the working directory to the directory where main.py is located.
base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(base_dir)
sys.path.insert(0, base_dir)
print("Current working directory:", os.getcwd())

# Import the necessary functions from the DataHandler package
from DataHandler.Data_SetUP import Initializer
from DataHandler.Standard_Asset_Allocation import run_optimization_routine

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
    
    #if "Simple_Returns.xlsx" in filtered_datasets:
    #    returns_df = filtered_datasets["Simple_Returns.xlsx"]
    #else:
        # Fallback: load from the Data folder.
    #    returns_df = pd.read_excel(os.path.join("Data", "Simple_Returns.xlsx"))
    
    returns_df = filtered_datasets.get("Simple_Returns.xlsx")
    if returns_df is None:
        returns_df = pd.read_excel(os.path.join("Data", "Simple_Returns.xlsx"))

    
    # Define rebalancing dates (for example, the last day of each year from 2013 to 2023).
    rebalancing_dates = ["2013-12-31", "2014-12-31", "2015-12-31", "2016-12-31", 
                         "2017-12-31", "2018-12-31", "2019-12-31", "2020-12-31", 
                         "2021-12-31", "2022-12-31", "2023-12-31"]
    
    # Run the full optimization routine:
    #   It computes optimal weights using a rolling window, calculates ex-post returns for the following 12 months,
    #   and then computes key performance metrics for each rebalancing date.
    results = run_optimization_routine(returns_df, rebalancing_dates, estimation_window=120, horizon=12)
    
    # Output the optimization results.
    for r_date, res in results.items():
        print(f"\nRebalancing Date: {r_date}")
        print("Optimal Weights:")
        print(res["weights"])
        print("Ex-Post Portfolio Returns:")
        print(res["ex_post_returns"])
        print("Performance Metrics:")
        for metric, value in res["performance_metrics"].items():
            print(f"  {metric}: {value}")
    
    print("\nOptimization routine complete. Portfolio characteristics have been computed over the sample.")

    
    
    
    
if __name__ == "__main__":
    main()


###############################################################################
# PART 2: Impact of the portfolio on climate
# Asset Allocation with a Carbon Emissions Reduction.
# Asset Allocation with a Net Zero Objective.
###############################################################################


# Main file
