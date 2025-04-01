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
from DataHandler.Standard_Asset_Allocation import run_portfolio_optimization

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
    
    # Run the portfolio optimization
    metrics, ex_post_returns = run_portfolio_optimization(returns_df)
    
    # Print results
    print("\nPortfolio Characteristics (P(mv)oos):")
    print(f"Annualized Average Return (μ̄p): {metrics['annualized_return']:.4f}")
    print(f"Annualized Volatility (σp): {metrics['annualized_volatility']:.4f}")
    print(f"Average Risk-free Rate: {metrics['avg_rf_rate']:.4f}")
    print(f"Sharpe Ratio (SRp): {metrics['sharpe_ratio']:.4f}")
    print(f"Minimum Return: {metrics['min_return']:.4f}")
    print(f"Maximum Return: {metrics['max_return']:.4f}")
    
    print("\nOptimization routine complete. Portfolio characteristics have been computed over the sample.")

    
    
    
    
if __name__ == "__main__":
    main()


###############################################################################
# PART 2: Impact of the portfolio on climate
# Asset Allocation with a Carbon Emissions Reduction.
# Asset Allocation with a Net Zero Objective.
###############################################################################


# Main file
