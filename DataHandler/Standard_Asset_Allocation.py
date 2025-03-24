import os
import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def calculate_expected_returns(returns_df, window=120):
    """
    Calculate the expected monthly returns for each firm as the arithmetic average
    of the monthly returns over the last 'window' months.
    
    Parameters:
        returns_df (pd.DataFrame): DataFrame containing monthly returns data for each firm.
                                   Expected structure: first two columns are metadata (e.g., 'Name', 'ISIN')
                                   and the remaining columns (sorted chronologically) are monthly returns.
        window (int): Number of months to use for the calculation (default is 120).
    
    Returns:
        expected_returns (pd.Series): Series containing the computed expected return for each firm.
    """
    numeric_data = returns_df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
    if numeric_data.shape[1] < window:
        window = numeric_data.shape[1]
    expected_returns = numeric_data.iloc[:, -window:].mean(axis=1)
    return expected_returns

def handle_missing_returns(returns_df):
    """
    Clean returns data by handling missing values and removing problematic assets.
    """
    # Get numeric data only (skip metadata columns)
    numeric_data = returns_df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
    
    # Remove assets with >50% missing values
    missing_pct = numeric_data.isna().mean(axis=1)
    valid_assets = missing_pct <= 0.5
    numeric_data = numeric_data[valid_assets]
    
    # Fill missing values and handle infinities
    numeric_data = numeric_data.ffill().bfill()
    numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan).fillna(numeric_data.mean())
    
    # Create cleaned DataFrame
    cleaned_returns = returns_df[valid_assets].copy()
    cleaned_returns.iloc[:, 2:] = numeric_data
    
    return cleaned_returns

def compute_covariance_matrix(returns_df, window=60):
    """
    Compute a stable covariance matrix using a rolling window.
    """
    numeric_data = returns_df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
    window_data = numeric_data.iloc[:, -window:]
    
    # Handle any remaining NaN or inf values
    window_data = window_data.replace([np.inf, -np.inf], np.nan)
    window_data = window_data.fillna(window_data.mean())
    
    # Standardize data to improve numerical stability
    scaler = (window_data.std() + 1e-8)  # Add small constant to avoid division by zero
    standardized_data = window_data / scaler
    
    # Compute covariance matrix
    cov_matrix = standardized_data.cov()
    
    # Scale back to original units
    cov_matrix = cov_matrix * scaler.values.reshape(-1, 1) * scaler.values.reshape(1, -1)
    
    # Ensure positive definiteness
    cov_matrix = (cov_matrix + cov_matrix.T) / 2
    min_eig = np.linalg.eigvalsh(cov_matrix)[0]
    if min_eig < 1e-6:  # Increased threshold for better stability
        reg = abs(min_eig) + 1e-6
        cov_matrix += reg * np.eye(cov_matrix.shape[0])
    
    return cov_matrix

def optimize_portfolio(cov_matrix):
    """
    Optimize portfolio weights to minimize variance.
    """
    n_assets = cov_matrix.shape[0]
    
    try:
        # Define optimization problem
        w = cp.Variable(n_assets)
        risk = cp.quad_form(w, cov_matrix)
        
        # Objective: minimize variance
        objective = cp.Minimize(risk)
        
        # Constraints: sum of weights = 1, all weights >= 0
        constraints = [cp.sum(w) == 1, w >= 0]
        
        # Solve using ECOS solver
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        
        if prob.status == "optimal":
            weights = w.value
            weights[np.abs(weights) < 1e-4] = 0
            weights = weights / np.sum(weights)  # Renormalize
            return weights
        else:
            raise ValueError("Optimization did not converge")
            
    except Exception as e:
        print(f"Optimization error: {str(e)}")
        # Fallback to equal weights
        return np.ones(n_assets) / n_assets

def compute_portfolio_metrics(returns, rf_rates):
    """
    Compute annualized portfolio metrics.
    """
    ann_factor = 12  # Monthly to annual conversion
    
    # Handle any NaN values
    returns = returns.fillna(0)
    rf_rates = rf_rates.fillna(0)
    
    # Compute annualized metrics
    mean_return = returns.mean() * ann_factor
    volatility = returns.std() * np.sqrt(ann_factor)
    avg_rf_rate = rf_rates.mean() * ann_factor
    
    # Compute Sharpe ratio
    excess_return = mean_return - avg_rf_rate
    sharpe_ratio = excess_return / volatility if volatility > 1e-8 else 0
    
    return {
        'annualized_return': mean_return,
        'annualized_volatility': volatility,
        'avg_rf_rate': avg_rf_rate,
        'sharpe_ratio': sharpe_ratio,
        'min_return': returns.min(),
        'max_return': returns.max()
    }

def run_portfolio_optimization(returns_df):
    """
    Run the complete portfolio optimization process with annual rebalancing.
    """
    # Clean returns data
    returns_df = handle_missing_returns(returns_df)
    print(f"\nShape after cleaning: {returns_df.shape}")
    
    # Get numeric data and dates
    numeric_data = returns_df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
    time_cols = returns_df.columns[2:]
    dt_cols = pd.to_datetime(time_cols)
    
    # Read risk-free rates
    base_dir = os.path.dirname(os.path.abspath(__file__))
    rf_file = os.path.join(base_dir, "..", "Data", "Risk_Free_Rate.xlsx")
    
    try:
        # Read and process risk-free rates
        rf_data = pd.read_excel(rf_file)
        dates = pd.to_datetime(rf_data.iloc[:, 0].astype(str).str.pad(6, fillchar='0'), format='%Y%m')
        rates = pd.to_numeric(rf_data.iloc[:, 1], errors='coerce') / 100.0
        
        # Create clean risk-free rates series
        rf_df = pd.DataFrame({'date': dates, 'rate': rates}).dropna()
        rf_df = rf_df.groupby('date')['rate'].mean().reset_index()
        rf_rates = pd.Series(rf_df['rate'].values, index=rf_df['date'])
        
        # Align with returns dates
        aligned_rates = []
        for date in dt_cols:
            mask = rf_rates.index <= date
            rate = rf_rates[mask].iloc[-1] if mask.any() else rf_rates.iloc[0]
            aligned_rates.append(rate)
        
        rf_rates = pd.Series(aligned_rates, index=dt_cols)
        
    except Exception as e:
        print(f"\nError reading risk-free rates: {str(e)}")
        rf_rates = pd.Series(0, index=time_cols)
    
    # Initialize results
    weights_dict = {}
    ex_post_returns = pd.Series(index=time_cols, dtype=float)
    
    # Process each year from 2013 to 2023
    for year in range(2013, 2024):
        print(f"\nProcessing year: {year}")
        
        # Find rebalancing date
        rebalancing_date = pd.Timestamp(f"{year}-12-31")
        closest_date = dt_cols[dt_cols <= rebalancing_date].max()
        
        if closest_date is None:
            continue
            
        # Get window data
        t = time_cols.get_loc(closest_date.strftime('%Y-%m-%d'))
        window_size = 60
        if t < window_size:
            continue
        
        window_cols = time_cols[t-window_size:t]
        window_data = numeric_data[window_cols]
        
        # Handle any NaN or inf values in window data
        window_data = window_data.replace([np.inf, -np.inf], np.nan)
        window_data = window_data.fillna(window_data.mean())
        
        # Optimize portfolio
        cov_matrix = compute_covariance_matrix(pd.DataFrame(window_data), window=window_size)
        weights = optimize_portfolio(cov_matrix)
        weights_dict[closest_date.strftime('%Y-%m-%d')] = weights
        
        # Print portfolio statistics
        print(f"Number of non-zero weights: {np.sum(weights > 0)}")
        print(f"Maximum weight: {np.max(weights):.4f}")
        print(f"Minimum non-zero weight: {np.min(weights[weights > 0]):.4f}")
        
        # Plot the covariance matrix and asset allocation for this year
        plot_covariance_matrix(cov_matrix, f"Covariance Matrix (Dec {year})")
        plot_asset_allocation(weights, f"Asset Allocation (Dec {year})")
        
        # Compute ex-post returns for the next year
        if t + 1 < len(time_cols):
            next_cols = time_cols[t+1:t+13]  # Get next 12 months
            if len(next_cols) > 0:
                next_returns = numeric_data[next_cols].values
                current_weights = weights.copy()
                
                for i in range(len(next_cols)):
                    month_returns = next_returns[:, i]
                    # Handle any NaN or inf values
                    month_returns = np.nan_to_num(month_returns, 0)
                    
                    # Ensure weights and returns have the same shape
                    if len(current_weights) != len(month_returns):
                        current_weights = np.pad(current_weights, (0, len(month_returns) - len(current_weights)))
                    
                    portfolio_return = np.sum(current_weights * month_returns)
                    ex_post_returns[next_cols[i]] = portfolio_return
                    
                    if i < len(next_cols) - 1:
                        # Update weights for next period
                        current_weights = current_weights * (1 + month_returns)
                        total_value = np.sum(current_weights)
                        if total_value > 0:
                            current_weights = current_weights / total_value
    
    # Compute final metrics
    metrics = compute_portfolio_metrics(ex_post_returns.dropna(), rf_rates[ex_post_returns.index])
    
    return metrics, ex_post_returns

def plot_covariance_matrix(cov_matrix, title="Covariance Matrix"):
    """
    Plot the covariance matrix as a heatmap with improved visualization.
    """
    plt.figure(figsize=(12, 10))
    
    # Scale the covariance matrix for better visualization
    scaled_cov = cov_matrix.copy()
    std_devs = np.sqrt(np.diag(scaled_cov))
    scaled_cov = scaled_cov / (std_devs[:, np.newaxis] * std_devs[np.newaxis, :])
    
    # Create heatmap with better color scaling
    heatmap = sns.heatmap(scaled_cov, 
                         cmap='RdYlBu_r',
                         center=0,
                         vmin=-1,
                         vmax=1,
                         square=True,
                         xticklabels=False,
                         yticklabels=False,
                         cbar_kws={'label': 'Correlation'})
    
    plt.title(f"{title}\n(Correlation Matrix View)")
    plt.tight_layout()
    plt.savefig('covariance_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_asset_allocation(weights, title="Asset Allocation"):
    """
    Plot the asset allocation as a bar chart with improved visualization.
    """
    plt.figure(figsize=(15, 8))
    
    # Convert weights to percentage
    weights_pct = weights * 100
    
    # Sort weights in descending order and get non-zero weights
    sorted_weights = pd.Series(weights_pct).sort_values(ascending=False)
    non_zero_weights = sorted_weights[sorted_weights > 0.1]  # Show weights > 0.1%
    
    # Create bar plot
    bars = plt.bar(range(len(non_zero_weights)), non_zero_weights.values)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.title(f"{title}\n(Showing {len(non_zero_weights)} positions > 0.1%)")
    plt.xlabel("Asset Rank")
    plt.ylabel("Weight (%)")
    plt.grid(True, alpha=0.3)
    
    # Add summary statistics
    stats_text = f"Max Weight: {weights_pct.max():.1f}%\n"
    stats_text += f"Min Weight (>0): {weights_pct[weights_pct > 0].min():.1f}%\n"
    stats_text += f"Num Positions: {(weights_pct > 0).sum()}"
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('asset_allocation.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Read returns data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    returns_file = os.path.join(base_dir, "..", "Data", "Simple_Returns.xlsx")
    
    if os.path.exists(returns_file):
        returns_df = pd.read_excel(returns_file)
        print("Returns data shape:", returns_df.shape)
        
        # Run portfolio optimization
        metrics, ex_post_returns = run_portfolio_optimization(returns_df)
        
        # Print results
        print("\nPortfolio Characteristics (P(mv)oos):")
        print(f"Annualized Average Return (μ̄p): {metrics['annualized_return']:.4f}")
        print(f"Annualized Volatility (σp): {metrics['annualized_volatility']:.4f}")
        print(f"Average Risk-free Rate: {metrics['avg_rf_rate']:.4f}")
        print(f"Sharpe Ratio (SRp): {metrics['sharpe_ratio']:.4f}")
        print(f"Minimum Return: {metrics['min_return']:.4f}")
        print(f"Maximum Return: {metrics['max_return']:.4f}")
        
        # Plot returns
        plt.figure(figsize=(15, 8))
        plt.plot(ex_post_returns.index, ex_post_returns.values)
        plt.title("Portfolio Returns Over Time")
        plt.xlabel("Date")
        plt.ylabel("Return")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('portfolio_returns.png')
        plt.close()
    else:
        print("Returns file not found.")
