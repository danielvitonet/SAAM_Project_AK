import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cvxpy as cp


class CarbonAwarePortfolio:
    """
    Class to implement carbon-aware portfolio allocation strategies.

    This class implements the following strategies:
    - Compute carbon footprint of a portfolio
    - Construct minimum variance portfolio with carbon footprint constraint
    - Construct benchmark-tracking portfolio with carbon footprint constraint
    - Construct a net-zero pathway portfolio
    """

    def __init__(self, market_cap_annual_df, returns_df, scope1_df, scope2_df, revenue_df):
        """
        Initialize the CarbonAwarePortfolio class with necessary datasets.

        Parameters:
            market_cap_annual_df (pd.DataFrame): Annual market cap data (year-end)
            returns_df (pd.DataFrame): Monthly returns data
            scope1_df (pd.DataFrame): Annual Scope 1 emissions data
            scope2_df (pd.DataFrame): Annual Scope 2 emissions data
            revenue_df (pd.DataFrame): Annual revenue data
        """
        self.market_cap_annual_df = market_cap_annual_df
        self.returns_df = returns_df
        self.scope1_df = scope1_df
        self.scope2_df = scope2_df
        self.revenue_df = revenue_df

        # For AK Group: Europe / Scope 1+2
        self.use_scope1and2 = True

        # Store portfolio weights and metrics
        self.mv_portfolio_weights = {}
        self.vw_portfolio_weights = {}
        self.mv_carbon_weights = {}
        self.vw_carbon_weights = {}
        self.nz_portfolio_weights = {}

        # Store carbon footprints
        self.carbon_footprints = {}

        # Compute carbon intensity for each firm for each year
        self.compute_carbon_intensity()

    def compute_carbon_intensity(self):
        """
        Compute carbon intensity for each firm for each year.
        Carbon intensity = (Scope 1 + Scope 2 emissions) / Revenue
        Units: tons CO2 equivalent per million USD revenue
        """
        # Initialize carbon intensity DataFrame
        self.carbon_intensity = pd.DataFrame()
        self.carbon_intensity['ISIN'] = self.scope1_df['ISIN']
        if 'Name' in self.scope1_df.columns:
            self.carbon_intensity['Name'] = self.scope1_df['Name']
        elif 'NAME' in self.scope1_df.columns:
            self.carbon_intensity['Name'] = self.scope1_df['NAME']

        # Get annual columns (excluding ISIN and Name)
        scope1_years = [col for col in self.scope1_df.columns if col not in ['ISIN', 'Name', 'NAME']]
        scope2_years = [col for col in self.scope2_df.columns if col not in ['ISIN', 'Name', 'NAME']]
        revenue_years = [col for col in self.revenue_df.columns if col not in ['ISIN', 'Name', 'NAME']]

        # Ensure we have common years across all datasets
        common_years = sorted(set(scope1_years) & set(scope2_years) & set(revenue_years))
        print(f"Computing carbon intensity for years: {common_years}")

        # Create a mapping from ISIN to index for each DataFrame
        scope1_isin_map = {isin: i for i, isin in enumerate(self.scope1_df['ISIN'])}
        scope2_isin_map = {isin: i for i, isin in enumerate(self.scope2_df['ISIN'])}
        revenue_isin_map = {isin: i for i, isin in enumerate(self.revenue_df['ISIN'])}

        # Compute carbon intensity for each year
        for year in common_years:
            # Create arrays for emissions and revenue
            n_firms = len(self.carbon_intensity)
            scope1_emissions = np.zeros(n_firms)
            scope2_emissions = np.zeros(n_firms)
            revenues = np.zeros(n_firms)

            # Fill arrays with data, handling missing values
            for i, isin in enumerate(self.carbon_intensity['ISIN']):
                # Get Scope 1 emissions
                if isin in scope1_isin_map:
                    scope1_idx = scope1_isin_map[isin]
                    scope1_val = self.scope1_df.iloc[scope1_idx][year]
                    if pd.notna(scope1_val):
                        scope1_emissions[i] = scope1_val

                # Get Scope 2 emissions
                if isin in scope2_isin_map:
                    scope2_idx = scope2_isin_map[isin]
                    scope2_val = self.scope2_df.iloc[scope2_idx][year]
                    if pd.notna(scope2_val):
                        scope2_emissions[i] = scope2_val

                # Get revenue
                if isin in revenue_isin_map:
                    revenue_idx = revenue_isin_map[isin]
                    revenue_val = self.revenue_df.iloc[revenue_idx][year]
                    if pd.notna(revenue_val):
                        revenues[i] = revenue_val

            # Compute total emissions (Scope 1 + Scope 2)
            if self.use_scope1and2:
                total_emissions = scope1_emissions + scope2_emissions
            else:
                total_emissions = scope1_emissions

            # Store total emissions for later use
            self.carbon_intensity[f"Emissions_{year}"] = total_emissions
            self.carbon_intensity[f"Revenue_{year}"] = revenues

            # Compute carbon intensity (emissions per million USD revenue)
            # Avoid division by zero
            carbon_intensity = np.zeros(n_firms)
            for i in range(n_firms):
                if revenues[i] > 0:
                    carbon_intensity[i] = total_emissions[i] / revenues[i]
                else:
                    carbon_intensity[i] = np.nan

            # Store carbon intensity
            self.carbon_intensity[f"CI_{year}"] = carbon_intensity

        # Handle missing values using forward fill
        # First, sort columns to ensure chronological order
        emissions_cols = sorted([col for col in self.carbon_intensity.columns if col.startswith("Emissions_")])
        revenue_cols = sorted([col for col in self.carbon_intensity.columns if col.startswith("Revenue_")])
        ci_cols = sorted([col for col in self.carbon_intensity.columns if col.startswith("CI_")])

        # Forward fill emissions
        for i, col in enumerate(emissions_cols):
            if i > 0:
                mask = self.carbon_intensity[col].isna()
                self.carbon_intensity.loc[mask, col] = self.carbon_intensity.loc[mask, emissions_cols[i - 1]]

        # Forward fill revenue
        for i, col in enumerate(revenue_cols):
            if i > 0:
                mask = self.carbon_intensity[col].isna()
                self.carbon_intensity.loc[mask, col] = self.carbon_intensity.loc[mask, revenue_cols[i - 1]]

        # Forward fill carbon intensity
        for i, col in enumerate(ci_cols):
            if i > 0:
                mask = self.carbon_intensity[col].isna()
                self.carbon_intensity.loc[mask, col] = self.carbon_intensity.loc[mask, ci_cols[i - 1]]

        print(f"Carbon intensity computation complete for {len(common_years)} years.")

    def calculate_portfolio_carbon_footprint(self, weights, year, initial_investment=1e6):
        """
        Calculate the carbon footprint of a portfolio.
        """
        # Ensure year is an integer
        year = int(year)

        # Get carbon intensity for the selected year
        ci_col = f"CI_{year}"
        emissions_col = f"Emissions_{year}"

        # Identify available years in market cap data
        available_years = [col for col in self.market_cap_annual_df.columns if isinstance(col, int)]
        available_years = sorted(available_years)

        # Find the closest available year for market cap data
        closest_year = min(available_years, key=lambda x: abs(int(x) - year))

        # Get market cap for the closest year
        market_cap_col_index = self.market_cap_annual_df.columns.get_loc(closest_year)
        market_cap = self.market_cap_annual_df.iloc[:, market_cap_col_index].values

        # Replace missing values with 0
        market_cap = np.nan_to_num(market_cap, 0)

        # Extract carbon intensity and emissions
        carbon_intensity = self.carbon_intensity[ci_col].values
        carbon_emissions = self.carbon_intensity[emissions_col].values

        # Replace missing values with 0
        carbon_intensity = np.nan_to_num(carbon_intensity, 0)
        carbon_emissions = np.nan_to_num(carbon_emissions, 0)

        # Ensure consistent dimensions
        min_length = min(len(weights), len(carbon_intensity), len(market_cap), len(carbon_emissions))

        weights = weights[:min_length]
        carbon_intensity = carbon_intensity[:min_length]
        market_cap = market_cap[:min_length]
        carbon_emissions = carbon_emissions[:min_length]

        # Normalize weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            # Fallback to equal weights if all weights are zero
            weights = np.ones(min_length) / min_length

        # Calculate weighted average carbon intensity (WACI)
        waci = np.sum(weights * carbon_intensity)

        # Calculate ownership of each firm
        total_market_cap = np.sum(market_cap)
        if total_market_cap > 0:
            market_weights = market_cap / total_market_cap
        else:
            market_weights = np.zeros_like(market_cap)

        # Calculate dollar investment in each firm
        dollar_investment = weights * initial_investment

        # Calculate ownership fraction for each firm
        ownership = np.zeros_like(weights)
        for i in range(len(weights)):
            if market_cap[i] > 0:
                ownership[i] = dollar_investment[i] / market_cap[i]
            else:
                ownership[i] = 0

        # Calculate carbon footprint (owned emissions per million USD invested)
        owned_emissions = ownership * carbon_emissions
        carbon_footprint = np.sum(owned_emissions) / initial_investment

        return waci, carbon_footprint

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

        # Define optimization variables
        w = cp.Variable(n_assets)
        risk = cp.quad_form(w, cov_matrix)

        # Define optimization problem
        prob = cp.Problem(
            cp.Minimize(risk),
            [cp.sum(w) == 1, w >= 0]
        )

        # Solve the problem
        try:
            prob.solve(solver=cp.ECOS)
            if prob.status == "optimal":
                weights = w.value
                weights[np.abs(weights) < 1e-6] = 0
                weights = weights / np.sum(weights)
                return weights
            else:
                print(f"Warning: Optimization problem status: {prob.status}")
                # Fallback to equal weights
                return np.ones(n_assets) / n_assets
        except Exception as e:
            print(f"Error in optimization: {str(e)}")
            # Fallback to equal weights
            return np.ones(n_assets) / n_assets

    def compute_mv_portfolio_with_carbon_constraint(self, returns, cov_matrix, year, carbon_footprint_limit):
        """
        Compute minimum variance portfolio with carbon footprint constraint.
        """
        # Trova la dimensione minima comune tra i vari array
        n_assets = min(
            cov_matrix.shape[0],
            len(returns),
            len(self.carbon_intensity[f"Emissions_{year}"].values),
            len(self.market_cap_annual_df.iloc[:, 2].values)
        )

        # Tronca tutti i dati alla stessa dimensione
        returns = returns[:n_assets]
        cov_matrix = cov_matrix[:n_assets, :n_assets]

        # Ottieni dati di emissioni e market cap
        emissions_col = f"Emissions_{year}"
        emissions = self.carbon_intensity[emissions_col].values[:n_assets]

        # Ottieni market cap per l'anno selezionato
        if str(year) in self.market_cap_annual_df.columns:
            market_cap_col_index = self.market_cap_annual_df.columns.get_loc(str(year))
            market_cap = self.market_cap_annual_df.iloc[:, market_cap_col_index].values[:n_assets]
        else:
            print(f"Warning: Market cap data not available for {year}. Using first available year.")
            market_cap = self.market_cap_annual_df.iloc[:, 2].values[:n_assets]

        # Gestisci valori mancanti
        emissions = np.nan_to_num(emissions, 0)
        market_cap = np.nan_to_num(market_cap, 0)

        # Aggiungi un piccolo valore per evitare divisione per zero
        market_cap[market_cap == 0] = 1e-10

        # Definisci variabili di ottimizzazione
        w = cp.Variable(n_assets)
        risk = cp.quad_form(w, cov_matrix)

        # Calcola frazione di proprietà
        ownership = cp.multiply(w, 1.0 / market_cap)

        # Calcola carbon footprint
        carbon_footprint = cp.sum(cp.multiply(ownership, emissions))

        # Definisci problema di ottimizzazione
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            carbon_footprint <= carbon_footprint_limit
        ]

        prob = cp.Problem(cp.Minimize(risk), constraints)

        # Prova solver multipli
        solvers = [cp.ECOS, cp.CVXOPT]

        for solver in solvers:
            try:
                prob.solve(solver=solver)

                if prob.status == "optimal":
                    weights = w.value
                    weights[np.abs(weights) < 1e-6] = 0
                    weights = weights / np.sum(weights)
                    return weights
            except Exception as e:
                print(f"Solver {solver} fallito: {str(e)}")

        # Metodo di fallback: campionamento casuale
        print("Utilizzo metodo di ottimizzazione di fallback...")
        best_weights = np.ones(n_assets) / n_assets
        best_risk = np.inf

        for _ in range(100):
            weights = np.random.random(n_assets)
            weights = weights / np.sum(weights)

            ownership = weights / market_cap
            carbon_footprint = np.sum(ownership * emissions)

            if carbon_footprint <= carbon_footprint_limit:
                risk = weights.T @ cov_matrix @ weights
                if risk < best_risk:
                    best_risk = risk
                    best_weights = weights

        return best_weights

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
        n_assets = cov_matrix.shape[0]

        # Get emissions and market cap data
        emissions_col = f"Emissions_{year}"
        emissions = self.carbon_intensity[emissions_col].values

        # Get market cap for the selected year
        if str(year) in self.market_cap_annual_df.columns:
            market_cap = self.market_cap_annual_df[str(year)].values
        else:
            print(f"Warning: Market cap data not available for {year}. Using first available year.")
            market_cap = self.market_cap_annual_df.iloc[:, 2].values

        # Replace missing values with 0
        emissions = np.nan_to_num(emissions, 0)
        market_cap = np.nan_to_num(market_cap, 0)

        # Ensure benchmark weights are properly sized
        if len(vw_weights) < n_assets:
            temp_weights = np.zeros(n_assets)
            temp_weights[:len(vw_weights)] = vw_weights
            vw_weights = temp_weights
        elif len(vw_weights) > n_assets:
            vw_weights = vw_weights[:n_assets]

        # Define optimization variables
        w = cp.Variable(n_assets)

        # Calculate tracking error
        tracking_error = cp.quad_form(w - vw_weights, cov_matrix)

        # Calculate ownership fractions
        ownership = cp.multiply(w, 1.0 / market_cap)

        # Calculate carbon footprint
        carbon_footprint = cp.sum(cp.multiply(ownership, emissions))

        # Define optimization problem
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            carbon_footprint <= carbon_footprint_limit
        ]

        prob = cp.Problem(cp.Minimize(tracking_error), constraints)

        # Solve the problem
        try:
            prob.solve()
            if prob.status == "optimal":
                weights = w.value
                weights[np.abs(weights) < 1e-6] = 0
                weights = weights / np.sum(weights)
                return weights
            else:
                print(f"Warning: Optimization problem status: {prob.status}")
                # Fallback to benchmark weights
                return vw_weights
        except Exception as e:
            print(f"Error in tracking error optimization: {str(e)}")
            print("Using fallback optimization method...")

            # Fallback method: start with benchmark weights and adjust to meet carbon constraint
            _, cf = self.calculate_portfolio_carbon_footprint(vw_weights, year)

            if cf <= carbon_footprint_limit:
                return vw_weights

            # Otherwise, try to find a feasible solution using a grid search
            best_weights = np.ones(n_assets) / n_assets
            best_tracking_error = np.inf

            # Generate 100 random portfolios and pick the one with lowest tracking error that meets carbon constraint
            for _ in range(100):
                weights = np.random.random(n_assets)
                weights = weights / np.sum(weights)
                _, cf = self.calculate_portfolio_carbon_footprint(weights, year)

                if cf <= carbon_footprint_limit:
                    tracking_error = np.sum((weights - vw_weights) ** 2)
                    if tracking_error < best_tracking_error:
                        best_tracking_error = tracking_error
                        best_weights = weights

            return best_weights

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
        # Calculate years since base year
        years_since_base = int(year) - int(base_year)
        if years_since_base < 0:
            years_since_base = 0

        # Calculate carbon footprint limit for this year
        carbon_footprint_limit = carbon_footprint_base * (1 - theta) ** (years_since_base)

        print(f"Net Zero Portfolio - Year {year}: Carbon limit = {carbon_footprint_limit:.2f}")

        # Compute portfolio using tracking error minimization with updated carbon constraint
        weights = self.compute_tracking_error_portfolio_with_carbon_constraint(
            vw_weights, cov_matrix, year, carbon_footprint_limit
        )

        return weights

    def compute_covariance_matrix(self, returns_window):
        """
        Compute covariance matrix from returns data.

        Parameters:
            returns_window (numpy.ndarray): Matrix of returns (time x assets)

        Returns:
            numpy.ndarray: Covariance matrix
        """
        # Remove rows with any NaN or inf values
        valid_mask = np.isfinite(returns_window).all(axis=1)
        clean_returns = returns_window[valid_mask]

        # If no valid returns, return identity matrix
        if clean_returns.size == 0:
            print("Warning: No valid returns found for covariance matrix. Using identity matrix.")
            return np.eye(returns_window.shape[0])

        # Compute sample covariance matrix
        try:
            cov_matrix = np.cov(clean_returns, rowvar=False)
        except Exception as e:
            print(f"Error computing covariance matrix: {e}")
            return np.eye(returns_window.shape[0])

        # Ensure positive definiteness
        try:
            eigvals = np.linalg.eigvals(cov_matrix)
            min_eig = np.min(np.real(eigvals))

            if min_eig < 0:
                # Add a small positive value to the diagonal
                cov_matrix = cov_matrix + (-min_eig + 1e-6) * np.eye(cov_matrix.shape[0])
        except Exception as e:
            print(f"Error ensuring positive definiteness: {e}")
            # Fallback to identity matrix
            return np.eye(returns_window.shape[0])

        return cov_matrix

    def compute_all_portfolio_weights(self, window_size=120, start_year=2013, end_year=2023):
        """
        Compute all portfolio weights for the specified time period.
        """
        print(f"Computing portfolio weights from {start_year} to {end_year}...")

        # Validate input years
        available_years = [col for col in self.market_cap_annual_df.columns if isinstance(col, int)]
        available_years = sorted(available_years)

        if not available_years:
            raise ValueError("No years available in market capitalization data")

        # Adjust start and end years to available data range
        start_year = max(start_year, available_years[0])
        end_year = min(end_year, available_years[-1])

        print(f"Adjusted years: {start_year} to {end_year}")
        print(f"Available market cap years: {available_years}")

        # Get returns data
        returns_data = self.returns_df.iloc[:, 2:].values

        # Convert returns to numpy array (time x assets)
        n_periods, n_assets = returns_data.shape

        # Value-weighted benchmark weights computation
        for year in range(start_year, end_year + 1):
            try:
                # Find the closest available year for market cap data
                closest_year = min(available_years, key=lambda x: abs(int(x) - year))
                print(f"Market cap data for {year} using closest year: {closest_year}")

                # Get market cap data for the year using integer column indexing
                market_cap_col_index = self.market_cap_annual_df.columns.get_loc(closest_year)
                market_cap = self.market_cap_annual_df.iloc[:, market_cap_col_index].values

                # Handle missing values
                market_cap = np.nan_to_num(market_cap, 0)

                # Compute value-weighted benchmark weights
                total_market_cap = np.sum(market_cap)
                if total_market_cap > 0:
                    vw_weights = market_cap / total_market_cap
                else:
                    # Fallback to equal weights
                    vw_weights = np.ones(n_assets) / n_assets

                # Store the value-weighted benchmark weights
                self.vw_portfolio_weights[year] = vw_weights

                # Calculate carbon footprint of benchmark portfolio
                waci_vw, cf_vw = self.calculate_portfolio_carbon_footprint(vw_weights, str(year))
                self.carbon_footprints[f"vw_{year}"] = (waci_vw, cf_vw)

                print(f"Year {year} - VW Portfolio - WACI: {waci_vw:.2f}, CF: {cf_vw:.2f}")

                # Extract window of returns for computing covariance matrix
                window_end = min(n_periods, (year - start_year + 1) * 12 + window_size)
                window_start = max(0, window_end - window_size)

                returns_window = returns_data[:, window_start:window_end]

                # Check if we have enough valid returns
                valid_returns = returns_window[np.isfinite(returns_window).all(axis=1)]

                if valid_returns.size == 0:
                    print(f"Warning: No valid returns found for year {year}. Skipping minimum variance portfolio.")
                    continue

                # Compute expected returns
                expected_returns = np.nanmean(valid_returns, axis=0)

                # Compute covariance matrix
                cov_matrix = self.compute_covariance_matrix(valid_returns)

                # Ensure expected returns match covariance matrix dimensions
                if len(expected_returns) != cov_matrix.shape[0]:
                    print(f"Warning: Dimension mismatch for year {year}. Skipping minimum variance portfolio.")
                    continue

                # Compute minimum variance portfolio weights
                mv_weights = self.compute_minimum_variance_weights(expected_returns, cov_matrix)
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
                self.mv_carbon_weights[year] = mv_carbon_weights

                # Calculate carbon footprint of carbon-constrained minimum variance portfolio
                waci_mvc, cf_mvc = self.calculate_portfolio_carbon_footprint(mv_carbon_weights, str(year))
                self.carbon_footprints[f"mvc_{year}"] = (waci_mvc, cf_mvc)

                print(f"Year {year} - MV Carbon Portfolio - WACI: {waci_mvc:.2f}, CF: {cf_mvc:.2f}")

                # Compute value-weighted portfolio with carbon constraint (50% of VW)
                carbon_limit = 0.5 * cf_vw
                vw_carbon_weights = self.compute_tracking_error_portfolio_with_carbon_constraint(
                    vw_weights, cov_matrix, str(year), carbon_limit
                )
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
                    self.nz_portfolio_weights[year] = nz_weights

                    # Calculate carbon footprint of net zero portfolio
                    waci_nz, cf_nz = self.calculate_portfolio_carbon_footprint(nz_weights, str(year))
                    self.carbon_footprints[f"nz_{year}"] = (waci_nz, cf_nz)

                    print(f"Year {year} - Net Zero Portfolio - WACI: {waci_nz:.2f}, CF: {cf_nz:.2f}")

            except Exception as e:
                print(f"Error processing year {year}: {str(e)}")
                continue

        print("All portfolio weights computed successfully.")

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
        # Get monthly returns data
        returns_df = self.returns_df.copy()

        # Get date columns
        date_cols = returns_df.columns[2:]

        # Convert date strings to datetime objects
        dates = pd.to_datetime(date_cols)

        # Create a dictionary to map each date to its corresponding weights
        weights_map = {}
        for year in range(start_year, end_year + 1):
            if year in weights_dict:
                # Find dates in this year
                year_dates = [d for d in dates if d.year == year]
                for date in year_dates:
                    weights_map[date] = weights_dict[year - 1]  # Use weights from previous year

        # Calculate portfolio returns for each month
        returns_list = []
        dates_list = []

        for date in sorted(weights_map.keys()):
            date_str = date.strftime('%Y-%m-%d')
            if date_str in returns_df.columns:
                weights = weights_map[date]
                returns = returns_df[date_str].values

                # Handle missing values
                returns = np.nan_to_num(returns, 0)

                # Ensure weights and returns have the same length
                min_len = min(len(weights), len(returns))
                weights = weights[:min_len]
                returns = returns[:min_len]

                # Normalize weights to sum to 1
                if np.sum(weights) > 0:
                    weights = weights / np.sum(weights)

                # Calculate portfolio return
                portfolio_return = np.sum(weights * returns)

                # Store result
                returns_list.append(portfolio_return)
                dates_list.append(date)

        # Create Series with returns
        portfolio_returns = pd.Series(returns_list, index=dates_list)

        return portfolio_returns

    def plot_carbon_footprints(self, start_year=2014, end_year=2023):
        """
        Plot carbon footprints of all portfolios over time.

        Parameters:
            start_year (int): Starting year for plotting
            end_year (int): Ending year for plotting
        """
        years = list(range(start_year, end_year + 1))

        # Extract carbon footprints for each portfolio type
        mv_footprints = [self.carbon_footprints.get(f"mv_{year}", (0, 0))[1] for year in years]
        vw_footprints = [self.carbon_footprints.get(f"vw_{year}", (0, 0))[1] for year in years]
        mvc_footprints = [self.carbon_footprints.get(f"mvc_{year}", (0, 0))[1] for year in years]
        vwc_footprints = [self.carbon_footprints.get(f"vwc_{year}", (0, 0))[1] for year in years]
        nz_footprints = [self.carbon_footprints.get(f"nz_{year}", (0, 0))[1] for year in years]

        # Create plot
        plt.figure(figsize=(12, 8))
        plt.plot(years, mv_footprints, 'o-', label='Minimum Variance')
        plt.plot(years, vw_footprints, 's-', label='Value-Weighted')
        plt.plot(years, mvc_footprints, '^-', label='MV with 50% Carbon Reduction')
        plt.plot(years, vwc_footprints, 'd-', label='VW with 50% Carbon Reduction')
        plt.plot(years, nz_footprints, 'x-', label='Net Zero Pathway')

        plt.title('Carbon Footprints Over Time')
        plt.xlabel('Year')
        plt.ylabel('Carbon Footprint (tons CO2 per million USD invested)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(years)
        plt.tight_layout()

        # Save plot
        plt.savefig('carbon_footprints.png', dpi=300)
        plt.close()

    def compare_portfolio_performance(self, start_year=2014, end_year=2023):
        """
        Compare performance of all portfolio strategies.
        """

        def safe_process_returns(returns):
            """
            Safely process portfolio returns by converting to numeric and handling non-finite values.
            """
            # Convert to pandas Series if not already
            if not isinstance(returns, pd.Series):
                returns = pd.Series(returns)

            # Convert to numeric, coercing errors
            returns_numeric = pd.to_numeric(returns, errors='coerce')

            # Remove NaN and inf values
            returns_clean = returns_numeric[np.isfinite(returns_numeric)]

            return returns_clean

        # Fallback function to compute portfolio metrics safely
        def compute_portfolio_metrics(returns):
            """
            Compute portfolio performance metrics with robust error handling.
            """
            # Clean returns
            clean_returns = safe_process_returns(returns)

            if len(clean_returns) == 0:
                print(f"Warning: No valid returns found for this portfolio.")
                return {
                    'Annual Return': 0,
                    'Annual Volatility': 0,
                    'Sharpe Ratio': 0,
                    'Max Drawdown': 0,
                    'Min Monthly Return': 0,
                    'Max Monthly Return': 0
                }

            # Annualization factor (monthly returns)
            ann_factor = 12

            # Calculate metrics
            annual_return = clean_returns.mean() * ann_factor
            annual_vol = clean_returns.std() * np.sqrt(ann_factor)

            # Simple Sharpe Ratio calculation (assuming 0 risk-free rate)
            sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0

            # Compute cumulative returns for max drawdown
            cum_returns = (1 + clean_returns).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns / running_max) - 1
            max_drawdown = drawdown.min()

            return {
                'Annual Return': annual_return,
                'Annual Volatility': annual_vol,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown': max_drawdown,
                'Min Monthly Return': clean_returns.min(),
                'Max Monthly Return': clean_returns.max()
            }

        # Compute portfolio returns with safe processing
        mv_returns = self.compute_portfolio_returns(self.mv_portfolio_weights, start_year, end_year)
        vw_returns = self.compute_portfolio_returns(self.vw_portfolio_weights, start_year, end_year)

        # Check if carbon-constrained portfolios were computed
        mvc_returns = (self.compute_portfolio_returns(self.mv_carbon_weights, start_year, end_year)
                       if self.mv_carbon_weights else pd.Series())
        vwc_returns = (self.compute_portfolio_returns(self.vw_carbon_weights, start_year, end_year)
                       if self.vw_carbon_weights else pd.Series())
        nz_returns = (self.compute_portfolio_returns(self.nz_portfolio_weights, start_year, end_year)
                      if self.nz_portfolio_weights else pd.Series())

        # Compute metrics for each portfolio
        metrics = {}
        portfolio_names = [
            ('Minimum Variance', mv_returns),
            ('Value-Weighted', vw_returns),
            ('MV Carbon-Constrained', mvc_returns),
            ('VW Carbon-Constrained', vwc_returns),
            ('Net Zero Pathway', nz_returns)
        ]

        for name, returns in portfolio_names:
            if not returns.empty:
                metrics[name] = compute_portfolio_metrics(returns)
            else:
                print(f"No returns found for {name} portfolio")

        # Create DataFrame with metrics
        metrics_df = pd.DataFrame(metrics)

        # Plot performance metrics
        plt.figure(figsize=(15, 10))

        metrics_to_plot = [
            'Annual Return',
            'Annual Volatility',
            'Sharpe Ratio',
            'Max Drawdown'
        ]

        for i, metric in enumerate(metrics_to_plot, 1):
            plt.subplot(2, 2, i)
            plt.bar(metrics_df.columns, metrics_df.loc[metric])
            plt.title(metric)
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig('portfolio_metrics_comparison.png')
        plt.close()

        return metrics_df

    def calculate_max_drawdown(self, returns):
        """
        Calculate maximum drawdown from a series of returns.

        Parameters:
            returns (pd.Series): Series of returns

        Returns:
            float: Maximum drawdown
        """
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()

        # Calculate running maximum
        running_max = cum_returns.cummax()

        # Calculate drawdown
        drawdown = (cum_returns / running_max) - 1

        # Calculate maximum drawdown
        max_drawdown = drawdown.min()

        return max_drawdown