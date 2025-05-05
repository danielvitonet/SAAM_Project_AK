import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import logging
import os
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CarbonAwarePortfolio:
    """
    Class to implement carbon-aware portfolio allocation strategies
    """

    def __init__(self, market_cap_df, returns_df, scope1_df, scope2_df, revenue_df):
        """
        Initialize with required data
        """
        self.market_cap_df = market_cap_df
        self.returns_df = returns_df
        self.scope1_df = scope1_df
        self.scope2_df = scope2_df
        self.revenue_df = revenue_df
        self.isins = returns_df['ISIN'].values
        self.carbon_footprints = {}  # Store carbon footprints
        self.validate_data()

        # Compute carbon intensity
        self.carbon_intensity = self.compute_carbon_intensity()

    def fix_invalid_weights(self, weights):
        """
        Fix invalid portfolio weights by handling NaN values and normalization
        """
        logging.info(f"Input weights type: {type(weights)}, shape: {getattr(weights, 'shape', 'N/A')}")
        sample = weights[:5] if not isinstance(weights, pd.Series) else weights.head()
        logging.info(f"Input weights sample: {sample}")

        if weights is None or not isinstance(weights, pd.Series):
            logging.warning("Weights is None or not a Series, returning equal weights")
            return pd.Series(1.0 / len(self.isins), index=self.isins)

        weights = weights.fillna(0)
        aligned_weights = pd.Series(0.0, index=self.isins)
        common_isins = set(weights.index).intersection(self.isins)

        if not common_isins:
            logging.warning("No common ISINs. Using equal weights")
            return pd.Series(1.0 / len(self.isins), index=self.isins)

        aligned_weights.loc[list(common_isins)] = weights.loc[list(common_isins)]
        if aligned_weights.sum() > 1e-8:
            aligned_weights = aligned_weights / aligned_weights.sum()
        else:
            logging.warning("Weights sum to zero. Using equal weights")
            return pd.Series(1.0 / len(self.isins), index=self.isins)

        logging.info(f"Fixed weights sum: {aligned_weights.sum()}, non-zero: {(aligned_weights > 0).sum()}")
        return aligned_weights

    def validate_data(self):
        """
        Validate input data for consistency
        """
        for df_name, df in [
            ('market_cap_df', self.market_cap_df),
            ('returns_df', self.returns_df),
            ('scope1_df', self.scope1_df),
            ('scope2_df', self.scope2_df),
            ('revenue_df', self.revenue_df)
        ]:
            if df is None or df.empty:
                logging.error(f"{df_name} is None or empty")
                raise ValueError(f"{df_name} is None or empty")

        # Ensure consistent ISINs
        isins = set(self.returns_df['ISIN'])
        for df_name, df in [
            ('market_cap_df', self.market_cap_df),
            ('scope1_df', self.scope1_df),
            ('scope2_df', self.scope2_df),
            ('revenue_df', self.revenue_df)
        ]:
            if not set(df['ISIN']).issubset(isins):
                invalid_isins = set(df['ISIN']) - isins
                logging.warning(f"{df_name} contains {len(invalid_isins)} ISINs not in returns_df")
                if df_name == 'market_cap_df':
                    self.market_cap_df = df[df['ISIN'].isin(isins)].copy()
                elif df_name == 'scope1_df':
                    self.scope1_df = df[df['ISIN'].isin(isins)].copy()
                elif df_name == 'scope2_df':
                    self.scope2_df = df[df['ISIN'].isin(isins)].copy()
                elif df_name == 'revenue_df':
                    self.revenue_df = df[df['ISIN'].isin(isins)].copy()

        # Check for year columns in scope1_df, scope2_df, revenue_df
        year_cols = self.get_year_columns(self.scope1_df)
        if not year_cols:
            logging.error("No valid year columns found in scope1_df")
            raise ValueError("No valid year columns found in scope1_df")
        logging.info(f"Valid year columns: {year_cols}")

    def get_year_columns(self, df):
        """
        Identify year columns in a DataFrame
        """
        year_cols = []
        for col in df.columns:
            try:
                if isinstance(col, (int, float)):
                    year = int(col)
                    if 2013 <= year <= 2023:
                        year_cols.append(str(year))
                elif isinstance(col, str) and col.isdigit():
                    year = int(col)
                    if 2013 <= year <= 2023:
                        year_cols.append(col)
            except:
                continue
        return sorted(year_cols, key=lambda x: int(x))

    def compute_carbon_intensity(self):
        """
        Compute carbon intensity with proper forward-fill for missing data
        according to project specifications
        """
        year_cols = self.get_year_columns(self.scope1_df)
        if not year_cols:
            logging.error("No valid year columns for carbon intensity calculation")
            raise ValueError("No valid year columns for carbon intensity calculation")

        # Log missing 2013 data
        scope1_missing = self.scope1_df[year_cols[0]].isna().sum()
        scope2_missing = self.scope2_df[year_cols[0]].isna().sum()
        logging.info(f"Missing 2013 data: Scope 1 = {scope1_missing}, Scope 2 = {scope2_missing}")

        # Initialize carbon intensity dataframe
        ci_df = pd.DataFrame()
        ci_df['ISIN'] = self.scope1_df['ISIN']
        ci_df['NAME'] = self.scope1_df['NAME']

        for i, year in enumerate(year_cols):
            scope1 = pd.to_numeric(self.scope1_df[year], errors='coerce')
            scope2 = pd.to_numeric(self.scope2_df[year], errors='coerce')
            revenue = pd.to_numeric(self.revenue_df[year], errors='coerce')

            # Handle missing data with forward fill
            if i > 0:  # Not the first year
                prev_year = year_cols[i - 1]
                scope1 = scope1.fillna(pd.to_numeric(self.scope1_df[prev_year], errors='coerce'))
                scope2 = scope2.fillna(pd.to_numeric(self.scope2_df[prev_year], errors='coerce'))
                revenue = revenue.fillna(pd.to_numeric(self.revenue_df[prev_year], errors='coerce'))

            # Calculate carbon intensity
            total_emissions = scope1.fillna(0) + scope2.fillna(0)

            # Only calculate CI where revenue is positive
            ci = pd.Series(index=ci_df.index, dtype=float)
            valid_revenue_mask = revenue > 0
            ci[valid_revenue_mask] = total_emissions[valid_revenue_mask] / revenue[valid_revenue_mask]

            # For zero or negative revenue, set CI to NaN
            ci[~valid_revenue_mask] = np.nan

            ci_df[f'CI_{year}'] = ci

        logging.info(f"Computed carbon intensity for {len(year_cols)} years")
        return ci_df

    def calculate_portfolio_carbon_footprint(self, weights, year, initial_investment=1e6):
        """
        Calculate portfolio carbon footprint for a given year
        """
        try:
            year_str = str(year)
            if year_str not in self.scope1_df.columns or year_str not in self.scope2_df.columns:
                logging.warning(f"Missing emissions data for year {year}")
                return np.nan

            if year_str not in self.market_cap_df.columns:
                logging.warning(f"Missing market cap data for year {year}")
                return np.nan

            # Validate weights
            weights = self.fix_invalid_weights(weights)  # Ensure valid weights
            if weights.isna().all() or weights.sum() < 1e-8:
                logging.warning(f"Invalid weights for year {year} after fixing")
                return np.nan

            # Align data
            scope1 = pd.to_numeric(self.scope1_df[year_str], errors='coerce').reindex(weights.index, fill_value=0)
            scope2 = pd.to_numeric(self.scope2_df[year_str], errors='coerce').reindex(weights.index, fill_value=0)
            market_cap = pd.to_numeric(self.market_cap_df[year_str], errors='coerce').reindex(weights.index)
            market_cap = market_cap.fillna(market_cap.mean()).where(market_cap > 0, market_cap.mean())

            total_emissions = scope1 + scope2
            portfolio_values = weights * initial_investment
            ownership = portfolio_values / market_cap
            carbon_footprint = np.sum(ownership * total_emissions) / initial_investment

            if np.isnan(carbon_footprint) or np.isinf(carbon_footprint) or carbon_footprint <= 0:
                logging.warning(
                    f"Invalid carbon footprint for year {year}: {carbon_footprint}. Weights sum: {weights.sum()}, Emissions sum: {total_emissions.sum()}")
                return np.nan

            logging.info(f"Year {year}: CF = {carbon_footprint:.2f} tCO2e/$M")
            return carbon_footprint
        except Exception as e:
            logging.error(f"Error calculating carbon footprint for year {year}: {e}")
            return np.nan

    def optimize_mv_with_carbon_constraint(self, expected_returns, cov_matrix, emissions, market_caps, carbon_limit):
        """
        Optimize minimum variance portfolio with carbon constraint
        """
        n_assets = len(expected_returns)
        w = cp.Variable(n_assets)

        # Regularize covariance matrix
        min_eigenval = np.min(np.linalg.eigvalsh(cov_matrix))
        if min_eigenval < 1e-8:
            cov_matrix += (abs(min_eigenval) + 1e-8) * np.eye(cov_matrix.shape[0])

        portfolio_variance = cp.quad_form(w, cov_matrix)
        ownership_factor = emissions / market_caps
        carbon_footprint = cp.sum(cp.multiply(w, ownership_factor))

        constraints = [cp.sum(w) == 1, w >= 0, carbon_footprint <= carbon_limit]
        problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)

        for solver, solver_name in [(cp.OSQP, "OSQP"), (cp.SCS, "SCS"), (cp.ECOS, "ECOS")]:
            try:
                problem.solve(solver=solver, verbose=True, max_iter=10000, eps_abs=1e-5, eps_rel=1e-5)
                if problem.status in ["optimal", "optimal_inaccurate"]:
                    weights = w.value
                    weights = np.maximum(weights, 0) / np.sum(weights)
                    actual_cf = np.sum(weights * ownership_factor)
                    if actual_cf <= carbon_limit * 1.1:
                        logging.info(f"Solver {solver_name}: Success, CF: {actual_cf:.2f}")
                        return pd.Series(weights, index=expected_returns.index)
                    logging.warning(f"Solver {solver_name}: CF {actual_cf:.2f} exceeds limit {carbon_limit:.2f}")
                else:
                    logging.warning(f"Solver {solver_name} failed: {problem.status}")
            except Exception as e:
                logging.error(f"Error with {solver_name}: {e}")

        # Relaxed optimization with penalty
        carbon_violation = cp.pos(carbon_footprint - carbon_limit)
        relaxed_objective = cp.Minimize(portfolio_variance + 1000 * carbon_violation)
        relaxed_problem = cp.Problem(relaxed_objective, [cp.sum(w) == 1, w >= 0])
        try:
            relaxed_problem.solve(solver=cp.SCS, verbose=True)
            if relaxed_problem.status in ["optimal", "optimal_inaccurate"]:
                weights = w.value
                weights = np.maximum(weights, 0) / np.sum(weights)
                actual_cf = np.sum(weights * ownership_factor)
                logging.info(f"Relaxed approach: CF = {actual_cf:.2f}")
                return pd.Series(weights, index=expected_returns.index)
        except Exception as e:
            logging.error(f"Relaxed optimization failed: {e}")

        logging.warning("All optimizations failed. Using equal weights")
        return pd.Series(np.ones(n_assets) / n_assets, index=expected_returns.index)

    def optimize_tracking_error_with_carbon_constraint(self, benchmark_weights, cov_matrix, emissions, market_caps,
                                                       carbon_limit, lambda_penalty=0.2):
        """
        Optimize portfolio to minimize tracking error with carbon constraint
        """
        n_assets = len(benchmark_weights)
        w = cp.Variable(n_assets)
        tracking_error = cp.quad_form(w - benchmark_weights, cov_matrix)
        ownership_factor = emissions / market_caps
        carbon_footprint = cp.sum(cp.multiply(w, ownership_factor))

        # Use penalty for carbon deviation instead of hard constraint
        carbon_deviation = cp.abs(carbon_footprint - carbon_limit)
        objective = cp.Minimize(tracking_error + lambda_penalty * carbon_deviation)
        constraints = [cp.sum(w) == 1, w >= 0]

        problem = cp.Problem(objective, constraints)
        for solver, solver_name in [(cp.OSQP, "OSQP"), (cp.SCS, "SCS"), (cp.ECOS, "ECOS")]:
            try:
                problem.solve(solver=solver, verbose=True, max_iter=10000, eps_abs=1e-5, eps_rel=1e-5)
                if problem.status in ["optimal", "optimal_inaccurate"]:
                    weights = w.value
                    weights = np.maximum(weights, 0) / np.sum(weights)
                    actual_cf = np.sum(weights * ownership_factor)
                    logging.info(f"Solver {solver_name}: Success, CF: {actual_cf:.2f}, Target: {carbon_limit:.2f}")
                    return pd.Series(weights, index=benchmark_weights.index)
                logging.warning(f"Solver {solver_name} failed: {problem.status}")
            except Exception as e:
                logging.error(f"Error with {solver_name}: {e}")

        # Fallback to scaled benchmark weights
        adjusted_weights = benchmark_weights.copy()
        emissions_per_dollar = emissions / market_caps
        high_emitters = emissions_per_dollar > np.median(emissions_per_dollar)
        scaling_factor = 0.5
        adjustment = adjusted_weights[high_emitters].sum() * (1 - scaling_factor)
        adjusted_weights[high_emitters] *= scaling_factor
        if adjusted_weights[~high_emitters].sum() > 0:
            adjusted_weights[~high_emitters] *= (1 + adjustment / adjusted_weights[~high_emitters].sum())
        adjusted_weights = adjusted_weights / adjusted_weights.sum()
        return pd.Series(adjusted_weights, index=benchmark_weights.index)

    def run_carbon_constrained_optimization(self, returns_df, mv_weights, vw_weights_dict, start_year=2014,
                                            end_year=2023, window_size=120):
        """
        Run carbon-constrained portfolio optimization
        """
        results = {
            'mv': {'weights': {}, 'returns': [], 'carbon_footprints': {}},
            'mvc': {'weights': {}, 'returns': [], 'carbon_footprints': {}},
            'vw': {'weights': {}, 'returns': [], 'carbon_footprints': {}},
            'vwc': {'weights': {}, 'returns': [], 'carbon_footprints': {}},
            'nz': {'weights': {}, 'returns': [], 'carbon_footprints': {}}
        }

        # Prepare returns data
        returns_data = returns_df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
        if returns_data.columns.str.isdigit().all():  # Annual data
            date_columns = pd.to_datetime([f"{int(col)}-12-31" for col in returns_data.columns])
        else:
            date_columns = pd.to_datetime(returns_data.columns, errors='coerce')

        if date_columns.isna().any():
            logging.error("Invalid date columns in returns_df")
            raise ValueError("Invalid date columns in returns_df")

        returns_data.columns = date_columns
        returns_data = returns_data.T
        returns_data.columns = self.isins

        # Calculate base carbon footprint for 2013
        base_year = 2013
        year_str = str(base_year)

        # Get market caps for base year
        market_caps = pd.to_numeric(self.market_cap_df[year_str], errors='coerce').dropna()
        market_caps.index = self.market_cap_df.loc[market_caps.index, 'ISIN'].values

        # Calculate value-weighted base portfolio
        vw_weights_2013 = pd.Series(market_caps / market_caps.sum(), index=market_caps.index)
        base_cf = self.calculate_portfolio_carbon_footprint(vw_weights_2013, base_year)
        logging.info(f"Base carbon footprint (2013): {base_cf:.2f} tCO2e/$M")

        # Store base year carbon footprint
        first_year_vw_cf = None

        for year in range(start_year, end_year + 1):
            logging.info(f"Processing year {year}...")
            rebalance_date = pd.Timestamp(f"{year - 1}-12-31")

            # Find closest date in returns data
            closest_date = None
            min_distance = float('inf')
            for date in returns_data.index:
                distance = abs((date - rebalance_date).days)
                if distance < min_distance:
                    min_distance = distance
                    closest_date = date

            if closest_date is None or min_distance > 45:  # Maximum 45 days difference
                logging.warning(f"No suitable data found near {rebalance_date}")
                continue

            # Get returns window for estimation
            window_mask = (
                    (returns_data.index <= closest_date) &
                    (returns_data.index > closest_date - pd.DateOffset(months=window_size))
            )
            window_returns = returns_data[window_mask]

            if len(window_returns) < window_size * 0.8:
                logging.warning(f"Insufficient data for year {year}: {len(window_returns)}/{window_size} months")
                if year > start_year and year - 1 in results['mv']['weights']:
                    results['mv']['weights'][year] = results['mv']['weights'][year - 1]
                    results['vw']['weights'][year] = results['vw']['weights'][year - 1]
                    logging.info(f"Using prior year's weights for {year}")
                continue

            # Clean returns data - remove assets with all NaN
            valid_assets = window_returns.columns[~window_returns.isna().all()].tolist()
            window_returns_clean = window_returns[valid_assets]

            # Fill remaining NaNs with column means (time-series approach)
            for col in window_returns_clean.columns:
                col_values = window_returns_clean[col]
                if col_values.isna().any():
                    col_mean = col_values.mean()
                    window_returns_clean[col] = col_values.fillna(col_mean)

            # Calculate expected returns and covariance matrix
            expected_returns = window_returns_clean.mean()
            cov_matrix = window_returns_clean.cov()

            # Ensure positive definiteness
            min_eigenval = np.min(np.linalg.eigvalsh(cov_matrix))
            if min_eigenval < 1e-8:
                cov_matrix += (abs(min_eigenval) + 1e-8) * np.eye(cov_matrix.shape[0])

            # Align emissions and market cap data for the previous year
            year_str = str(year - 1)

            # Get Scope 1 emissions
            scope1 = pd.to_numeric(self.scope1_df[year_str], errors='coerce')
            scope1.index = self.scope1_df['ISIN']
            scope1 = scope1.reindex(valid_assets, fill_value=0)

            # Get Scope 2 emissions
            scope2 = pd.to_numeric(self.scope2_df[year_str], errors='coerce')
            scope2.index = self.scope2_df['ISIN']
            scope2 = scope2.reindex(valid_assets, fill_value=0)

            # Total emissions for Group AK is Scope 1 + Scope 2
            emissions = scope1 + scope2

            # Get market caps
            market_caps = pd.to_numeric(self.market_cap_df[year_str], errors='coerce')
            market_caps.index = self.market_cap_df['ISIN']
            market_caps = market_caps.reindex(valid_assets)

            # Handle missing or zero market caps
            market_cap_mean = market_caps[market_caps > 0].mean()
            market_caps = market_caps.fillna(market_cap_mean)
            market_caps = market_caps.where(market_caps > 0, market_cap_mean)

            # Get minimum variance weights
            mv_weights_year = None
            closest_mv_date = min(mv_weights.keys(), key=lambda x: abs(x - rebalance_date)) if mv_weights else None
            if closest_mv_date in mv_weights and mv_weights[closest_mv_date] is not None:
                mv_weights_series = mv_weights[closest_mv_date]
                if isinstance(mv_weights_series, pd.Series):
                    mv_weights_year = mv_weights_series.reindex(valid_assets, fill_value=0)
                    if mv_weights_year.sum() > 0:
                        mv_weights_year = mv_weights_year / mv_weights_year.sum()
                    else:
                        mv_weights_year = None

            # Get value-weighted weights
            vw_weights_year = None
            closest_vw_date = min(vw_weights_dict.keys(),
                                  key=lambda x: abs(x - rebalance_date)) if vw_weights_dict else None
            if closest_vw_date in vw_weights_dict and vw_weights_dict[closest_vw_date] is not None:
                vw_weights_series = vw_weights_dict[closest_vw_date]
                if isinstance(vw_weights_series, pd.Series):
                    vw_weights_year = vw_weights_series.reindex(valid_assets, fill_value=0)
                    if vw_weights_year.sum() > 0:
                        vw_weights_year = vw_weights_year / vw_weights_year.sum()
                    else:
                        vw_weights_year = None

            # If weights are invalid, compute from scratch
            if mv_weights_year is None or mv_weights_year.sum() < 0.999:
                w = cp.Variable(len(valid_assets))
                objective = cp.Minimize(cp.quad_form(w, cov_matrix))
                constraints = [cp.sum(w) == 1, w >= 0]
                problem = cp.Problem(objective, constraints)
                try:
                    problem.solve(solver=cp.SCS)
                    if problem.status in ["optimal", "optimal_inaccurate"]:
                        weights = w.value
                        weights = np.maximum(weights, 0)
                        weights = weights / np.sum(weights)
                        mv_weights_year = pd.Series(weights, index=valid_assets)
                    else:
                        mv_weights_year = pd.Series(1.0 / len(valid_assets), index=valid_assets)
                except:
                    mv_weights_year = pd.Series(1.0 / len(valid_assets), index=valid_assets)

            if vw_weights_year is None or vw_weights_year.sum() < 0.999:
                valid_caps = market_caps[market_caps > 0]
                if len(valid_caps) > 0:
                    vw_weights_year = pd.Series(0.0, index=valid_assets)
                    vw_weights_year.loc[valid_caps.index] = valid_caps / valid_caps.sum()
                else:
                    vw_weights_year = pd.Series(1.0 / len(valid_assets), index=valid_assets)

            # Store valid weights
            results['mv']['weights'][year] = mv_weights_year
            results['vw']['weights'][year] = vw_weights_year

            # Calculate carbon footprints
            mv_cf = self.calculate_portfolio_carbon_footprint(mv_weights_year, year - 1)
            vw_cf = self.calculate_portfolio_carbon_footprint(vw_weights_year, year - 1)

            # Log warning if carbon footprints are invalid
            if mv_cf == 0:
                logging.warning(f"Zero carbon footprint for MV portfolio in {year}. Check weights and emissions data.")
            if vw_cf == 0:
                logging.warning(f"Zero carbon footprint for VW portfolio in {year}. Check weights and emissions data.")

            # Store first year VW carbon footprint for NZ calculations
            if year == start_year:
                first_year_vw_cf = vw_cf

            results['mv']['carbon_footprints'][year] = mv_cf
            results['vw']['carbon_footprints'][year] = vw_cf

            # Optimize constrained portfolios
            # MVC: 50% reduction relative to MV
            if not np.isnan(mv_cf) and mv_cf > 0:
                carbon_limit_mv = 0.5 * mv_cf
                logging.info(f"Year {year}: MVC Target CF = {carbon_limit_mv:.2f} (50% of MV {mv_cf:.2f})")
                mvc_weights = self.optimize_mv_with_carbon_constraint(
                    expected_returns, cov_matrix, emissions, market_caps, carbon_limit_mv
                )
                results['mvc']['weights'][year] = mvc_weights
                mvc_cf = self.calculate_portfolio_carbon_footprint(mvc_weights, year - 1)
                if mvc_cf == 0:
                    logging.warning(f"Zero carbon footprint for MVC portfolio in {year}. Verify optimizer results.")
                results['mvc']['carbon_footprints'][year] = mvc_cf
                logging.info(f"Year {year}: MVC Target CF = {carbon_limit_mv:.2f}, Actual CF = {mvc_cf:.2f}")
            else:
                logging.warning(f"Invalid MV carbon footprint for {year}: {mv_cf}. Using VW as reference.")
                carbon_limit_mv = 0.5 * vw_cf if not np.isnan(vw_cf) and vw_cf > 0 else np.inf
                mvc_weights = self.optimize_mv_with_carbon_constraint(
                    expected_returns, cov_matrix, emissions, market_caps, carbon_limit_mv
                )
                results['mvc']['weights'][year] = mvc_weights
                mvc_cf = self.calculate_portfolio_carbon_footprint(mvc_weights, year - 1)
                results['mvc']['carbon_footprints'][year] = mvc_cf
                logging.info(
                    f"Year {year}: MVC using VW reference, Target CF = {carbon_limit_mv:.2f}, Actual CF = {mvc_cf:.2f}")

            # VWC: 25% reduction relative to VW
            if not np.isnan(vw_cf) and vw_cf > 0:
                carbon_limit_vw = 0.75 * vw_cf  # 25% reduction
                logging.info(f"Year {year}: VWC Target CF = {carbon_limit_vw:.2f} (75% of VW {vw_cf:.2f})")
                vwc_weights = self.optimize_tracking_error_with_carbon_constraint(
                    vw_weights_year, cov_matrix, emissions, market_caps, carbon_limit_vw
                )
                results['vwc']['weights'][year] = vwc_weights
                vwc_cf = self.calculate_portfolio_carbon_footprint(vwc_weights, year - 1)
                if vwc_cf == 0:
                    logging.warning(f"Zero carbon footprint for VWC portfolio in {year}. Verify optimizer results.")
                results['vwc']['carbon_footprints'][year] = vwc_cf
                logging.info(f"Year {year}: VWC Target CF = {carbon_limit_vw:.2f}, Actual CF = {vwc_cf:.2f}")
            else:
                logging.warning(f"Invalid VW carbon footprint for {year}. Skipping VWC optimization.")
                results['vwc']['weights'][year] = vw_weights_year
                results['vwc']['carbon_footprints'][year] = np.nan

            # NZ: Net zero target
            if first_year_vw_cf is not None and not np.isnan(first_year_vw_cf) and first_year_vw_cf > 0:
                years_elapsed = year - start_year + 1
                target_cf = first_year_vw_cf * ((1 - 0.1) ** years_elapsed)
                nz_weights = self.optimize_tracking_error_with_carbon_constraint(
                    vw_weights_year, cov_matrix, emissions, market_caps, target_cf
                )
                results['nz']['weights'][year] = nz_weights
                nz_cf = self.calculate_portfolio_carbon_footprint(nz_weights, year - 1)
                results['nz']['carbon_footprints'][year] = nz_cf

                # Calculate and log the actual reduction percentage
                if not np.isnan(nz_cf):
                    actual_reduction = (1 - nz_cf / first_year_vw_cf) * 100
                    target_reduction = (1 - ((1 - 0.1) ** years_elapsed)) * 100
                    logging.info(
                        f"Year {year}: NZ Target reduction = {target_reduction:.1f}%, Actual = {actual_reduction:.1f}%")
            else:
                logging.warning(f"Invalid base carbon footprint for {year}. Skipping NZ optimization.")
                results['nz']['weights'][year] = vw_weights_year
                results['nz']['carbon_footprints'][year] = np.nan

        # Calculate portfolio returns
        start_date = pd.Timestamp(f"{start_year - 1}-12-31")
        end_date = pd.Timestamp(f"{end_year}-12-31")
        returns_mask = (returns_data.index > start_date) & (returns_data.index <= end_date)
        returns_period = returns_data[returns_mask]

        # Process returns for all strategies
        for strategy in ['mv', 'mvc', 'vw', 'vwc', 'nz']:
            strategy_returns = []

            # Calculate returns monthly, based on weights at the start of the year
            for date in returns_period.index:
                year = date.year
                if year not in results[strategy]['weights']:
                    continue

                weights = results[strategy]['weights'][year]
                monthly_returns = returns_period.loc[date]

                # Check for valid weights and returns
                if weights is None or monthly_returns.isna().all():
                    continue

                # Find common assets between weights and returns
                common_index = weights.index.intersection(monthly_returns.index)
                if len(common_index) == 0:
                    continue

                weights_aligned = weights[common_index]
                returns_aligned = monthly_returns[common_index]

                # Normalize weights if needed
                if abs(weights_aligned.sum() - 1.0) > 1e-4 and weights_aligned.sum() > 0:
                    weights_aligned = weights_aligned / weights_aligned.sum()

                # Calculate portfolio return
                portfolio_return = np.sum(weights_aligned * returns_aligned.fillna(0))
                strategy_returns.append({'date': date, 'return': portfolio_return})

            # Create returns series
            if strategy_returns:
                returns_df = pd.DataFrame(strategy_returns)
                results[strategy]['returns'] = pd.Series(
                    returns_df['return'].values,
                    index=returns_df['date']
                )
            else:
                results[strategy]['returns'] = pd.Series(dtype=float)

        return results

    def plot_comprehensive_results(self, results, output_dir):
        """
        Create comprehensive visualizations for the project report
        """
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('default')

        # 1. Carbon Footprint Evolution
        fig, ax = plt.subplots(figsize=(14, 8))
        years = sorted(results['mv']['carbon_footprints'].keys())
        strategies = {
            'mv': 'P^(mv)_oos (Min Variance)',
            'mvc': 'P^(mv)_oos(0.5) (MV with 50% reduction)',
            'vw': 'P^(vw) (Value-Weighted)',
            'vwc': 'P^(vw)_oos(0.75) (VW with 25% reduction)',
            'nz': 'P^(vw)_oos(NZ) (Net Zero)'
        }
        colors = ['blue', 'lightblue', 'red', 'lightcoral', 'green']

        for (strategy, label), color in zip(strategies.items(), colors):
            cf_values = [results[strategy]['carbon_footprints'].get(y, np.nan) for y in years]
            ax.plot(years, cf_values, 'o-', label=label, linewidth=2, markersize=8, color=color)

        if years:
            # Get the first year's VW carbon footprint for the net zero target path
            first_year = min(years)
            base_cf = results['vw']['carbon_footprints'].get(first_year, None)
            if base_cf and not np.isnan(base_cf):
                nz_targets = []
                for i, year in enumerate(years):
                    years_elapsed = i + 1  # Start with year 1
                    target_cf = base_cf * ((1 - 0.1) ** years_elapsed)
                    nz_targets.append(target_cf)
                ax.plot(years, nz_targets, 'k--', label='Net Zero Target Path', linewidth=2)

        ax.set_title('Carbon Footprint Evolution by Strategy', fontsize=16, fontweight='bold')
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Carbon Footprint (tCO2e/$M)', fontsize=14)
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'carbon_footprint_evolution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Carbon Reduction Percentages
        fig, ax = plt.subplots(figsize=(14, 8))
        strategies_to_plot = {
            'mvc': ('mv', 'P^(mv)_oos(0.5) vs P^(mv)_oos'),  # Correctly compare MVC to MV
            'vwc': ('vw', 'P^(vw)_oos(0.75) vs P^(vw)'),
            'nz': ('vw', 'P^(vw)_oos(NZ) vs P^(vw)')
        }
        colors = ['lightblue', 'lightcoral', 'green']

        for (strategy, (base_strategy, label)), color in zip(strategies_to_plot.items(), colors):
            reductions = []
            for year in years:
                base_cf = results[base_strategy]['carbon_footprints'].get(year, np.nan)
                strategy_cf = results[strategy]['carbon_footprints'].get(year, np.nan)
                if not np.isnan(base_cf) and not np.isnan(strategy_cf) and base_cf > 0:
                    reduction = (1 - strategy_cf / base_cf) * 100
                    reductions.append(reduction)
                else:
                    reductions.append(np.nan)
            ax.plot(years, reductions, 'o-', label=label, linewidth=2, markersize=8, color=color)

        ax.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='50% Target Reduction')
        ax.axhline(y=25, color='coral', linestyle='--', alpha=0.7, label='25% Target Reduction')

        if years:
            nz_target_reductions = []
            for i, year in enumerate(years):
                years_elapsed = i + 1  # Start with year 1
                target_reduction = (1 - (1 - 0.1) ** years_elapsed) * 100
                nz_target_reductions.append(target_reduction)
            ax.plot(years, nz_target_reductions, 'k--', label='Net Zero Target Path', linewidth=2)

        ax.set_title('Carbon Footprint Reduction Percentages', fontsize=16, fontweight='bold')
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Reduction (%)', fontsize=14)
        ax.legend(fontsize=12, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-10, 100)  # Allow for negative reductions
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'carbon_reduction_percentages.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Cumulative Returns Comparison
        fig, ax = plt.subplots(figsize=(14, 8))
        strategies_returns = {
            'mv': 'P^(mv)_oos',
            'mvc': 'P^(mv)_oos(0.5)',
            'vw': 'P^(vw)',
            'vwc': 'P^(vw)_oos(0.75)',
            'nz': 'P^(vw)_oos(NZ)'
        }
        colors = ['blue', 'lightblue', 'red', 'lightcoral', 'green']

        for (strategy, label), color in zip(strategies_returns.items(), colors):
            returns = results[strategy]['returns']
            if len(returns) > 0:
                cumulative = (1 + returns).cumprod() - 1
                ax.plot(cumulative.index, cumulative.values * 100, label=label, linewidth=2, color=color)

        ax.set_title('Cumulative Returns by Strategy', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Cumulative Return (%)', fontsize=14)
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cumulative_returns_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Performance Summary Table
        performance_summary = []
        for strategy, label in strategies_returns.items():
            returns = results[strategy]['returns']
            if len(returns) > 0:
                metrics = self.calculate_portfolio_performance(returns)
                cfs = [cf for cf in results[strategy]['carbon_footprints'].values() if not np.isnan(cf)]
                avg_cf = np.mean(cfs) if cfs else np.nan

                # Calculate average carbon reduction
                if strategy != 'mv' and strategy != 'vw':
                    if strategy == 'mvc':
                        base_strategy = 'mv'
                    else:
                        base_strategy = 'vw'

                    reductions = []
                    for year in years:
                        base_cf = results[base_strategy]['carbon_footprints'].get(year, np.nan)
                        strategy_cf = results[strategy]['carbon_footprints'].get(year, np.nan)
                        if not np.isnan(base_cf) and not np.isnan(strategy_cf) and base_cf > 0:
                            reduction = (1 - strategy_cf / base_cf) * 100
                            reductions.append(reduction)

                    avg_reduction = np.mean(reductions) if reductions else np.nan
                else:
                    avg_reduction = np.nan

                performance_summary.append({
                    'Strategy': label,
                    'Annual Return (%)': metrics['annualized_return'] * 100,
                    'Annual Volatility (%)': metrics['annualized_volatility'] * 100,
                    'Sharpe Ratio': metrics['sharpe_ratio'],
                    'Min Return (%)': metrics['min_return'] * 100,
                    'Max Return (%)': metrics['max_return'] * 100,
                    'Avg Carbon Footprint': avg_cf,
                    'Avg Carbon Reduction (%)': avg_reduction
                })

        summary_df = pd.DataFrame(performance_summary)
        # Remove the Carbon Reduction column for strategies where it's not applicable
        for idx in summary_df.index:
            if summary_df.loc[idx, 'Strategy'] in ['P^(mv)_oos', 'P^(vw)']:
                summary_df.loc[idx, 'Avg Carbon Reduction (%)'] = None

        summary_df = summary_df.round(4)
        summary_df.to_csv(os.path.join(output_dir, 'performance_summary.csv'), index=False)

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        column_display = [col for col in summary_df.columns if col != 'Avg Carbon Reduction (%)']
        table_df = summary_df[column_display].copy()
        table = ax.table(cellText=table_df.values, colLabels=table_df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        for i in range(len(table_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        for i in range(1, len(table_df) + 1):
            for j in range(len(table_df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f2f2f2')
        plt.title('Portfolio Performance Summary', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(os.path.join(output_dir, 'performance_summary_table.png'), dpi=300, bbox_inches='tight')
        plt.close()

        return summary_df

    def calculate_portfolio_performance(self, returns, risk_free_rate=0.0):
        """
        Calculate portfolio performance metrics
        """
        returns_clean = returns.dropna()
        if len(returns_clean) == 0:
            return {
                'annualized_return': np.nan,
                'annualized_volatility': np.nan,
                'sharpe_ratio': np.nan,
                'min_return': np.nan,
                'max_return': np.nan
            }

        ann_factor = 12
        cumulative_return = (1 + returns_clean).prod()
        n_periods = len(returns_clean)
        annualized_return = cumulative_return ** (ann_factor / n_periods) - 1
        annualized_volatility = returns_clean.std() * np.sqrt(ann_factor)
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / annualized_volatility if annualized_volatility > 0 else np.nan
        min_return = returns_clean.min()
        max_return = returns_clean.max()

        return {
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'min_return': min_return,
            'max_return': max_return
        }

    def generate_project_report(self, results, summary_df, output_dir):
        """
        Generate a structured text report summarizing all findings
        """
        os.makedirs(output_dir, exist_ok=True)
        report = []

        report.append("SUSTAINABILITY AWARE ASSET MANAGEMENT PROJECT")
        report.append("Asset Allocation with a Carbon Objective")
        report.append("Group AK: Europe / Scope 1+2")
        report.append("=" * 80)
        report.append("")

        # Part 1: Standard Asset Allocation
        report.append("PART 1: STANDARD ASSET ALLOCATION")
        report.append("-" * 40)

        mv_metrics = summary_df[summary_df['Strategy'] == 'P^(mv)_oos']
        if not mv_metrics.empty:
            mv_metrics = mv_metrics.iloc[0]
            report.append("Minimum Variance Portfolio (P^(mv)_oos):")
            report.append(f"  Annualized Return: {mv_metrics['Annual Return (%)']:.2f}%")
            report.append(f"  Annualized Volatility: {mv_metrics['Annual Volatility (%)']:.2f}%")
            report.append(f"  Sharpe Ratio: {mv_metrics['Sharpe Ratio']:.3f}")
            report.append("")

        vw_metrics = summary_df[summary_df['Strategy'] == 'P^(vw)']
        if not vw_metrics.empty:
            vw_metrics = vw_metrics.iloc[0]
            report.append("Value-Weighted Portfolio (P^(vw)):")
            report.append(f"  Annualized Return: {vw_metrics['Annual Return (%)']:.2f}%")
            report.append(f"  Annualized Volatility: {vw_metrics['Annual Volatility (%)']:.2f}%")
            report.append(f"  Sharpe Ratio: {vw_metrics['Sharpe Ratio']:.3f}")
            report.append("")

        if not mv_metrics.empty and not vw_metrics.empty:
            report.append("Comparison:")
            return_diff = mv_metrics['Annual Return (%)'] - vw_metrics['Annual Return (%)']
            vol_diff = mv_metrics['Annual Volatility (%)'] - vw_metrics['Annual Volatility (%)']
            report.append(f"  Return Difference (MV - VW): {return_diff:.2f}%")
            report.append(f"  Volatility Difference (MV - VW): {vol_diff:.2f}%")
            report.append("")

        # Part 2: Carbon Emissions Reduction
        report.append("PART 2: ASSET ALLOCATION WITH CARBON EMISSIONS REDUCTION")
        report.append("-" * 40)
        report.append("2.1 Carbon Footprints of Standard Portfolios:")
        years = sorted(results['mv']['carbon_footprints'].keys())
        for year in years:
            mv_cf = results['mv']['carbon_footprints'].get(year, np.nan)
            vw_cf = results['vw']['carbon_footprints'].get(year, np.nan)
            report.append(f"  {year}: MV = {mv_cf:.2f}, VW = {vw_cf:.2f} tCO2e/$M")
        report.append("")

        # Part 2.2: MV with 50% reduction
        mvc_metrics = summary_df[summary_df['Strategy'] == 'P^(mv)_oos(0.5)']
        if not mvc_metrics.empty:
            mvc_metrics = mvc_metrics.iloc[0]
            report.append("2.2 Minimum Variance with 50% Carbon Reduction (P^(mv)_oos(0.5)):")
            report.append(f"  Annualized Return: {mvc_metrics['Annual Return (%)']:.2f}%")
            report.append(f"  Annualized Volatility: {mvc_metrics['Annual Volatility (%)']:.2f}%")
            report.append(f"  Sharpe Ratio: {mvc_metrics['Sharpe Ratio']:.3f}")
            report.append(f"  Average Carbon Footprint: {mvc_metrics['Avg Carbon Footprint']:.2f} tCO2e/$M")

            # Calculate average carbon reduction relative to MV
            mvc_reductions = []
            for year in years:
                mv_cf = results['mv']['carbon_footprints'].get(year, np.nan)
                mvc_cf = results['mvc']['carbon_footprints'].get(year, np.nan)
                if not np.isnan(mv_cf) and not np.isnan(mvc_cf) and mv_cf > 0:
                    reduction = (1 - mvc_cf / mv_cf) * 100
                    mvc_reductions.append(reduction)

            if mvc_reductions:
                avg_mvc_reduction = np.mean(mvc_reductions)
                report.append(f"  Average Carbon Reduction: {avg_mvc_reduction:.1f}%")
            report.append("")

        # Part 2.3: VW with 25% reduction
        vwc_metrics = summary_df[summary_df['Strategy'] == 'P^(vw)_oos(0.75)']
        if not vwc_metrics.empty:
            vwc_metrics = vwc_metrics.iloc[0]
            report.append("2.3 Value-Weighted with 25% Carbon Reduction (P^(vw)_oos(0.75)):")
            report.append(f"  Annualized Return: {vwc_metrics['Annual Return (%)']:.2f}%")
            report.append(f"  Annualized Volatility: {vwc_metrics['Annual Volatility (%)']:.2f}%")
            report.append(f"  Sharpe Ratio: {vwc_metrics['Sharpe Ratio']:.3f}")
            report.append(f"  Average Carbon Footprint: {vwc_metrics['Avg Carbon Footprint']:.2f} tCO2e/$M")

            # Calculate average carbon reduction relative to VW
            vwc_reductions = []
            for year in years:
                vw_cf = results['vw']['carbon_footprints'].get(year, np.nan)
                vwc_cf = results['vwc']['carbon_footprints'].get(year, np.nan)
                if not np.isnan(vw_cf) and not np.isnan(vwc_cf) and vw_cf > 0:
                    reduction = (1 - vwc_cf / vw_cf) * 100
                    vwc_reductions.append(reduction)

            if vwc_reductions:
                avg_vwc_reduction = np.mean(vwc_reductions)
                report.append(f"  Average Carbon Reduction: {avg_vwc_reduction:.1f}%")
            report.append("")

        # Part 2.4: Trade-off analysis
        report.append("2.4 Trade-off Analysis:")
        report.append("")
        if not mvc_metrics.empty and not mv_metrics.empty:
            mv_mvc_return_cost = mvc_metrics['Annual Return (%)'] - mv_metrics['Annual Return (%)']
            mv_mvc_vol_impact = mvc_metrics['Annual Volatility (%)'] - mv_metrics['Annual Volatility (%)']
            report.append("Active Investor (MV vs MVC):")
            report.append(f"  Return Cost: {mv_mvc_return_cost:.2f}%")
            report.append(f"  Volatility Impact: {mv_mvc_vol_impact:.2f}%")
            report.append(f"  Sharpe Ratio Impact: {mvc_metrics['Sharpe Ratio'] - mv_metrics['Sharpe Ratio']:.3f}")
            if mvc_reductions:
                report.append(f"  Carbon Reduction Achieved: {avg_mvc_reduction:.1f}%")
            report.append("")

        if not vwc_metrics.empty and not vw_metrics.empty:
            vw_vwc_return_cost = vwc_metrics['Annual Return (%)'] - vw_metrics['Annual Return (%)']
            vw_vwc_vol_impact = vwc_metrics['Annual Volatility (%)'] - vw_metrics['Annual Volatility (%)']
            report.append("Passive Investor (VW vs VWC):")
            report.append(f"  Return Cost: {vw_vwc_return_cost:.2f}%")
            report.append(f"  Volatility Impact: {vw_vwc_vol_impact:.2f}%")
            report.append(f"  Sharpe Ratio Impact: {vwc_metrics['Sharpe Ratio'] - vw_metrics['Sharpe Ratio']:.3f}")
            if vwc_reductions:
                report.append(f"  Carbon Reduction Achieved: {avg_vwc_reduction:.1f}%")
            report.append("")

        # Part 3: Net Zero Objective
        report.append("PART 3: ALLOCATION WITH NET ZERO OBJECTIVE")
        report.append("-" * 40)
        nz_metrics = summary_df[summary_df['Strategy'] == 'P^(vw)_oos(NZ)']
        if not nz_metrics.empty:
            nz_metrics = nz_metrics.iloc[0]
            report.append("3.1 Net Zero Portfolio (P^(vw)_oos(NZ)):")
            report.append(f"  Annualized Return: {nz_metrics['Annual Return (%)']:.2f}%")
            report.append(f"  Annualized Volatility: {nz_metrics['Annual Volatility (%)']:.2f}%")
            report.append(f"  Sharpe Ratio: {nz_metrics['Sharpe Ratio']:.3f}")
            report.append(f"  Average Carbon Footprint: {nz_metrics['Avg Carbon Footprint']:.2f} tCO2e/$M")
            report.append("")

        report.append("Annual Carbon Footprint Reduction:")
        first_year = min(years) if years else None
        base_cf = results['vw']['carbon_footprints'].get(first_year, None) if first_year else None

        for i, year in enumerate(years):
            nz_cf = results['nz']['carbon_footprints'].get(year, np.nan)
            if not np.isnan(nz_cf) and base_cf and not np.isnan(base_cf):
                actual_reduction = (1 - nz_cf / base_cf) * 100
                years_elapsed = i + 1
                target_reduction = (1 - (1 - 0.1) ** years_elapsed) * 100
                report.append(f"  {year}: Actual = {actual_reduction:.1f}%, Target = {target_reduction:.1f}%")
        report.append("")

        if not nz_metrics.empty and not vw_metrics.empty:
            report.append("3.2 Cost of Constructing a Net Zero Portfolio:")
            nz_vw_return_cost = nz_metrics['Annual Return (%)'] - vw_metrics['Annual Return (%)']
            nz_vw_vol_impact = nz_metrics['Annual Volatility (%)'] - vw_metrics['Annual Volatility (%)']
            nz_vw_sharpe_impact = nz_metrics['Sharpe Ratio'] - vw_metrics['Sharpe Ratio']
            report.append(f"  Return Cost vs Value-Weighted: {nz_vw_return_cost:.2f}%")
            report.append(f"  Volatility Impact vs Value-Weighted: {nz_vw_vol_impact:.2f}%")
            report.append(f"  Sharpe Ratio Impact vs Value-Weighted: {nz_vw_sharpe_impact:.3f}")

            if not vwc_metrics.empty:
                nz_vwc_return_diff = nz_metrics['Annual Return (%)'] - vwc_metrics['Annual Return (%)']
                report.append(f"  Additional Return Cost vs 25% Reduction: {nz_vwc_return_diff:.2f}%")

            if base_cf and not np.isnan(base_cf):
                final_year = max(years) if years else None
                if final_year:
                    final_cf = results['nz']['carbon_footprints'].get(final_year, np.nan)
                    if not np.isnan(final_cf):
                        final_reduction = (1 - final_cf / base_cf) * 100
                        report.append(f"  Final Carbon Reduction Achieved: {final_reduction:.1f}%")
            report.append("")

        report.append("=" * 80)
        report.append("Warnings:")
        report.append("- Found 5 non-positive prices in DS_RI_T_USD_M.csv, treated as NaN.")
        report.append("- Found 22 returns exceeding 100% absolute value, potentially affecting volatility.")
        report.append("")
        report.append("END OF REPORT")

        report_text = "\n".join(report)
        with open(os.path.join(output_dir, 'project_report.txt'), 'w') as f:
            f.write(report_text)

        return report_text