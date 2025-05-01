# Questo file fa sì che Python riconosca la directory come un package
# Import principali per rendere più facile accedere ai moduli
from .CarbonAwarePortfolio import CarbonAwarePortfolio
from .Data_SetUP import Initializer
from .Standard_Asset_Allocation import run_portfolio_optimization
from .Value_Weighted_Portfolio import calculate_value_weighted_portfolio, plot_cumulative_returns, compare_portfolio_performance