# Operating system imports
import os

# Pandas imports
import pandas as pd

# Numpy imports 
import numpy as np

# Matplotlib imports
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Get the current working directory. 
cwd = os.getcwd()






class PortfolioBalancer:
    def __init__(self) -> None:
        coin_path = f"{cwd}\\HistoricalData\\coin_data.csv"    
        self.coin_data = pd.read_csv(coin_path)

        # Remove the first row. 
        self.coin_data = self.coin_data.iloc[1:]

        # Preprocessing. These are the dates that are faulty
        self.coin_data = self.coin_data[~self.coin_data['date'].str.endswith('04:00:00')]
        self.coin_data = self.coin_data[~self.coin_data['date'].str.endswith('00:00:00')]

        # Reverse the rows.
        self.coin_data = self.coin_data.iloc[::-1]
        # Set dca_info to an empty dataframe when the class is initialized. 
        self.dca_info = pd.DataFrame()
        # Set metrics_df to an empty dataframe when the class is initialized.
        self.metrics_df = pd.DataFrame()

    '''------------------------------- Setters & Getters -------------------------------'''
    def set_dca_info(self, dca_info: pd.DataFrame):
        self.dca_info = dca_info
    '''-------------------------------'''
    def get_dca_info(self) -> pd.DataFrame:
        return self.dca_info
    '''-------------------------------'''
    def set_metrics_df(self, metrics_df) -> None:
        self.metrics_df = metrics_df
    '''-------------------------------'''
    def get_metrics_df(self) -> pd.DataFrame:
        return self.metrics_df
    '''-------------------------------'''
    def calculate_dca_strategy(self, coin_column: str, start_date: str, end_date: str, investment_amount: float, interval_days: int):
        """
        Calculate the total amount of a specific cryptocurrency acquired through Dollar-Cost Averaging (DCA) strategy.

        Parameters:
            - coin_column (str): Column name specifying the price of the coin of interest.
            - start_date (str): Start date of the DCA strategy in 'YYYY-MM-DD' format.
            - end_date (str): End date of the DCA strategy in 'YYYY-MM-DD' format.
            - investment_amount (float): Amount to invest in each DCA interval.
            - interval_days (int): Number of days between each DCA interval.

        Returns:
            - dca_info (DataFrame): DataFrame containing information about each DCA interval, including Total Invested (USD) and Drawdown.
        """
        # Convert the 'date' column to datetime format
        self.coin_data['date'] = pd.to_datetime(self.coin_data['date'])

        # Filter the self.coin_data for the specified date range
        self.coin_data = self.coin_data[(self.coin_data['date'] >= start_date) & (self.coin_data['date'] <= end_date)]
        self.coin_data = self.coin_data.reset_index()

        # Calculate the number of DCA intervals
        num_intervals = len(self.coin_data)

        # Initialize lists to store DCA information
        coin_acquired = []
        total_coin_accumulated = []
        total_value_at_interval = []
        total_invested_list = []
        drawdown_list = []

        # Calculate DCA information for each interval
        total_accumulated = 0.0
        total_investment = 0.0
        for i, price in enumerate(self.coin_data[coin_column]):
            if i % interval_days == 0:
                coin_amount = investment_amount / price
                total_accumulated += coin_amount
                total_investment += investment_amount

            total_invested_list.append(total_investment)
            coin_acquired.append(coin_amount)
            total_coin_accumulated.append(total_accumulated)
            total_value_at_interval.append(total_accumulated * price)
            
            # Calculate drawdown
            max_value = max(total_value_at_interval)
            drawdown = (total_value_at_interval[-1] - max_value) / max_value
            drawdown_list.append(drawdown)

        # Create a DataFrame with DCA information, including Drawdown
        dca_info = pd.DataFrame({
            'Date': self.coin_data['date'],
            'Amount Acquired': coin_acquired,
            'Total Coin Accumulated': total_coin_accumulated,
            'Total Value (USD)': total_value_at_interval,
            'Total Invested (USD)': total_invested_list,
            'Drawdown': drawdown_list
        })
        
        return dca_info

    
    '''-------------------------------'''
    def compare_portfolio_metrics(self, dca_info_list: list):
        """
        Compare and print portfolio metrics for a list of portfolios (dca_info_list).

        Parameters:
            - dca_info_list (list of DataFrames): List of DataFrames, each containing DCA strategy information for a portfolio.

        Returns:
            - metrics_df (DataFrame): DataFrame displaying metrics for all portfolios in the list.
        """


        # Initialize lists to store metrics for each portfolio
        total_invested_list = []
        total_value_list = []  # To display Total Value in dollars
        total_return_list = []
        total_return_percentage_list = []
        max_drawdown_list = []
        portfolio_volatility_list = []

        # Iterate through each portfolio DataFrame in the list
        for i, dca_info in enumerate(dca_info_list):
            # Extract metrics for the current portfolio
            total_invested = round(dca_info['Total Invested (USD)'].iloc[-1], 2)
            total_value = round(dca_info['Total Value (USD)'].iloc[-1], 2)
            total_return = round(total_value - total_invested, 2)
            total_return_percentage = round((total_return / total_invested) * 100, 2)
            max_drawdown = round(min(dca_info['Drawdown']) * 100, 2)  # Convert to percentage
            portfolio_volatility = round(self.calculate_portfolio_volatility(dca_info) * 100, 2)  # Convert to percentage

            # Append metrics to respective lists
            total_invested_list.append(total_invested)
            total_value_list.append(total_value)
            total_return_list.append(total_return)
            total_return_percentage_list.append(total_return_percentage)
            max_drawdown_list.append(max_drawdown)
            portfolio_volatility_list.append(portfolio_volatility)

        # Create a DataFrame to display the metrics for all portfolios
        metrics_df = pd.DataFrame({
            'Portfolio': [f'Portfolio {i+1}' for i in range(len(dca_info_list))],
            'Total Invested ($)': total_invested_list,
            'Total Value ($)': total_value_list,
            'Total Return ($)': total_return_list,
            'Total Return Percentage': total_return_percentage_list,
            'Maximum Drawdown (%)': max_drawdown_list,
            'Portfolio Volatility (%)': portfolio_volatility_list
        })

        return metrics_df

    '''-------------------------------'''
    def calculate_portfolio_volatility(self):
    
        """
        Calculate the volatility (standard deviation of daily returns) of a portfolio's total value.

        Parameters:
            - dca_info (DataFrame): DataFrame containing historical information about the portfolio,
            including 'Date' and 'Total Value (USD)' columns.

        Returns:
            - portfolio_volatility (float): The calculated portfolio volatility.
        """
        # Convert the 'Date' column to datetime if it's not already in datetime format
        self.dca_info['Date'] = pd.to_datetime(self.dca_info['Date'])

        # Calculate daily returns
        daily_returns = self.dca_info['Total Value (USD)'].pct_change().dropna()

        # Calculate the standard deviation of daily returns (volatility)
        portfolio_volatility = np.std(daily_returns)

        return portfolio_volatility

    '''-------------------------------'''
    def calculate_and_print_metrics(self):
        """
        Calculate and print the total return in percentage, maximum drawdown, and portfolio volatility based on the provided DCA information.

        Parameters:
            - dca_info (DataFrame): DataFrame containing DCA strategy information, including Total Value (USD),
            Total Invested (USD), and Drawdown.

        Returns:
            None (prints the total return percentage, maximum drawdown, and portfolio volatility)
        """
        # Extract the Total Value and Total Invested at the end of the period
        total_value_end = self.dca_info['Total Value (USD)'].iloc[-1]
        total_invested_end = self.dca_info['Total Invested (USD)'].iloc[-1]

        # Calculate the total return percentage
        total_return_percentage = ((total_value_end - total_invested_end) / total_invested_end) * 100

        # Calculate the maximum drawdown
        max_drawdown = min(self.dca_info['Drawdown'])  # Drawdown is stored as negative values

        # Calculate portfolio volatility
        portfolio_volatility = self.calculate_portfolio_volatility()

        # Print the total return percentage, maximum drawdown, and portfolio volatility
        print(f"Total Return Percentage: {total_return_percentage:.2f}%")
        print(f"Maximum Drawdown: {max_drawdown * 100:.2f}%")
        print(f"Portfolio Volatility: {portfolio_volatility * 100:.2f}%")
    '''-------------------------------'''
    '''-------------------------------'''
    '''------------------------------- Utilities -------------------------------'''
    # Used to format the y-axis in percentage form. 
    def percentage_formatter(self, x, pos):
        return f"{x * 100:.2f}"
    '''-------------------------------'''
    '''-------------------------------'''
    '''-------------------------------'''
