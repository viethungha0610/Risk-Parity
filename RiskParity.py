# Necessary packages for the programme to run
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import scipy.linalg as la
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
import requests
import time

# Parent class
class RiskParity():
    """
    This is the main Risk Parity class, where main operations such as calculating the risk function
    """
    def __init__(self, 
                 cov_mat,
                 assets=None,
                 w_guess=None):
        """Inserting data into class"""
        """Following guidance in Braga (2016), the class will by default favour optimisation via SQP"""
        """
        Args:
            cov_mat (array): the return covariance matrix 
            assets (list): list of asset names or ticker 
            w_guess (list or array): initial guess for Risk Parity asset allocation, if not specified the default will be equal weights 

        Raises:
            ValueError: if the number of assets is not equal to the number of rows/columns of the covariance matrix.
        """
        self.cov_mat = cov_mat
        self.assets = assets
        if w_guess is None:
            w_guess=np.full((self.cov_mat.shape[0]), 1/self.cov_mat.shape[0])
            self.w_guess = w_guess
        self.w_guess = w_guess
        # Error handling: Making sure len(assets) == len(cov_mat.shape[0])
        if len(self.assets) != self.cov_mat.shape[0]:
            raise ValueError('Number of assets must be equal to number of rows/columns of the covariance matrix')

    def risk_func(self, w=None):
        """Main risk function to minimise - convex risk function. In this class, this is a helper function.
        Args:
            w (list or array): initial guess for Risk Parity asset allocation, if not specified the default will be equal weights (list or array)

        Returns:
            risk_func (float): the calculated risk function
        """
        # Start off with vector x
        if w is None:
            w = self.w_guess
        b_T = 1/len(w)
        w_T = w.T
        x = w / (np.sqrt(w_T.dot(self.cov_mat).dot(w)))
        # Then the main function
        x_T = x.T
        risk_func = 0.5*x_T.dot(self.cov_mat).dot(x) - b_T*(np.sum(np.log(x)))
        self.risk_func_ = risk_func
        return risk_func
    
    def optimize(self, assets=None, method='SLSQP'):
        """This method minimize the risk function (risk_func) and displays a Risk Parity asset allocation.

        Args:
            assets (list): list of name of assets, if not specified then using the same list of assets when the class instance is initiated. 
            method (str): method for optimization, default is Sequential Least Squares Programming (SLSQP). 
            For full list of optimization method,
                refer to [https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html]
        
        Attributes:
            allocation_ (list): the Risk Parity asset allocation 
            allocation_df_ (pandas DataFrame): a pandas DataFrame with 2 columns: 1. asset names and 2. their Risk Parity allocation 
            minimised_val_ (float): the minimised value of the convex risk function 
        """
        self.method = method
        if assets is None:
            assets=self.assets
        opti_result = minimize(self.risk_func, self.w_guess, 
                               method=self.method)
        allocation = opti_result.x / sum(opti_result.x)
        print('Minimised convex risk function value: {:.4f}'.format(opti_result.fun))
        allocation_df = pd.DataFrame({'Assets':assets, 
                                      'Allocation':np.round(allocation, 4)})
        print(allocation_df)
        self.allocation_ = allocation
        self.allocation_df_ = allocation_df
        self.minimised_val_ = opti_result.fun
    
    def cal_risk_stats(self, assets=None):
        """This method calculates marginal risk contribution (MRC), risk contribution (RC)
        and relative risk contribution (RRC) of the ith asset.
        
        Args:
            assets (list or array): list of name of assets, if not specified then using the same list of assets when the class instance is initiated. 

        Attributes:
            MRC_ (array): marginal risk contribution of the ith asset 
            RC_ (array): risk contribution of the ith asset 
            RRC_ (array): relative risk contribution of the ith asset 
        """
        if assets is None:
            assets=self.assets
        w_rpp = self.allocation_
        w_rpp_T = w_rpp.T
        portfolio_vol = np.sqrt(w_rpp_T.dot(self.cov_mat).dot(w_rpp))
        
        # Marginal risk contribution (MRC)
        MRC_num = self.cov_mat.dot(w_rpp)
        MRC_denom = portfolio_vol
        MRC = []
        for val, i in zip(MRC_num, range(len(assets))):
            MRC.append(MRC_num[i] / MRC_denom)
        self.MRC_ = MRC
        
        # Risk contribution (RC)
        RC_component = self.cov_mat.dot(w_rpp)
        RC_denom = portfolio_vol
        RC = []
        for i in range(len(assets)):
            RC.append((w_rpp[i]*RC_component[i]) / RC_denom)
        self.RC_ = RC
        
        # Relative risk contribution (RRC)
        RRC_component = self.cov_mat.dot(w_rpp)
        RRC_denom = portfolio_vol**2
        RRC = []
        for i in range(len(assets)):
            RRC.append((w_rpp[i]*RRC_component[i]) / RRC_denom)
        self.RRC_ = RRC
    
    def visualise_risk_stats(self, display=False):
        """
        This method visualises the risk statistics from the cal_risk_stats method above.
        
        Attributes:
            MRC_fig_ (matplotlib fig): matplotlib figure of the marginal risk contribution (MRC) of all assets in the portfolios
            RC_fig_ (matplotlib fig): matplotlib figure of the risk contribution (RC) of all assets in the portfolios
            RRC_fig_ (matplotlib fig): matplotlib figure of the relative risk contribution (RRC) of all assets in the portfolios
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        # MRC_i visualisation
        MRC_fig = sns.barplot(y=self.MRC_, x=self.assets)
        MRC_fig.set_title('Marginal risk contribution (MRC) to total portfolio risk')
        self.MRC_fig_ = MRC_fig
        if display==True:
            plt.show()
        else:
            pass
        # RC_i visualisation
        RC_fig = sns.barplot(y=self.RC_, x=self.assets)
        RC_fig.set_title('Risk contribution (RC) to total portfolio risk')
        self.RC_fig_ = RC_fig
        if display==True:
            plt.show()
        else:
            pass
        # RRC_i visualisation
        RRC_fig = sns.barplot(y=self.RRC_, x=self.assets)
        RRC_fig.set_title('Relative risk contribution (RRC) to total portfolio risk')
        self.RRC_fig_ = RRC_fig
        if display==True:
            plt.show()
        else:
            pass
        
    def out_excel(self, directory=None):
        """This method outputs an excel file with the asset allocation.
        
        Args:
            directory (str): the directory in which to output the Excel file.
                If the directory is not specified, the method will output the file in the current working directory"""
        import os
        self.directory = directory
        if directory is None:
            self.directory = os.getcwd()
        allocation_to_excel = self.allocation_df_.to_excel(f'{self.directory}/RP_allocation.xlsx', index=False)
        return allocation_to_excel

# Child class of RiskParity
class NonConvexRiskParity(RiskParity):
    """
    This child class inherits all of methods from the parent RiskParity class, except risk_func() and optimize()
        to account for the non-convex formulation of the optimization problem.
    """
    def risk_func(self, w=None):
        """This is the modified optimise function for non-convex problem formulation
        Args:
            w (list or array): initial guess for Risk Parity asset allocation, if not specified the default will be equal weights

        Returns:
            risk_func (float): the calculated risk function
        """
        if w is None:
            w = self.w_guess
        n = len(w)
        risk_budget = 1 / n
        risks = w * (self.cov_mat.dot(w))
        norm_risks = risks / risk_budget
        g = np.tile(norm_risks, n) - np.repeat(norm_risks, n)
        """Example: assuming n = 2 and norm_risk is [a, b, c]
                                np.tile -> [a, b, c, a, b, c]
                                np.repeat -> [a, a, b, b, c, c]
                                np.tile - np.repeat
                                i = 1, 2, 3 ;  j = 1
                                Taking all the 1, 2, 3 minus the 1
                                i = 1, 2, 3 ;  j = 2
                                and so on ...
                                i = 1, 2, 3 ;  j = 3
                                By taking np.tile - np.repeat, one gets all the permutations of N assets taken 2 at a time (pairs of i and j) 
                                """
        return np.sum(g**2) # This is the main function to minimise, a SUM OF SQUARES, (NOT SQUARE OF SUM) 
    
    def optimize(self, assets=None, method='BFGS'):
        """This method minimize the risk function (risk_func) and displays a Risk Parity asset allocation.

        Args:
            assets (list): list of name of assets, if not specified then using the same list of assets when the class instance is initiated. 
            method (str): method for optimization, default is the Broyden–Fletcher–Goldfarb–Shanno (BFGS). 
            For full list of optimization method,
                refer to [https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html]
        
        Attributes:
            allocation_ (list): the Risk Parity asset allocation 
            allocation_df_ (pandas DataFrame): a pandas DataFrame with 2 columns: 1. asset names and 2. their Risk Parity allocation 
            minimised_val_ (float): the minimised value of the convex risk function"""
        self.method = method
        if assets is None:
            assets=self.assets
        opti_result = minimize(self.risk_func, self.w_guess, 
                               method=self.method)
        allocation = opti_result.x / sum(opti_result.x)
        print('Minimised non-convex risk function value: {:.4f}'.format(opti_result.fun))
        allocation_df = pd.DataFrame({'Assets':assets, 
                                      'Allocation':np.round(allocation, 4)})
        print(allocation_df)
        # Important attributes below
        self.allocation_ = allocation
        self.allocation_df_ = allocation_df
        self.minimised_val_ = opti_result.fun


# The preprocessing part of the programme. Receiving csv -> Returning cov_mat
class DataPreprocessor():
    """This class aims to turn an imported csv file into a return covariance matrix (Matrix A), which can be used as an input for the RiskParity or NonConvexRiskParity classes above"""
    import pandas as pd
    import numpy as np
    def __init__(self, df=None):
        """
        Initiator

        Args:
            df (pandas DataFrame): the DataFrame to preprocess 
                It should have the following format:
                    1 column with the dates
                    n columns with the asset prices, with correct date-asset price combination
        """
        self.df = df
        
    def transform(self, df=None):
        """
        This method transforms the correctly formatted pandas DataFrame into a covariance matrix to be as an input for the RiskParity or NonConvexRiskParity classes above.
        
        Attributes:
            assets_ (list): the names of the assets i.e. the column names of the formatted pandas DataFrame 

        Returns:
            cov_mat (array): the return covariance matrix, which can be used as an input for the RiskParity or NonConvexRiskParity classes above 
        """
        print("Transforming data into usable format (Covariance Matrix)")
        if df is None:
            df = self.df
        # Drop NaN values to make sure asset time series have the same # of observatives
        df = df.dropna()
        # Filter by numeric data types
        numeric_cols = list(df.loc[:, df.dtypes==np.float64].columns) + list(df.loc[:, df.dtypes==np.int64].columns)
        price_data = df[numeric_cols]
        assets = price_data.columns.tolist()
        returns_df = (price_data.iloc[1:,:].values / price_data.iloc[:-1,:] - 1)
        excess_returns_df = returns_df - returns_df.mean()
        excess_returns = np.array(excess_returns_df)
        excess_returns_T = excess_returns.T
        cov_mat = excess_returns_T.dot(excess_returns) / (excess_returns.shape[0] - 1)
        # Getting the name of the assets through this attribute
        self.assets_ = assets
        print("Data transformation complete.")
        return cov_mat

class GetRapidAPIData():
    """
    This class imports data from RapidAPI's Yahoo Finance API, then outputs a time series DataFrame with date and prices of selected assets.
    """
    def __init__(self, api_key, assets, start, finish, freq):
        """
        Initiator

        Args:
            api_key (str): one's own API key. For instructions, refer to the Instruction notebook.
            assets (list): List of ticker, must be valid and available on Yahoo finance
            start (str): start date of time series, in the format of YYYY-MM-DD
            finish (str): finish date of time series, in the format of YYYY-MM-DD
            freq (str): the frequency of data:
                  '1d': daily
                  '1wk': weekly
                  '1mo': monthly 
        """
        self.api_key = api_key
        self.assets = assets # List of assets, to be looped later
        self.start = start
        self.finish = finish
        self.freq = freq
    
    def time_to_unix(self, date, pattern='%Y.%m.%d'):
        """
        Helper method: converting real time to unix timestamps

        Args:
            date (str): date, in the format of YYYY-MM-DD by default
            pattern (str): date pattern to process, by default it is '%Y.%m.%d'

        Returns:
            unix (int): equivalent timestamp item of date arg. 
        """
        import time
        unix = int(time.mktime(time.strptime(date, pattern)))
        return unix

    def get_api_data(self, api_key, asset, start, finish, freq):
        """
        Helper method: getting raw data from RapidAPI's Yahoo Finance API
        
        Returns:
            response_json (json): a json file containing historical data 
        """
        import requests
        import json
        start = self.time_to_unix(self.start, pattern='%Y.%m.%d')
        finish = self.time_to_unix(self.finish, pattern='%Y.%m.%d')
        url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v2/get-historical-data"
        querystring = {"frequency":self.freq, "filter":"history", "period1":start,
                       "period2":finish,"symbol":asset}
        headers = {
            'x-rapidapi-host': "apidojo-yahoo-finance-v1.p.rapidapi.com",
            'x-rapidapi-key': self.api_key
            }
        response = requests.request("GET", url, headers=headers, params=querystring)
        response_json = json.loads(response.text)
        return response_json

    def make_time_series(self, response_json, asset=None):
        """
        Helper method: turning the response_json file into a time series DataFrame (with date and price columns)

        Args:
            response_json (json): a json file containing historical data
            asset (str, optional): name of asset / ticker

        Returns:
            times_series_df (pandas DataFrame): time series DataFrame
        """
        from datetime import datetime
        import pandas as pd
        date_list_unix = []
        adj_close_list = []
        for index in range(len(response_json['prices'])):
            if len(response_json['prices'][index]) == 7:
                adj_close = response_json['prices'][index]['adjclose']
                adj_close_list.append(adj_close)
                date = response_json['prices'][index]['date']
                date_list_unix.append(date)
        date_list = [datetime.fromtimestamp(unix_date) for unix_date in date_list_unix] # Convert unix to human date
        date_list_ymd = [item.date() for item in date_list] # Getting date (years, months and days) only
        time_series_df = pd.DataFrame({'Date':date_list_ymd, f'{asset}':adj_close_list})
        return time_series_df

    def prep_data(self, api_key=None, assets=None, 
                  start=None, finish=None, freq=None): 
        """
        Main function, using the helper methods. This method gets the data from the API, processes it and returns a DataFrame:
            1. a date column
            2. N price columns
            3. n rows each containing an observation 
            --> Shape: (n ,1+N)

        Returns:
            merged_df (pandas DataFrame): the combined time series DataFrame
        """
        if api_key is None:
            api_key = self.api_key
        if assets is None:
            assets = self.assets
        if start is None:
            start = self.start
        if finish is None:
            finish = self.finish
        if freq is None:
            freq = self.freq
        time_series_dict = {}
        time_series_list = []
        for asset in self.assets: 
            # Go through the motions
            print(f"Getting data for {asset} ...")
            response_dict = self.get_api_data(api_key=self.api_key, asset=asset, freq=self.freq, start=self.start, finish=self.finish)
            time_series_df = self.make_time_series(response_json=response_dict, asset=asset)
            time_series_dict[f"{asset}_df"] = time_series_df # For example MSFT_df is a time series DataFrame with 2 cols: date and price
            # Now we got a time series dictionary with security name key and their time series DataFrame as value
            time_series_list.append(time_series_dict[f"{asset}_df"])
        print("Pulling together a consolidated DataFrame ...")
        merged_df = pd.concat(time_series_list, axis=1, join='outer')
        merged_df = merged_df.reindex(index=merged_df.index[::-1])
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
        print("Relevant data has been downloaded successfully.")
        return merged_df