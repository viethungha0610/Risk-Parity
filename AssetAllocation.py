import pandas as pd
import numpy as np
import sympy
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
    from scipy.optimize import minimize
    import numpy as np
    def __init__(self, 
                 cov_mat,
                 assets=None,
                 w_guess=None):
        """Inserting data into class"""
        """Following guidance in Braga (2016), the class will by default favour optimisation via SQP"""
        # Making sure len(assets) == len(cov_mat.shape[0])
        self.cov_mat = cov_mat
        self.assets = assets
        if w_guess is None:
            w_guess=np.full((self.cov_mat.shape[0]), 1/self.cov_mat.shape[0])
            self.w_guess = w_guess
        self.w_guess = w_guess
        # Error handling
        if len(self.assets) != self.cov_mat.shape[0]:
            raise ValueError('Number of assets must be equal to number of rows/columns in the covariance matrix')

    def risk_func(self, w=None):
        """Main function to minimise"""
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
        """Returns an risk parity asset allocation"""
        """Attribute: 
        - allocation_ -> Risk Parity allocation
        - minimised_val_ -> Minimised risk function value"""
        self.method = method
        if assets is None:
            assets=self.assets
        opti_result = minimize(self.risk_func, self.w_guess, 
                               method=self.method)
        allocation = opti_result.x / sum(opti_result.x)
        print('Minimised convex risk function value: {:.4f}'.format(opti_result.fun))
        allocation_df = pd.DataFrame({'Assets':assets, 
                                      'Allocation':np.round(allocation, 4)})
        display(allocation_df)
        self.allocation_ = allocation
        self.allocation_df_ = allocation_df
        self.minimised_val_ = opti_result.fun
    
    def cal_risk_stats(self, assets=None):
        """A post-optimisation method"""
        """Calculate marginal risk contribution (MRC), risk contribution (RC)
        and relative risk contribution (RRC) of the ith asset"""
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
    
    def visualise_risk_stats(self):
        import seaborn as sns
        import matplotlib.pyplot as plt
        # MRC_i visualisation
        MRC_fig = sns.barplot(y=self.MRC_, x=self.assets)
        MRC_fig.set_title('Marginal risk contribution to total portfolio risk')
        self.MRC_fig_ = MRC_fig
        plt.show()
        # RC_i visualisation
        RC_fig = sns.barplot(y=self.RC_, x=self.assets)
        RC_fig.set_title('Risk contribution (RC) to total portfolio risk')
        self.RC_fig_ = RC_fig
        plt.show()
        # RRC_i visualisation
        RRC_fig = sns.barplot(y=self.RRC_, x=self.assets)
        RRC_fig.set_title('Relative risk contribution (RRC) to total portfolio risk')
        self.RRC_fig_ = RRC_fig
        plt.show()
        
    def out_excel(self, directory=None):
        """Outputs an excel file with the asset allocation.
        If the directory is not specified, the method will output the file in the current working directory"""
        import os
        self.directory = directory
        if directory is None:
            self.directory = os.getcwd()
        allocation_to_excel = self.allocation_df_.to_excel(f'{self.directory}/RP_allocation.xlsx', index=False)
        return allocation_to_excel

# Child class of RiskParity
class NonConvexRP(RiskParity):
    def risk_func(self, w=None):
        """Modified optimise function for non-convex problem formulation"""
        if w is None:
            w = self.w_guess
        n = len(w)
        risk_budget = 1 / n
        risks = w * (self.cov_mat.dot(w))
        norm_risks = risks / risk_budget
        g = np.tile(norm_risks, n) - np.repeat(norm_risks, n)
        return np.sum(g**2) # This is the main function to minimise
    
    def optimize(self, assets=None, method='BFGS'):
        self.method = method
        if assets is None:
            assets=self.assets
        opti_result = minimize(self.risk_func, self.w_guess, 
                               method=self.method)
        allocation = opti_result.x / sum(opti_result.x)
        print('Minimised non-convex risk function value: {:.4f}'.format(opti_result.fun))
        allocation_df = pd.DataFrame({'Assets':assets, 
                                      'Allocation':np.round(allocation, 4)})
        display(allocation_df)
        # Important attributes below
        self.allocation_ = allocation
        self.allocation_df_ = allocation_df
        self.minimised_val_ = opti_result.fun

# Designing back-end of the programme. Receiving csv -> Returning cov_mat
class PrepDataRP():
    """This class aims to turn an imported csv file into a return covariance matrix (Matrix A)"""
    import pandas as pd
    import numpy as np
    def __init__(self, df=None):
        self.df = df
        
    def transform(self, df=None):
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
        return cov_mat

class api_data():
    def __init__(self, api_key, assets, start, finish, freq):
        self.api_key = api_key
        self.assets = assets # List of assets, to be looped later
        self.start = start
        self.finish = finish
        self.freq = freq
    
    def time_to_unix(self, date, pattern='%Y.%m.%d'):
        import time
        unix = int(time.mktime(time.strptime(date, pattern)))
        return unix

    def get_api_data(self, api_key, asset, freq, start, finish):
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

    def make_time_series(self, response_json, asset=None): # NEED TO FILL OUT THE ARGS
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
                  start=None, finish=None, freq=None): # Combined function, using the helper function
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
            response_json = self.get_api_data(api_key=self.api_key, asset=asset, freq=self.freq, start=self.start, finish=self.finish)
            time_series_df = self.make_time_series(response_json=response_json, asset=asset)
            time_series_dict[f"{asset}_df"] = time_series_df #For example MSFT_df is a time series dataframe with 2 cols: date and price
            # Now we got a time series dictionary with security name key and their time series dataframe as value
            time_series_list.append(time_series_dict[f"{asset}_df"])
        merged_df = pd.concat(time_series_list, axis=1, join='outer')
        merged_df = merged_df.reindex(index=merged_df.index[::-1])
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
        return merged_d