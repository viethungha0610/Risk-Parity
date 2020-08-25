import pandas as pd
import numpy as np
import sympy
from scipy.optimize import minimize
import scipy.linalg as la

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
        price_data = df.iloc[:,1:] 
        returns_df = (price_data.iloc[1:,:].values / price_data.iloc[:-1,:] - 1)
        excess_returns_df = returns_df - returns_df.mean()
        excess_returns = np.array(excess_returns_df)
        excess_returns_T = excess_returns.T
        cov_mat = excess_returns_T.dot(excess_returns) / (excess_returns.shape[0] - 1)
        return cov_mat    