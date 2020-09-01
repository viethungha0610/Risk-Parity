api_key = "0d85196943mshb6828091ff3fedap147d34jsnee33a93be78a"
security = "MSFT"
freq = "1mo"
start = "2010-30-08"
finish = "2020-30-08"

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
        return merged_df