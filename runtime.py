from flask import Flask, request, send_file
from flasgger import Swagger
from RiskParity import RiskParity, DataPreprocessor, GetRapidAPIData
import pandas as pd
import seaborn as sns

app = Flask(__name__)
swagger = Swagger(app)

@app.route('/riskparity')
def RiskParityMain():
    """End point for the Risk Parity portfolio optimizer
    ---
    parameters:
      - name: RapidAPI_key
        in: query
        type: string
        description: Rapid API Key for the Yahoo Finance API, get a free API key from https://rapidapi.com/apidojo/api/yahoo-finance1 
        required: true
      - name: Asset_list
        in: query
        type: string
        required: true
      - name: Start_date
        in: query
        type: string
        required: true
      - name: End_date
        in: query
        type: string
        required: true
      - name: Frequency
        in: query
        type: string
        description: 1mo for monthly frequency; 1wk for weekly frequency; 1d for daily frequency
        required: true
    responses:
        200:
            description: Portfolio allocation to achieve Risk Parity
            schema:
                type: file
    """
    # Getting all the parameters to start
    api_key = request.args.get('RapidAPI_key')
    assets = request.args.get('Asset_list')
    start_date = request.args.get('Start_date')
    end_date = request.args.get('End_date')
    freq = request.args.get('Frequency', default='1mo')

    # Process assets list
    assets_list = []
    for asset in assets.split(', '):
        assets_list.append(str(asset))

    # Importing data
    data = GetRapidAPIData(api_key, assets_list, start_date, end_date, freq)
    outcome_df = data.prep_data()

    # Preprocess data
    preprocessing = DataPreprocessor()
    cov_mat = preprocessing.transform(outcome_df)
    assets = preprocessing.assets_

    # Optimize the portfolio to achieve Risk Parity
    rpp = RiskParity(cov_mat, assets)
    rpp.optimize()

    # Getting allocation output
    allocation = rpp.allocation_
    allocation_dict = {}
    for asset, weight in zip(assets, allocation):
        allocation_dict[asset] = weight
    
    # Allocation figure
    allocation_fig = sns.barplot(y=allocation, x=assets)
    allocation_fig.set_title('Risk Parity portfolio allocation')
    allocation_fig.figure.savefig('Allocation_fig.png')

    # Getting risk statistics output
    rpp.cal_risk_stats()
    rpp.visualise_risk_stats()
    MRC_fig = rpp.MRC_fig_
    MRC_fig.figure.savefig('MRC_fig.png')
    RC_fig = rpp.RC_fig_
    RC_fig.figure.savefig('RC_fig.png')
    RRC_fig = rpp.RRC_fig_
    RRC_fig.figure.savefig('RRC_fig.png')

    # Generate summary Excel file
    writer = pd.ExcelWriter('RiskParity_summary.xlsx', engine='xlsxwriter')
    rpp.allocation_df_.to_excel(writer, sheet_name='Allocation', index=False)
    worksheet_allocation = writer.sheets['Allocation']
    worksheet_allocation.insert_image('D1', 'Allocation_fig.png')
    risk_stats_df = pd.DataFrame({'Assets': assets, 
                            'Marginal Risk Contribution': rpp.MRC_,
                            'Risk Contribution': rpp.RC_,
                            'Relative Risk Contribution': rpp.RRC_})
    risk_stats_df.to_excel(writer, sheet_name='Risk Stats', index=False)
    worksheet_riskstats = writer.sheets['Risk Stats']
    worksheet_riskstats.insert_image('F1', 'MRC_fig.png')
    worksheet_riskstats.insert_image('F25', 'RC_fig.png')
    worksheet_riskstats.insert_image('F49', 'RRC_fig.png')
    outcome_df.to_excel(writer, sheet_name='Data', index=False)
    writer.save()

    # Return summary Excel file
    return send_file('RiskParity_summary.xlsx', 
                    mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 
                    as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)