# Risk-Parity
Simple-Risk-Parity

Version 1.0

This project provides an investment portfolio allocation tool to provide Risk Parity. 
Pioneered by Bridgewater Associates, Risk Parity aims to diversify investments based on the individual investment's risk profiles and the investments' correlation.
It does so by allocating the portfolio in a way that each investment contributes the equal amount of risk to the portfolio.

*IMPORTANT*
To use the data importing `GetRapidAPIData` class, it is crucial that you sign up for the Yahoo Finance API via the RapidAPI website.
[https://rapidapi.com/apidojo/api/yahoo-finance1]

From here you can get an API key and then paste it into <code>secret.py</code> file.

*HOW TO USE THE RISK PARITY APP*
### Method 1: Run the Docker image
Pull the following Docker image from my public Docker repo and run it from localhost port 5000 on your machine.

`docker pull viethungha0610/portfolio_projects:risk_parity_mini`

Then use your browser to open this `http://localhost:5000/apidocs/`

### Method 2: Fork and clone this Github repo and run the `runtime.py` file.

Then use your browser to open this `http://localhost:5000/apidocs/`

### Method 3: Fork and clone this Github repo and run the `Illustrative Case Study.ipynb` Jupyter notebook.
