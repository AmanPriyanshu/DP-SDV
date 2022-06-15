# NASDAQ 100 2019

This demo dataset contains stock market data from all 2019 for the NASDAQ 100 companies, as of
August 24, 2020, as they were listed in the Wikipedia on October 15th 2020.

The data was downloaded using [yfinance](https://pypi.org/project/yfinance/) and augmented to
include the MarketCap, Sector and Industry contextual columns taken from
[Opendatasoft](https://public.opendatasoft.com/explore/dataset/nasdaq-companies).

The result is a table of shape (23343, 11) that contains data from 93 different Tickers.
