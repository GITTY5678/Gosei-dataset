import pandas as pd
from datetime import datetime

class   TimeSeriesGenerator:
    def __init__(self,dataset):
        """
Initialize the TimeSeriesGenerator.

Parameters
----------
dataset : pandas.DataFrame, dict, or str
    Input dataset provided as:
    - A Pandas DataFrame
    - A Python dictionary
    - A CSV file path

Raises
------
TypeError
    If the dataset is not a DataFrame, dictionary, or CSV file path.
"""
        if isinstance(dataset,pd.DataFrame):
            self.df=dataset.copy()
        elif isinstance(dataset,str):
            self.df=pd.read_csv(dataset)
        elif isinstance(dataset,dict):
            self.df=pd.DataFrame(dataset)
        else:
            raise TypeError("Dataset must be in a dictionary or in a dataframe or file path format")
    def manual_insertion(self,column_name,column_values):
        """
Insert a generated time series column at the beginning of the dataset.

Parameters
----------
column_name : str
    Name of the column to be inserted.

column_values : array-like
    Generated time series values.

Returns
-------
pandas.DataFrame
    Dataset with the new time series column inserted at position 0.

Raises
------
ValueError
    If the number of generated values does not match
    the number of rows in the dataset.
"""
        data=self.df.copy()
        if len(column_values)!=len(data):
            raise ValueError(f"Please enter the proper number of values,in your case you should have exactly {len(data)}")
        data.insert(0,column_name,column_values)
        
        return data
    def year_series(self,start=None,step=1):
        """
Generate a yearly time series column.

Parameters
----------
start : str, optional
    Starting year. Defaults to the current year.

step : int, default=1
    Number of years between consecutive values.

Returns
-------
pandas.DataFrame
    Dataset with a YEAR column inserted as the first column.
"""
        if start is None:
            start=str(datetime.now().year)
        dates=pd.date_range(start=start,periods=len(self.df),freq=f"{step}YS")
        return self.manual_insertion("YEAR",dates)
    def month_series(self,start=None,step=1):
        """
Generate a monthly time series column.

Parameters
----------
start : str, optional
    Starting month in a valid date format.
    Defaults to the current month.

step : int, default=1
    Number of months between consecutive values.

Returns
-------
pandas.DataFrame
    Dataset with a MONTH column inserted as the first column.
"""
        if start is None:
            start=datetime.now().strftime("%Y-%m-01")
        dates=pd.date_range(start=start,periods=len(self.df),freq=f"{step}MS")
        return self.manual_insertion("MONTH",dates.strftime("%Y-%m"))
    def week_series(self,start=None,step=1):
        """
Generate a weekly time series column.

Parameters
----------
start : str, optional
    Starting date for the weekly sequence.
    Defaults to the current date.

step : int, default=1
    Number of weeks between consecutive values.

Returns
-------
pandas.DataFrame
    Dataset with a WEEK column inserted as the first column.
"""
        if start is None:
            start=datetime.now().strftime("%Y-%m-%d")
        dates=pd.date_range(start=start,periods=len(self.df),freq=f"{7*step}D")
        return self.manual_insertion("WEEK",dates.strftime("%Y-%m-%d"))
    def day_series(self,start=None,step=1):
        """
Generate a daily time series column.

Parameters
----------
start : str, optional
    Starting date for the daily sequence.
    Defaults to the current date.

step : int, default=1
    Number of days between consecutive values.

Returns
-------
pandas.DataFrame
    Dataset with a DAY column inserted as the first column.
"""
        if start is None:
            start=datetime.now().strftime("%Y-%m-%d")
        dates=pd.date_range(start=start,periods=len(self.df),freq=f"{step}D")
        return self.manual_insertion("DAY",dates.strftime("%Y-%m-%d"))
    def hour_series(self,start=None,step=1):
        """
Generate an hourly time series column.

Parameters
----------
start : str, optional
    Starting datetime for the hourly sequence.
    Defaults to the current hour.

step : int, default=1
    Number of hours between consecutive values.

Returns
-------
pandas.DataFrame
    Dataset with a HOUR column inserted as the first column.
"""
        if start is None:
            start=datetime.now().strftime("%Y-%m-%d %H-00-00")
        dates=pd.date_range(start=start,periods=len(self.df),freq=f"{step}h")
        return self.manual_insertion("HOUR",dates.strftime("%Y-%m-%d %H-00-00"))
    def minute_series(self,start=None,step=1):
        """
Generate a minute-level time series column.

Parameters
----------
start : str, optional
    Starting datetime for the minute sequence.
    Defaults to the current minute.

step : int, default=1
    Number of minutes between consecutive values.

Returns
-------
pandas.DataFrame
    Dataset with a MINUTE column inserted as the first column.
"""
        if start is None:
            start=datetime.now().strftime("%Y-%m-%d %H-%M-00")
        dates=pd.date_range(start=start,periods=len(self.df),freq=f"{step}min")
        return self.manual_insertion("MINUTE",dates.strftime("%Y-%m-%d %H-%M-00"))
    def second_series(self,start=None,step=1):
        """
Generate a second-level time series column.

Parameters
----------
start : str, optional
    Starting datetime for the second sequence.
    Defaults to the current second.

step : int, default=1
    Number of seconds between consecutive values.

Returns
-------
pandas.DataFrame
    Dataset with a SECOND column inserted as the first column.
"""
        if start is None:
            start=datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        dates=pd.date_range(start=start,periods=len(self.df),freq=f"{step}s")
        return self.manual_insertion("SECOND",dates.strftime("%Y-%m-%d %H-%M-%S"))
    def timestamp_series(self,start=None,step=1):
        """
Generate a timestamp column.

Parameters
----------
start : str, optional
    Starting datetime for the timestamp sequence.
    Defaults to the current timestamp.

step : int, default=1
    Number of seconds between consecutive timestamps.

Returns
-------
pandas.DataFrame
    Dataset with a TIMESTAMP column inserted as the first column.
"""
        if start is None:
            start=datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        dates=pd.date_range(start=start,periods=len(self.df),freq=f"{step}s")
        return self.manual_insertion("TIMESTAMP",dates)