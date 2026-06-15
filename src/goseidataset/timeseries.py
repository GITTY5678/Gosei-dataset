import pandas as pd
from datetime import datetime

class   TimeSeriesGenerator:
    def __init__(self,dataset):
        
        if isinstance(dataset,pd.DataFrame):
            self.df=dataset.copy()
        elif isinstance(dataset,str):
            self.df=pd.read_csv(dataset)
        elif isinstance(dataset,dict):
            self.df=pd.DataFrame(dataset)
        else:
            raise TypeError("Dataset must be in a dictionary or in a dataframe or file path format")
    def manual_insertion(self,column_name,column_values):
        data=self.df.copy()
        if len(column_values)!=len(data):
            raise ValueError(f"Please enter the proper number of values,in your case you should have exactly {len(data)}")
        data.insert(0,column_name,column_values)
        
        return data
    def year_series(self,start=None,step=1):
        if start is None:
            start=str(datetime.now().year)
        dates=pd.date_range(start=start,periods=len(self.df),freq=f"{step}YS")
        return self.manual_insertion("YEAR",dates)
    def month_series(self,start=None,step=1):
        if start is None:
            start=datetime.now().strftime("%Y-%m-01")
        dates=pd.date_range(start=start,periods=len(self.df),freq=f"{step}MS")
        return self.manual_insertion("MONTH",dates.strftime("%Y-%m"))
    def week_series(self,start=None,step=1):
        if start is None:
            start=datetime.now().strftime("%Y-%m-%d")
        dates=pd.date_range(start=start,periods=len(self.df),freq=f"{7*step}D")
        return self.manual_insertion("WEEK",dates.strftime("%Y-%m-%d"))
    def day_series(self,start=None,step=1):
        if start is None:
            start=datetime.now().strftime("%Y-%m-%d")
        dates=pd.date_range(start=start,periods=len(self.df),freq=f"{step}D")
        return self.manual_insertion("DAY",dates.strftime("%Y-%m-%d"))
    def hour_series(self,start=None,step=1):
        if start is None:
            start=datetime.now().strftime("%Y-%m-%d %H-00-00")
        dates=pd.date_range(start=start,periods=len(self.df),freq=f"{step}h")
        return self.manual_insertion("HOUR",dates.strftime("%Y-%m-%d %H-00-00"))
    def minute_series(self,start=None,step=1):
        if start is None:
            start=datetime.now().strftime("%Y-%m-%d %H-%M-00")
        dates=pd.date_range(start=start,periods=len(self.df),freq=f"{step}min")
        return self.manual_insertion("MINUTE",dates.strftime("%Y-%m-%d %H-%M-00"))
    def second_series(self,start=None,step=1):
        if start is None:
            start=datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        dates=pd.date_range(start=start,periods=len(self.df),freq=f"{step}s")
        return self.manual_insertion("SECOND",dates.strftime("%Y-%m-%d %H-%M-%S"))
    def timestamp_series(self,start=None,step=1):
        if start is None:
            start=datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        dates=pd.date_range(start=start,periods=len(self.df),freq=f"{step}s")
        return self.manual_insertion("TIMESTAMP",dates)