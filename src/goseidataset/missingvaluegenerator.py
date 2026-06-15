"""
MissingValueGenerator

A utility class for intentionally introducing missing values
into datasets.

This module is designed primarily for:

- Educational purposes
- Learning data preprocessing techniques
- Testing missing value handling strategies
- Evaluating imputation methods
- Benchmarking data cleaning pipelines
- Simulating real-world incomplete datasets

The generated missing values can be used to assess the
performance of various imputation techniques such as:

- Mean Imputation
- Median Imputation
- Mode Imputation
- KNN Imputation
- Iterative Imputation
- Random Forest Imputation
- XGBoost Imputation

Notes
-----
This module is intended for experimentation, research,
teaching, and model evaluation. It should not be used
to intentionally corrupt production datasets.

Examples
--------
>>> from goseidataset import MissingValueGenerator

>>> mv = MissingValueGenerator(dataset)

>>> mv.inject_missing_values(
...     percentage=20,
...     strategy="random"
... )

This will randomly replace approximately 20% of the
dataset values with missing values (NaN).
"""

import pandas as pd
import random
import numpy as np
class MissingValueGenerator:
    def __init__(self,dataset):
        """
        Initialize the MissingValueGenerator.

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
            If the dataset is not a DataFrame,
            dictionary, or CSV file path.

        Notes
        -----
        This class is intended for educational
        purposes and for testing the effectiveness
        of missing value imputation methods.

        The generated missing values can be used
        to benchmark and compare different
        imputation techniques.
        """
        if isinstance(dataset, pd.DataFrame):
            self.df = dataset.copy()

        elif isinstance(dataset, str):
            self.df = pd.read_csv(dataset)

        elif isinstance(dataset, dict):
            self.df = pd.DataFrame(dataset)

        else:
            raise TypeError(
                "Dataset must be a DataFrame, "
                "dictionary, or CSV file path."
            )
        
    def random_missing(self,percentage):
        """
    Randomly introduce missing values into the dataset.

    Parameters
    ----------
    percentage : float
        Percentage of dataset cells to be replaced
        with missing values (NaN).

    Returns
    -------
    pandas.DataFrame
        Dataset with randomly injected missing values.

    Raises
    ------
    ValueError
        If percentage is not between 0 and 100.

    Notes
    -----
    This method is intended for educational
    purposes and for testing imputation methods.

    Missing values are inserted completely
    at random throughout the dataset.
    """
        if percentage<0 or percentage>100:
            raise ValueError("percentage must be between 0 and 100.")
        data=self.df.copy()
        total_cells=data.shape[0]*data.shape[1]
        missing_cells=int((percentage/100)*total_cells)
        for _ in range(missing_cells):
            row=random.randint(0,data.shape[0]-1)
            col=random.randint(0,data.shape[1]-1)
            data.iat(row,col)=np.nan
        return data
    def column_missing(self,columns,percentage):
        """
    Introduce missing values into selected columns.

    Parameters
    ----------
    columns : list
        List of column names where missing values
        should be injected.

    percentage : float
        Percentage of values to replace with NaN
        in each selected column.

    Returns
    -------
    pandas.DataFrame
        Dataset with injected missing values.

    Raises
    ------
    ValueError
        If percentage is not between 0 and 100.

    ValueError
        If any specified column does not exist.

    Notes
    -----
    This method is intended for educational
    purposes and testing imputation methods.
    """
        if percentage<0 or percentage>100:
            raise ValueError("percentage must be between 0 and 100.")
        data=self.df.copy()
        for col in columns:
            if col not in data.columns:
                raise ValueError(
                f"Column '{col}' not found in dataset."
            )
            missing=int((percentage/100)*len(data))
            rows=np.random.choice(data.index,size=missing,replace=False)
            data.loc(rows,col)=np.nan
        return data
    def row_missing(self,rows):
        """
    Introduce missing values into selected rows.

    Parameters
    ----------
    rows : list
        List of row indices where values should
        be replaced with NaN.

    Returns
    -------
    pandas.DataFrame
        Dataset with injected missing values.

    Raises
    ------
    ValueError
        If any row index is outside the dataset.

    Notes
    -----
    This method is intended for educational
    purposes and testing imputation methods.

    All values in the selected rows are replaced
    with missing values (NaN).
    """
        data=self.df.copy()
        for row in rows:
            if row not in data.index:
                raise ValueError(
                f"Row index {row} not found in dataset."
            )
            data.loc(row)=np.nan
        return data
    def block_missing(self,start,end):
        """
    Introduce a continuous block of missing values
    into the dataset.

    Parameters
    ----------
    start : int
        Starting row index of the missing block.

    end : int
        Ending row index of the missing block
        (inclusive).

    Returns
    -------
    pandas.DataFrame
        Dataset with a continuous missing block.

    Raises
    ------
    ValueError
        If start or end indices are invalid.

    Notes
    -----
    This method is intended for educational
    purposes and testing imputation methods.

    All values within the specified row range
    are replaced with missing values (NaN).

    This is particularly useful for simulating:

    - Sensor failures
    - System outages
    - Missing time-series segments
    - Data transmission interruptions
    """
        data = self.df.copy()

        if start < 0:
            raise ValueError(
                "start index must be greater than or equal to 0."
            )

        if end >= len(data):
            raise ValueError(
                "end index exceeds dataset length."
            )

        if start > end:
            raise ValueError(
                "start index cannot be greater than end index."
            )
        data.loc[start:end]=np.nan
        return data
    def consecutive_missing(self,column,length):
        """
    Introduce a consecutive sequence of missing
    values into a selected column.

    Parameters
    ----------
    column : str
        Column where missing values should be
        introduced.

    length : int
        Number of consecutive values to replace
        with NaN.

    Returns
    -------
    pandas.DataFrame
        Dataset with consecutive missing values.

    Raises
    ------
    ValueError
        If the column does not exist.

    ValueError
        If length is less than 1.

    ValueError
        If length exceeds the number of rows
        in the dataset.

    Notes
    -----
    This method is intended for educational
    purposes and testing imputation methods.

    Consecutive missing values are useful for
    simulating:

    - Sensor outages
    - Device failures
    - Missing time-series intervals
    - Network interruptions
    """
        data = self.df.copy()

        if column not in data.columns:
            raise ValueError(
                f"Column '{column}' not found in dataset."
            )

        if length < 1:
            raise ValueError(
                "length must be greater than 0."
            )

        if length > len(data):
            raise ValueError(
                "length cannot exceed dataset size."
            )
        start=np.random.randint(start=0,end=len(data)-length+1)
        end=start+length
        data.loc[start:end]=np.nan
        return data

