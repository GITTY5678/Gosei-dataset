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
