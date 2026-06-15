import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.base import clone

from sklearn.impute import KNNImputer
class Supervised_learning:

    def __init__(self, dataset_a, dataset_b, target):
        self.data_1 = dataset_a
        self.data_2 = dataset_b
        self.tar = target

    def merge_same_target(self):

        if len(self.data_1) != len(self.data_2):
            raise ValueError(
                "Both datasets must have the same number of rows."
            )

        elif (
            self.tar not in self.data_1.columns
            or self.tar not in self.data_2.columns
        ):
            raise ValueError(
                f"{self.tar} not found in one of the datasets."
            )

        elif not self.data_1[self.tar].equals(
            self.data_2[self.tar]
        ):
            raise ValueError(
                "Target values does not match."
            )

        data_2_features = self.data_2.drop(
            columns=[self.tar]
        )

        return pd.concat(
            [self.data_1, data_2_features],
            axis=1
        )

    def compare_models(self,model,test_size=0.2,random_state=42):
        """
        Compare the performance of the same model
        on Dataset A and Dataset B.

        Parameters
        ----------
        model : sklearn model
            Any sklearn-compatible model.

        test_size : float, default=0.2
            Fraction of data used for testing.

        random_state : int, default=42
            Random seed for train-test split.

        Returns
        -------
        dict
            Comparison report containing metrics
            for both datasets and the winner.
        """

        x_a=self.data_1.drop(columns=[self.tar])
        y_a=self.data_1[self.tar]
        
        x_b=self.data_2.drop(columns=[self.tar])
        y_b=self.data_2[self.tar]
        
        x_train_a,x_test_a,y_train_a,y_test_a=train_test_split(x_a,y_a,test_size=test_size,random_state=random_state)
        x_train_b,x_test_b,y_train_b,y_test_b=train_test_split(x_b,y_b,test_size=test_size,random_state=random_state)
        
        model_a=clone(model)
        model_b=clone(model)
        
        model_a.fit(x_train_a,y_train_a)
        model_b.fit(x_train_b,y_train_b)
        
        pred_a=model_a.predict(x_test_a)
        pred_b=model_b.predict(x_test_b)
        
        r2_a = r2_score(y_test_a, pred_a)
        mae_a = mean_absolute_error(y_test_a, pred_a)
        rmse_a = np.sqrt(
            mean_squared_error(y_test_a, pred_a)
        )

        # Metrics B
        r2_b = r2_score(y_test_b, pred_b)
        mae_b = mean_absolute_error(y_test_b, pred_b)
        rmse_b = np.sqrt(
            mean_squared_error(y_test_b, pred_b)
        )

        # Store trained models
        self.model_a = model_a
        self.model_b = model_b

        # Determine winner
        if r2_a > r2_b:
            winner = "dataset_a"
        elif r2_b > r2_a:
            winner = "dataset_b"
        else:
            winner = "tie"

        return {
            "dataset_a": {
                "r2": round(r2_a, 4),
                "mae": round(mae_a, 4),
                "rmse": round(rmse_a, 4)
            },

            "dataset_b": {
                "r2": round(r2_b, 4),
                "mae": round(mae_b, 4),
                "rmse": round(rmse_b, 4)
            },

            "winner": winner
        }
        
    def weighted_ensemble(self,model,test_size=0.2,random_state=42):
        """
    Create a weighted ensemble using Dataset A and Dataset B.

    Parameters
    ----------
    model : sklearn model
        Any sklearn-compatible model.

    data_a : pandas.DataFrame
        New data containing features of Dataset A.

    data_b : pandas.DataFrame
        New data containing features of Dataset B.

    test_size : float, default=0.2
        Fraction of data used for testing.

    random_state : int, default=42
        Random seed.

    Returns
    -------
    dict
        Ensemble predictions and report.
    """
        x_a=self.data_1.drop(columns=[self.tar])
        y_a=self.data_1[self.tar]
        
        x_b=self.data_2.drop(columns=[self.tar])
        y_b=self.data_2[self.tar]
        
        x_train_a,x_test_a,y_train_a,y_test_a=train_test_split(x_a,y_a,test_size=test_size,random_state=random_state)
        x_train_b,x_test_b,y_train_b,y_test_b=train_test_split(x_b,y_b,test_size=test_size,random_state=random_state)
        
        model_a=clone(model)
        model_b=clone(model)
        
        model_a.fit(x_train_a,y_train_a)
        model_b.fit(x_train_b,y_train_b)
        
        pred_a=model_a.predict(x_test_a)
        pred_b=model_b.predict(x_test_b)
        
        r2_a = r2_score(y_test_a, pred_a)
        r2_b = r2_score(y_test_b, pred_b)
        
        total_r2=r2_a+r2_b
        
        #weights
        if total_r2==0:
            weight_a,weight_b=0,0
        else:
            weight_a=abs(r2_a)/total_r2
            weight_b=abs(r2_b)/total_r2
            
        ensemble_pred = (
    weight_a * pred_a
    +
    weight_b * pred_b
)
        final_table = pd.concat(
    [
        self.data_1.reset_index(drop=True),
        self.data_2.reset_index(drop=True),
        pd.DataFrame({"R": ensemble_pred})
    ],
    axis=1
)
        return {
    "ensemble_dataset": final_table,

    "weight_a": round(weight_a, 4),
    "weight_b": round(weight_b, 4),

    "dataset_a_r2": round(r2_a, 4),
    "dataset_b_r2": round(r2_b, 4)
}
        
    def vertical_stack(self,method="mean",n_neighbour=5):
        """
Vertically stack Dataset A and Dataset B and perform
missing value imputation using the user-selected method.

Parameters
----------
method : str, default="mean"

    Imputation method to use.

    Supported methods:
    - "mean"
    - "median"
    - "mode"
    - "knn"
    - "iterative"
    - "random_forest"
    - "xgboost"

n_neighbors : int, default=5

    Number of neighbors used for KNN imputation.
    Applicable only when method="knn".

predictors : dict, optional

    Required for:
    - method="random_forest"
    - method="xgboost"

    Dictionary specifying which predictor columns
    should be used to impute each target column.

    Example:

    {
        "Sleep": ["Retention"],
        "X": ["Retention", "Revision"]
    }

Returns
-------
dict

    Returns a dictionary containing:

    dataset : pandas.DataFrame

        Vertically stacked dataset after
        performing imputation.

    report : dict

        Detailed imputation report including:

        - method used
        - columns imputed
        - rows imputed
        - missing values before imputation
        - missing values after imputation
        - model performance metrics (where applicable)
        - success/failure status for each column
        - failure reasons (if any)

Notes
-----

For "mean", "median", "mode", and "knn":

    Missing values are filled directly using the
    selected statistical or distance-based method.

For "random_forest" and "xgboost":

    A separate predictive model is trained for
    each target column specified in the predictors
    dictionary.

    The model is trained using available rows and
    then used to estimate missing values.

    Performance metrics such as R² and MAE are
    included in the report.

For "iterative":

    IterativeImputer is used to estimate missing
    values by repeatedly modeling each column
    using the remaining columns.

Examples
--------

Mean Imputation

>>> sl.vertical_stack(
...     method="mean"
... )

KNN Imputation

>>> sl.vertical_stack(
...     method="knn",
...     n_neighbors=3
... )

Random Forest Imputation

>>> sl.vertical_stack(
...     method="random_forest",
...     predictors={
...         "Sleep": ["Retention"],
...         "Revision": ["Retention"]
...     }
... )

XGBoost Imputation

>>> sl.vertical_stack(
...     method="xgboost",
...     predictors={
...         "Sleep": ["Retention", "Revision"]
...     }
... )

Iterative Imputation

>>> sl.vertical_stack(
...     method="iterative"
... )
"""
        data=pd.concat([self.data_1,self.data_2],axis=0,ignore_index=False,sort=False)
        
        #mean imputation
        if method.lower()=="mean":
            for col in data.columns:
                if data[col].isnull().sum()>0:
                    data[col]=data[col].fillna(data[col].mean())
        #median
        elif method.lower()=="median":
            for col in data.columns:
                if data[col].isnull().sum()>0:
                    data[col]=data[col].fillna(data[col].median())
        #mode
        elif method.lower()=="mode":
            for col in data.columns:
                if data[col].isnull().sum()>0:
                    data[col]=data[col].fillna(data[col].mode())
        #knn
        elif method.lower()=="knn":
            imputer=KNNImputer(n_neighbors=n_neighbour)
            data[:]=imputer.fit_transform(data)
        elif method.lower() in [
    "random_forest",
    "xgboost"
]:

            report = {}

            for target_col, predictor_cols in predictors.items():

                try:

                    if target_col not in data.columns:

                        report[target_col] = {
                            "status": "failed",
                            "reason": "Column not found"
                        }

                        continue

                    train_df = data[
                        data[target_col].notna()
                    ]

                    predict_df = data[
                        data[target_col].isna()
                    ]

                    if len(train_df) == 0:

                        report[target_col] = {
                            "status": "failed",
                            "reason": "No training rows available"
                        }

                        continue

                    if train_df[predictor_cols].isnull().sum().sum() > 0:

                        report[target_col] = {
                            "status": "failed",
                            "reason": "Predictor columns contain missing values"
                        }

                        continue

                    X = train_df[predictor_cols]
                    y = train_df[target_col]

                    X_train, X_test, y_train, y_test = (
                        train_test_split(
                            X,
                            y,
                            test_size=0.2,
                            random_state=42
                        )
                    )

                    if method.lower() == "random_forest":

                        model = RandomForestRegressor(
                            random_state=42
                        )

                    else:

                        if XGBRegressor is None:

                            raise ImportError(
                                "xgboost is required. "
                                "Install using "
                                "'pip install xgboost'"
                            )

                        model = XGBRegressor(
                            random_state=42
                        )

                    model.fit(
                        X_train,
                        y_train
                    )

                    y_pred = model.predict(
                        X_test
                    )

                    r2 = r2_score(
                        y_test,
                        y_pred
                    )

                    mae = mean_absolute_error(
                        y_test,
                        y_pred
                    )

                    if len(predict_df) > 0:

                        if (
                            predict_df[
                                predictor_cols
                            ]
                            .isnull()
                            .sum()
                            .sum()
                            > 0
                        ):

                            report[target_col] = {
                                "status": "failed",
                                "reason":
                                "Rows to predict contain missing predictors"
                            }

                            continue

                        pred_values = model.predict(
                            predict_df[
                                predictor_cols
                            ]
                        )

                        data.loc[
                            data[target_col].isna(),
                            target_col
                        ] = pred_values

                    report[target_col] = {
                        "status": "success",
                        "method": method,
                        "predictors": predictor_cols,
                        "r2": round(r2, 4),
                        "mae": round(mae, 4),
                        "rows_imputed": len(predict_df)
                    }

                except Exception as e:

                    report[target_col] = {
                        "status": "failed",
                        "reason": str(e)
                    }

            return {
                "dataset": data,
                "report": report
            }
        elif method.lower() == "iterative":
    
            missing_before = (
                data
                .isnull()
                .sum()
                .sum()
            )

            imputer = IterativeImputer(
                random_state=42
            )

            imputed_data = imputer.fit_transform(
                data
            )

            data = pd.DataFrame(
                imputed_data,
                columns=data.columns
            )

            report = {
                "method": "iterative",
                "missing_before": int(
                    missing_before
                ),
                "missing_after": int(
                    data
                    .isnull()
                    .sum()
                    .sum()
                )
            }

            return {
                "dataset": data,
                "report": report
            }
            
        else:
    
            raise ValueError(
                """
                Supported methods:
                mean
                median
                mode
                knn
                iterative
                random_forest
                xgboost
                """
            )
        
        return {"dataset":data}
    
    
