import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.base import clone


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
        
        
        
        