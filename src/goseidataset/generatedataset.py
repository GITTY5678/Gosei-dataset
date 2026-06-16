import pandas as pd
import numpy as np

class DatasetGenerator:
    def __init__(self):
        pass
    def generate_random(self,n_rows,constraints,bias=None):
        """
        Generate a random dataset using
        user-defined constraints.

        Parameters
        ----------
        n_rows : int
            Number of rows to generate.

        constraints : dict
            Dictionary defining the values
            to generate for each column.

            Numerical columns:

            {
                "sleep": [4, 10],
                "revision": [0, 8]
            }

            Categorical columns:

            {
                "session": ["M", "J"],
                "subject": [
                    "Math",
                    "Physics",
                    "Chemistry"
                ]
            }

        bias : dict, optional

            Controls the distribution of
            numerical columns.

            Supported values:

            - "high"
            - "low"
            - "center"
            - numeric value

            Example:

            {
                "sleep": "high",
                "revision": "center",
                "stress": 20
            }

            Default is None.

        Returns
        -------
        pandas.DataFrame
            Randomly generated dataset.

        Raises
        ------
        ValueError
            If n_rows is less than 1.

        ValueError
            If constraints is empty.

        ValueError
            If numerical constraints do not
            contain exactly [min, max].

        ValueError
            If a column contains mixed
            data types.

        Notes
        -----
        This method is intended for
        educational purposes,
        synthetic dataset generation,
        testing machine learning pipelines,
        and benchmarking algorithms.
        """
        if n_rows < 1:
            raise ValueError(
                "n_rows must be greater than 0."
            )

        if not constraints:
            raise ValueError(
                "constraints cannot be empty."
            )

        if bias is None:
            bias = {}
        data={}
        for column,values in constraints.items():
            if (isinstance(v,(int,float)) for v in values):
                if len(values) != 2:
                    raise ValueError(
                        f"{column} must contain "
                        "[min, max] for numerical data."
                    )

                min_value = values[0]
                max_value = values[1]

                if min_value > max_value:
                    raise ValueError(
                        f"{column}: minimum value "
                        "cannot be greater than "
                        "maximum value."
                    )
                column_bias=bias.get(column,0)
                #no bias
                if column_bias is None:
                    generated=np.random.uniform(min_value,max_value,n_rows)
                #high bias
                if column_bias.lower()=="high":
                    generated=np.random.beta(5,2,n_rows)
                    generated=(min_value+generated*(max_value-min_value))
                #low bias
                if column_bias.lower()=="low":
                    generated=np.random.beta(2,5,n_rows)
                    generated=(min_value+generated*(max_value-min_value))
                #center bias
                if column_bias.lower()=="center":
                    generated=np.random.beta(5,5,n_rows)
                    generated=(min_value+generated*(max_value-min_value))
                #mean bias
                elif isinstance(column_bias,(int,float)):
                    mean=column_bias
                    
                    std=(max_value-min_value)/6
                    generated=np.random.normal(loc=mean,scale=std,size=n_rows)
                    generated=np.clip(generated,min_value,max_value)
                    
                else:
    
                    raise ValueError(
                        f"Invalid bias "
                        f"for column '{column}'."
                    )
                data[column]=generated
                
            elif all(isinstance(v,str) for v in values):
                if len(values) == 0:
                    raise ValueError(
                        f"{column} must contain "
                        "at least one category."
                    )

                data[column] = np.random.choice(
                    values,
                    size=n_rows
                )
            else:
    
                raise ValueError(
                    f"{column} contains mixed "
                    "data types. Use either "
                    "all numeric values or "
                    "all string values."
                )
        return pd.DataFrame(data)