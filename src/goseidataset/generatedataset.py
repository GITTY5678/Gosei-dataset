import pandas as pd
import numpy as np

class DatasetGenerator:
    def __init__(self):
        pass
    def generate_random(
    self,
    n_rows,
    constraints,
    bias=None,
    dtypes=None
):
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

    dtypes : dict, optional

        Specifies the data type for
        numerical columns.

        Supported values:

        - "int"
        - "float"

        Example:

        {
            "age": "int",
            "salary": "int",
            "cgpa": "float"
        }

        Columns not specified are
        generated as float by default.

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

    ValueError
        If an invalid dtype is provided.

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

        if dtypes is None:
            dtypes = {}

        data = {}

        for column, values in constraints.items():

            # Numerical column
            if all(
                isinstance(v, (int, float))
                for v in values
            ):

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

                column_bias = bias.get(
                    column,
                    None
                )

                # No bias
                if column_bias is None:

                    generated = np.random.uniform(
                        min_value,
                        max_value,
                        n_rows
                    )

                # String bias
                elif isinstance(
                    column_bias,
                    str
                ):

                    if (
                        column_bias.lower()
                        == "high"
                    ):

                        generated = np.random.beta(
                            5,
                            2,
                            n_rows
                        )

                        generated = (
                            min_value
                            +
                            generated
                            * (
                                max_value
                                - min_value
                            )
                        )

                    elif (
                        column_bias.lower()
                        == "low"
                    ):

                        generated = np.random.beta(
                            2,
                            5,
                            n_rows
                        )

                        generated = (
                            min_value
                            +
                            generated
                            * (
                                max_value
                                - min_value
                            )
                        )

                    elif (
                        column_bias.lower()
                        == "center"
                    ):

                        generated = np.random.beta(
                            5,
                            5,
                            n_rows
                        )

                        generated = (
                            min_value
                            +
                            generated
                            * (
                                max_value
                                - min_value
                            )
                        )

                    else:

                        raise ValueError(
                            f"Invalid bias "
                            f"'{column_bias}' "
                            f"for column "
                            f"'{column}'."
                        )

                # Mean bias
                elif isinstance(
                    column_bias,
                    (int, float)
                ):

                    mean = column_bias

                    std = (
                        max_value
                        - min_value
                    ) / 6

                    generated = np.random.normal(
                        loc=mean,
                        scale=std,
                        size=n_rows
                    )

                    generated = np.clip(
                        generated,
                        min_value,
                        max_value
                    )

                else:

                    raise ValueError(
                        f"Invalid bias "
                        f"for column "
                        f"'{column}'."
                    )

                # Apply dtype
                column_dtype = dtypes.get(
                    column,
                    "float"
                )

                if column_dtype == "int":

                    generated = np.round(
                        generated
                    ).astype(int)

                elif column_dtype == "float":

                    generated = generated.astype(
                        float
                    )

                else:

                    raise ValueError(
                        f"Invalid dtype "
                        f"'{column_dtype}' "
                        f"for column "
                        f"'{column}'. "
                        "Use 'int' or 'float'."
                    )

                data[column] = generated

            # Categorical column
            elif all(
                isinstance(v, str)
                for v in values
            ):

                if len(values) == 0:
                    raise ValueError(
                        f"{column} must contain "
                        "at least one category."
                    )

                data[column] = np.random.choice(
                    values,
                    size=n_rows
                )

            # Mixed types
            else:

                raise ValueError(
                    f"{column} contains mixed "
                    "data types. Use either "
                    "all numeric values or "
                    "all string values."
                )

        return pd.DataFrame(data)
    def generate_correlated(
    self,
    n_rows,
    target,
    correlations,
    constraints,
    return_report=False
):
        """
    Generate a synthetic dataset with user-defined
    correlations against a target variable.

    Parameters
    ----------
    n_rows : int
        Number of rows to generate.

    target : str
        Name of the target column.

    correlations : dict
        Desired correlations with target.

        Example:

        {
            "sleep": 0.3,
            "revision": 0.8,
            "stress": -0.4
        }

    constraints : dict
        Numerical ranges for columns.

        Example:

        {
            "sleep": [4, 10],
            "revision": [0, 8],
            "stress": [1, 100],
            "retention": [0, 100]
        }

    return_report : bool, optional

        If True, returns both
        dataset and correlation report.

        Default is False.

    Returns
    -------
    pandas.DataFrame

        Generated dataset.

    or

    dict

        Returned when
        return_report=True

        {
            "dataset": DataFrame,
            "correlation_report": DataFrame
        }

    Raises
    ------
    ValueError

        If n_rows < 1.

    ValueError

        If target is not present
        inside constraints.

    ValueError

        If correlation values
        are outside [-1, 1].

    Notes
    -----
    This method is intended for
    educational purposes and
    synthetic dataset generation.

    Actual correlations may differ
    slightly from requested values
    because of random sampling.
    """

        if n_rows < 1:
            raise ValueError(
                "n_rows must be greater than 0."
            )

        if target not in constraints:
            raise ValueError(
                f"{target} must be present "
                "inside constraints."
            )

        for feature, corr in correlations.items():

            if feature not in constraints:

                raise ValueError(
                    f"{feature} not found "
                    "inside constraints."
                )

            if corr < -1 or corr > 1:

                raise ValueError(
                    f"Correlation for "
                    f"{feature} must be "
                    "between -1 and 1."
                )

        data = {}

        # ----------------------------------
        # Generate Target Signal
        # ----------------------------------

        target_signal = np.random.normal(
            0,
            1,
            n_rows
        )

        target_min = constraints[target][0]
        target_max = constraints[target][1]

        target_scaled = (
            (
                target_signal
                -
                target_signal.min()
            )
            /
            (
                target_signal.max()
                -
                target_signal.min()
            )
        )

        target_scaled = (
            target_scaled
            *
            (
                target_max
                -
                target_min
            )
            +
            target_min
        )

        data[target] = target_scaled

        # ----------------------------------
        # Generate Correlated Features
        # ----------------------------------

        report = []

        for feature, corr in correlations.items():

            noise = np.random.normal(
                0,
                1,
                n_rows
            )

            feature_signal = (
                corr
                *
                target_signal
                +
                np.sqrt(
                    1 - corr**2
                )
                *
                noise
            )

            feature_min = constraints[feature][0]
            feature_max = constraints[feature][1]

            feature_scaled = (
                (
                    feature_signal
                    -
                    feature_signal.min()
                )
                /
                (
                    feature_signal.max()
                    -
                    feature_signal.min()
                )
            )

            feature_scaled = (
                feature_scaled
                *
                (
                    feature_max
                    -
                    feature_min
                )
                +
                feature_min
            )

            data[feature] = feature_scaled

        dataset = pd.DataFrame(data)

        # ----------------------------------
        # Correlation Report
        # ----------------------------------

        for feature, requested_corr in correlations.items():

            actual_corr = (
                dataset[
                    [feature, target]
                ]
                .corr()
                .iloc[0, 1]
            )

            report.append(
                {
                    "Feature": feature,
                    "Requested": round(
                        requested_corr,
                        4
                    ),
                    "Actual": round(
                        actual_corr,
                        4
                    )
                }
            )

        report = pd.DataFrame(report)

        if return_report:

            return {
                "dataset": dataset,
                "correlation_report": report
            }

        return dataset
    def generate_formula(self,n_rows,formula,constraints,target,noise=0.1):
        """
    Generate a synthetic dataset using
    a mathematical formula.

    Parameters
    ----------
    n_rows : int
        Number of rows to generate.

    formula : str
        Formula used to create target.

        Example:

        "0.4*sleep + 0.8*revision - 0.2*stress"

    constraints : dict
        Numerical constraints.

        Example:

        {
            "sleep":[4,10],
            "revision":[0,8],
            "stress":[1,100],
            "retention":[0,100]
        }

    target : str
        Target column name.

    noise : float, default=0.1
        Amount of random noise added.

    Returns
    -------
    pandas.DataFrame
        Generated dataset.

    Raises
    ------
    ValueError
        If target is missing
        from constraints.

    Notes
    -----
    Intended for educational
    purposes and synthetic
    dataset generation.
    """
        if n_rows < 1:
            raise ValueError(
            "n_rows must be greater than 0."
        )

        if target not in constraints:
            raise ValueError(
                f"{target} not found "
                "in constraints."
            )
        data={}
        for column, values in constraints.items():
    
            if column == target:
                continue

            if len(values) != 2:
                raise ValueError(
                    f"{column} must contain "
                    "[min,max]."
                )

            data[column] = np.random.uniform(
                values[0],
                values[1],
                n_rows
            )
        df=pd.DataFrame(data)
        try:

            target_values = eval(
                formula,
                {"np": np},
                df.to_dict("series")
            )

        except Exception as e:

            raise ValueError(
                f"Invalid formula: {e}"
            )

        # ---------------------------
        # Add Noise
        # ---------------------------

        target_values = (
            target_values
            +
            np.random.normal(
                0,
                noise,
                n_rows
            )
        )

        # ---------------------------
        # Scale Target
        # ---------------------------

        target_min = constraints[target][0]
        target_max = constraints[target][1]

        target_values = (
            (
                target_values
                -
                np.min(target_values)
            )
            /
            (
                np.max(target_values)
                -
                np.min(target_values)
            )
        )

        target_values = (
            target_values
            *
            (target_max - target_min)
            +
            target_min
        )

        df[target] = target_values

        return df
