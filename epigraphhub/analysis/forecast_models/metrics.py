import pandas as pd
from sklearn.metrics import explained_variance_score as evs
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_log_error as msle


# computing some metrics
def compute_metrics(df_pred: pd.DataFrame) -> pd.DataFrame:
    """
    This function evaluates the predictions obtained in the `train_eval`
    method in the train and test sample. The predictions must be saved
    in a dataset with the following columns: 'median', 'target' and
    'train_size'.

    This function uses the following metrics:

    - explained variance score;
    - mean absolute error;
    - mean squared error;
    - root mean squared error;
    - mean squared log error;
    - mean absolute percentage error.
    To compute this metrics we use the implementations of the
    sklearn.metrics package.

    Parameters
    ----------
    df_pred : pd.DataFrame
        Dataframe with the columns: 'median', 'target' and 'train_size'.

    Returns
    -------
    pd.DataFrame
        DataFrame with two columns: out_sample and in_sample and with
        the metrics as index.
    """

    metrics = [
        "explained_variance_score",
        "mean_absolute_error",
        "mean_squared_error",
        "root_mean_squared_error",
        "mean_squared_log_error",
        "mean_absolute_percentage_error",
    ]

    # computing error in train sample
    df_metrics = pd.DataFrame(columns=["metrics", "in_sample", "out_sample"])

    df_metrics["metrics"] = metrics

    split = df_pred["train_size"][0]
    y_true_in = df_pred["target"].iloc[:split]
    y_pred_in = df_pred["median"].iloc[:split]
    y_true_out = df_pred["target"].iloc[split:]
    y_pred_out = df_pred["median"].iloc[split:]

    df_metrics["in_sample"] = [
        evs(y_true_in, y_pred_in),
        mae(y_true_in, y_pred_in),
        mse(y_true_in, y_pred_in),
        mse(y_true_in, y_pred_in, squared=False),
        msle(y_true_in, y_pred_in),
        mape(y_true_in, y_pred_in),
    ]

    df_metrics["out_sample"] = [
        evs(y_true_out, y_pred_out),
        mae(y_true_out, y_pred_out),
        mse(y_true_out, y_pred_out),
        mse(y_true_out, y_pred_out, squared=False),
        msle(y_true_out, y_pred_out),
        mape(y_true_out, y_pred_out),
    ]

    df_metrics.set_index("metrics", inplace=True)

    return df_metrics
