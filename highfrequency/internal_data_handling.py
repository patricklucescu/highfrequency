import polars as pl
import warnings
from helpers.exchange_names import EXCHANGES


def check_trade_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    | Check if the input price data is a polars DataFrame and that the necessary columns are present.
    Additionally, the data types of the columns are checked.

    :param df: DataFrame to be checked.
    :return: DataFrame if it is a polars DataFrame.
    """

    if not isinstance(df, pl.DataFrame):
        raise TypeError("df must be a polars DataFrame")

    df_schema = df.schema

    if 'price' not in df_schema:
        raise KeyError(f"Column 'price' not found in DataFrame columns.")

    if 'dt' not in df_schema:
        raise KeyError(f"Column 'dt' not found in DataFrame columns.")

    if 'symbol' not in df_schema and 'sym_root' not in df_schema:
        warnings.warn(f"Column 'symbol' (or 'sym_root') not found in DataFrame columns.", UserWarning)

    if df_schema['price'] != pl.Float64:
        raise TypeError(f"Column 'price' should be of type Float64.")

    if df_schema['size'] not in [pl.Float64, pl.Int64]:
        raise TypeError(f"Column 'size' should be of type Float64 or Int64.")

    if df_schema['dt'] != pl.Datetime:
        raise TypeError(f"Column 'dt' should be of type Datetime.")

    return df


def check_quote_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    | Check if the input quote data is a polars DataFrame and that the necessary columns are present.
    Additionally, the data types of the columns are checked.

    :param df: DataFrame to be checked.
    :return: DataFrame if it is a polars DataFrame.
    """

    if not isinstance(df, pl.DataFrame):
        raise TypeError("df must be a polars DataFrame")

    df_schema = df.schema

    if 'dt' not in df_schema:
        raise KeyError(f"Column 'dt' not found in DataFrame columns.")

    if 'bid' not in df_schema:
        raise KeyError(f"Column 'bid' not found in DataFrame columns.")

    if 'ofr' not in df_schema:
        raise KeyError(f"Column 'ofr' not found in DataFrame columns.")

    if 'symbol' not in df_schema and 'sym_root' not in df_schema:
        warnings.warn(f"Column 'symbol' (or 'sym_root') not found in DataFrame columns.", UserWarning)

    if df_schema['bid'] != pl.Float64:
        raise TypeError(f"Column 'bid' should be of type Float64.")

    if df_schema['ofr'] != pl.Float64:
        raise TypeError(f"Column 'ofr' should be of type Float64.")

    if df_schema['bidsiz'] not in [pl.Float64, pl.Int64]:
        raise TypeError(f"Column 'bidsiz' should be of type Float64 or Int64.")

    if df_schema['ofrsiz'] not in [pl.Float64, pl.Int64]:
        raise TypeError(f"Column 'ofrsiz' should be of type Float64 or Int64.")

    if df_schema['dt'] != pl.Datetime:
        raise TypeError(f"Column 'dt' should be of type Datetime.")

    return df


def check_column_names(df: pl.DataFrame) -> pl.DataFrame:
    """
    | Sets column names according to RTAQ format using quantmod conventions, such that all the other functions find
    the correct information.

    :param df: polars DataFrame containing trade data.
    :return: DataFrame with column names set according to RTAQ format.
    """

    # lowercase all column names
    df = df.rename({col: col.lower() for col in df.columns})

    # Change column names to previous RTAQ format
    rename_map = {
        "ask": "ofr",
        "sym_root": "symbol",
        "bidsize": "bidsiz",
        "asksize": "ofrsiz",
        "asksiz": "ofrsiz",
        "ofrsize": "ofrsiz",
        "tr_scond": "cond",
        "cr": "corr",
        "tr_corr": "corr"
    }

    # Apply the renaming map only if the column exists in the dataframe
    df = df.rename({old: new for old, new in rename_map.items() if old in df.columns})

    return df
