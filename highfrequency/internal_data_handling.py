import polars as pl
import warnings
import numpy as np
from .helpers.schemas import TColumns, QColumns, column_name_mapper


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

    for column in TColumns:
        # check column is present
        if column.value.name not in df_schema:
            raise KeyError(f"Column {column.value.name} not found in DataFrame columns.")

        # check column data type
        if column.value.dtype is not None:
            if not isinstance(column.value.dtype, list):
                if df_schema[column.value.name] != column.value.dtype:
                    raise TypeError(f"Column {column.value.name} should be of type {column.value.dtype}.")
            else:
                if not any(df_schema[column.value.name] == dtype for dtype in column.value.dtype):
                    raise TypeError(f"Column {column.value.name} should be of type {column.value.dtype}.")

    return df


def check_quote_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    | Check if the input quote data is a polars DataFrame and that the necessary columns are present.
    Additionally, the data types of the columns are checked.

    :param df: DataFrame to be checked.
    :return: DataFrame if it is a polars DataFrame.
    """

    if not isinstance(df, pl.DataFrame):
        raise TypeError("df must be a polars DataFrame.")

    df_schema = df.schema

    for column in QColumns:
        # check column is present
        if column.value.name not in df_schema:
            raise KeyError(f"Column {column.value.name} not found in DataFrame columns.")

        # check column data type
        if column.value.dtype is not None:
            if not isinstance(column.value.dtype, list):
                if df_schema[column.value.name] != column.value.dtype:
                    raise TypeError(f"Column {column.value.name} should be of type {column.value.dtype}.")
            else:
                if not any(df_schema[column.value.name] == dtype for dtype in column.value.dtype):
                    raise TypeError(f"Column {column.value.name} should be of type {column.value.dtype}.")
    return df


def check_column_names(df: pl.DataFrame) -> pl.DataFrame:
    """
    | Sets column names according to RTAQ format using quantmod conventions, such that all the other functions find
    the correct information.

    :param df: polars DataFrame containing TAQ data.
    :return: DataFrame with column names set according to RTAQ format.
    """

    # lowercase all column names
    df = df.rename({col: col.lower() for col in df.columns})

    # Apply the renaming map only if the column exists in the dataframe
    df = df.rename({old: new for old, new in column_name_mapper.items() if old in df.columns})

    return df


