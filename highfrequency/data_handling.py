import polars as pl
from internal_data_handling import *
from typing import List
from helpers.exchange_names import EXCHANGES
from helpers.schemas import TColumns, QColumns


def select_exchange(
    t_data: pl.DataFrame,
    exchange: str = "N",
) -> pl.DataFrame:
    """
    | Filter the Raw Trade data to retain only the data from the specified exchange. Be default the NYSE is selected.

    :param t_data: DataFrame containing raw trade data.
    :param exchange: Name of the exchange to filter the data for.
    :return:
    """

    # check data consistency
    t_data = check_column_names(t_data)
    t_data = check_trade_data(t_data)

    # ensure the exchange selected is one of the allowed exchanges
    if exchange not in EXCHANGES:
        raise ValueError(f"Exchange '{exchange}' not found in the list of exchanges")

    # Filter the data where 'ex' column matches the specified 'exchange' value
    filtered_data = t_data.filter(pl.col(TColumns.EXCHANGE.value.name) == exchange)

    return filtered_data.sort(TColumns.TIME.value.name)


def auto_select_exchange_trades(
    t_data: pl.DataFrame, print_exchange: bool = False
) -> pl.DataFrame:
    """
    | Retain only data from the stock exchange with the highest trading volume.

    | Filters raw trade data and return only data that stems from the exchange with the highest value for the variable
    *SIZE*, i.e. the highest trade volume. The function can use the default column names for the trade volume and the
    exchange, but it can also be customized. Additionally, the function can either use the pre-defined list of exchanges
    or a custom list.

    :param t_data: polars DataFrame containing trade data with at least columns *size* and *ex*.
    :param print_exchange: If True, print the exchange with the highest trading volume.
    :return: DataFrame containing only data from the exchange with the highest trading volume.
    """
    # check data consistency
    t_data = check_column_names(t_data)
    t_data = check_trade_data(t_data)

    # Step 1: Group by 'symbol' and 'exchange', and aggregate 'size' by summing
    aggregated_df = t_data.group_by(
        [TColumns.SYMBOL.value.name, TColumns.EXCHANGE.value.name]
    ).agg(pl.sum(TColumns.EXCHANGE.value.name).alias("total_size"))

    # Step 2: Find the maximum size exchange for each symbol
    max_size_df = aggregated_df.group_by(TColumns.SYMBOL.value.name).agg(
        pl.max("total_size").alias("max_size")
    )

    # Step 3: Join the maximum size with the original aggregated dataframe to filter the exchanges
    joined_df = aggregated_df.join(max_size_df, on="symbol", how="inner")

    # Step 4: Filter to keep only the exchanges with the maximum size for each symbol
    filtered_df = joined_df.filter(pl.col("total_size") == pl.col("max_size"))

    # Step 5: Join back to the original dataframe to get the original rows
    result_df = t_data.join(
        filtered_df[[TColumns.SYMBOL.value.name, TColumns.EXCHANGE.value.name]],
        on=[TColumns.SYMBOL.value.name, TColumns.EXCHANGE.value.name],
        how="inner",
    )

    # Print the exchange with the highest trading volume
    if print_exchange:
        for row in filtered_df.iter_rows():
            symbol, exchange, _, _ = row
            print(f"Symbol: {symbol}, Selected Exchange: {EXCHANGES[exchange]}")

    return result_df.sort(TColumns.TIME.value.name)


def remove_zero_prices(t_data: pl.DataFrame) -> pl.DataFrame:
    """
    | Remove rows with zero prices from the DataFrame.

    :param t_data: polars DataFrame containing raw trade data.
    :return: DataFrame with rows containing zero prices removed.
    """

    # check data consistency
    t_data = check_column_names(t_data)
    t_data = check_trade_data(t_data)

    # return non zero price dataframes
    return t_data.filter(t_data[TColumns.PRICE.value.name] != 0).sort(TColumns.TIME.value.name)


def remove_zero_quotes(df: pl.DataFrame) -> pl.DataFrame:
    """
    | Remove rows with zero bid and offer prices from the DataFrame.

    :param df: polars DataFrame containing quote data.
    :return: DataFrame with rows containing zero quotes removed.
    """

    # check data consistency
    df = check_column_names(df)
    df = check_quote_data(df)

    # remove zero quotes and sort by time
    return df.filter(
        (df[QColumns.BID.value.name] != 0) & (df[QColumns.OFR.value.name] != 0)
    ).sort(QColumns.TIME.value.name)


def exchange_hours_only(
    t_data: pl.DataFrame,
    market_open: tuple[int] = (9, 30),
    market_close: tuple[int] = (16, 0),
) -> pl.DataFrame:
    """
    | Filter Raw Trade DataFrame to contain only data from regular trading hours.

    :param t_data: DataFrame containing raw trade data.
    :param market_open: Tuple containing the opening time of the market. First entry represents the hour, second the minute.
    :param market_close: Tuple containing the closing time of the market. First entry represents the hour, second the minute.
    :return: DataFrame containing only data from regular trading hours.
    """

    # check data consistency
    t_data = check_column_names(t_data)
    t_data = check_trade_data(t_data)

    # check market open and close are consistent
    if len(market_open) != 2 or len(market_close) != 2:
        raise ValueError("Market open and close must be tuples with two elements.")
    if not all(isinstance(item, int) for item in market_open):
        raise TypeError("All elements in market_open must be integers")
    if not all(isinstance(item, int) for item in market_close):
        raise TypeError("All elements in market_close must be integers")

    # generate the market open and close durations
    market_open_duration = pl.duration(hours=market_open[0], minutes=market_open[1])
    market_close_duration = pl.duration(hours=market_close[0], minutes=market_close[1])

    # filter the data
    df_filtered = t_data.filter(
        (
            pl.col(TColumns.TIME.value.name)
            >= pl.col(TColumns.TIME.value.name).dt.truncate("1d") + market_open_duration
        )
        & (
            pl.col(TColumns.TIME.value.name)
            <= pl.col(TColumns.TIME.value.name).dt.truncate("1d") + market_close_duration
        )
    )
    return df_filtered.sort(TColumns.TIME.value.name)


def remove_negative_spread(
    q_data: pl.DataFrame,
) -> pl.DataFrame:
    """
    | Remove rows with negative spread from the raw Quote data.

    :param q_data: DataFrame containing quote data.
    :return: DataFrame containing only data with positive spread.
    """

    # check data consistency
    q_data = check_column_names(q_data)
    q_data = check_quote_data(q_data)

    filtered_data = q_data.filter(pl.col(QColumns.OFR.value.name) > pl.col(QColumns.BID.value.name))

    return filtered_data.sort(QColumns.TIME.value.name)
