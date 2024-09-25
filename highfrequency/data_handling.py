import polars as pl
from internal_data_handling import *
from typing import List
from helpers.exchange_names import EXCHANGES


def select_exchange(
        df: pl.DataFrame,
        exchange: str = "N",
        exchange_column: str = 'ex',
        time_column: str = 'dt') -> pl.DataFrame:
    """
    | Filter the data to retain only the data from the specified exchange. Be default the NYSE is selected.

    :param df: DataFrame containing data.
    :param exchange: Name of the exchange to filter the data for.
    :param exchange_column: Name of the column containing the exchange information.
    :param time_column: Name of the column containing the trade time.
    :return:
    """

    # Check if 'df' is a Polars DataFrame
    if not isinstance(df, pl.DataFrame):
        raise ValueError("Input has to be a polars DataFrame.")

    # Ensure required column names are present
    if exchange_column not in df.columns:
        raise ValueError(f"Data needs '{exchange_column}' column.")

    # Check for "dt" column if needed for further processing
    if time_column not in df.columns:
        raise ValueError(f"Data needs '{time_column}' column.")

    # ensure the exchange selected is one of the allowed exchanges
    if exchange not in EXCHANGES:
        raise ValueError(f"Exchange '{exchange}' not found in the list of exchanges")

    # Filter the data where 'ex' column matches the specified 'exchange' value
    filtered_data = df.filter(pl.col(exchange_column) == exchange)

    return filtered_data.sort(time_column)


def auto_select_exchange_trades(
        t_data: pl.DataFrame,
        volume_column: str = 'size',
        exchange_column: str = 'ex',
        time_column: str = 'dt',
        print_exchange: bool = False) -> pl.DataFrame:
    """
    | Retain only data from the stock exchange with the highest trading volume.

    | Filters raw trade data and return only data that stems from the exchange with the highest value for the variable
    *SIZE*, i.e. the highest trade volume. The function can use the default column names for the trade volume and the
    exchange, but it can also be customized. Additionally, the function can either use the pre-defined list of exchanges
    or a custom list.

    :param t_data: polars DataFrame containing trade data with at least columns *size* and *ex*.
    :param volume_column: Name of the column containing the trade volume.
    :param exchange_column: Name of the column containing the exchange.
    :param time_column: Name of the column containing the trade time.
    :param print_exchange: If True, print the exchange with the highest trading volume.
    :return: DataFrame containing only data from the exchange with the highest trading volume.
    """

    t_data = check_column_names(t_data)
    t_data = check_trade_data(t_data)

    if exchange_column not in t_data.columns:
        raise ValueError(f"Data needs '{exchange_column}' column.")

    # Step 1: Group by 'symbol' and 'exchange', and aggregate 'size' by summing
    aggregated_df = t_data.group_by(['symbol', exchange_column]).agg(pl.sum(volume_column).alias('total_size'))

    # Step 2: Find the maximum size exchange for each symbol
    max_size_df = aggregated_df.group_by('symbol').agg(
        pl.max('total_size').alias('max_size')
    )

    # Step 3: Join the maximum size with the original aggregated dataframe to filter the exchanges
    joined_df = aggregated_df.join(max_size_df, on='symbol', how='inner')

    # Step 4: Filter to keep only the exchanges with the maximum size for each symbol
    filtered_df = joined_df.filter(pl.col('total_size') == pl.col('max_size'))

    # Step 5: Join back to the original dataframe to get the original rows
    result_df = t_data.join(
        filtered_df[['symbol', exchange_column]],
        on=['symbol', exchange_column],
        how='inner'
    )

    # Print the exchange with the highest trading volume
    if print_exchange:
        for row in filtered_df.iter_rows():
            symbol, exchange, _, _ = row
            print(f"Symbol: {symbol}, Selected Exchange: {EXCHANGES[exchange]}")

    return result_df.sort(time_column)





def remove_zero_prices(
        df: pl.DataFrame,
        price_column: str = 'price',
        time_column: str = 'dt') -> pl.DataFrame:
    """
    | Remove rows with zero prices from the DataFrame.

    :param df: polars DataFrame containing trade data.
    :param price_column: Name of the column containing the trade price.
    :param time_column: Name of the column containing the trade time.
    :return: DataFrame with rows containing zero prices removed.
    """

    # check data consistency
    df = check_column_names(df)
    df = check_trade_data(df)

    # return non zero price dataframes
    return df.filter(df[price_column] != 0).sort(time_column)


def remove_zero_quotes(
        df: pl.DataFrame,
        bid_column: str = 'bid',
        ofr_column: str = 'ofr',
        time_column: str = 'dt') -> pl.DataFrame:
    """
    | Remove rows with zero bid and offer prices from the DataFrame.

    :param df: polars DataFrame containing quote data.
    :param bid_column: Name of the column containing the bid price.
    :param ofr_column: Name of the column containing the offer price.
    :param time_column: Name of the column containing the quote time.
    :return: DataFrame with rows containing zero quotes removed.
    """

    # check data consistency
    df = check_column_names(df)
    df = check_quote_data(df)

    # remove zero quotes and sort by time
    return df.filter((df[bid_column] != 0) & (df[ofr_column] != 0)).sort(time_column)


def exchange_hours_only(
        df: pl.DataFrame,
        market_open: tuple[int] = (9, 30),
        market_close: tuple[int] = (16, 0),
        time_column: str = 'dt') -> pl.DataFrame:
    """
    | Filter DataFrame to contain only data from regular trading hours.

    :param df: DataFrame containing trade data.
    :param market_open: Tuple containing the opening time of the market. First entry represents the hour, second the minute.
    :param market_close: Tuple containing the closing time of the market. First entry represents the hour, second the minute.
    :param time_column: Name of the column containing the trade time.
    :return: DataFrame containing only data from regular trading hours.
    """


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
    df_filtered = df.filter(
        (pl.col(time_column) >= pl.col(time_column).dt.truncate('1d') + market_open_duration) &
        (pl.col(time_column) <= pl.col(time_column).dt.truncate('1d') + market_close_duration)
    )
    return df_filtered.sort(time_column)


def remove_negative_spread(
        q_data: pl.DataFrame,
        time_column: str = 'dt',
        bid_column: str = 'bid',
        ofr_column: str = 'ofr'
) -> pl.DataFrame:
    """
    | Remove rows with negative spread from the DataFrame.

    :param q_data: DataFrame containing quote data.
    :param time_column: Name of the column containing the quote time.
    :param bid_column: Name of the column containing the bid price.
    :param ofr_column: Name of the column containing the offer price.
    :return: DataFrame containing only data with positive spread.
    """

    # Check if q_data is a polars DataFrame
    if not isinstance(q_data, pl.DataFrame):
        raise ValueError("Input has to be a polars DataFrame.")

    q_data = check_column_names(q_data)
    q_data = check_quote_data(q_data)

    filtered_data = q_data.filter(pl.col(ofr_column) > pl.col(bid_column))

    return filtered_data.sort(time_column)