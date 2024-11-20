import pandas as pd
import polars as pl
from .internal_data_handling import *
from typing import List
from .helpers.exchange_names import EXCHANGES
from .helpers.schemas import TColumns, QColumns
import warnings


# done and checked
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


# done and checked
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
    ).agg(pl.sum(TColumns.SIZE.value.name).alias("total_size"))

    # Step 2: Find the maximum size exchange for each symbol
    max_size_df = aggregated_df.group_by(TColumns.SYMBOL.value.name).agg(
        pl.max("total_size").alias("max_size")
    )

    # Step 3: Join the maximum size with the original aggregated dataframe to filter the exchanges
    joined_df = aggregated_df.join(max_size_df, on=TColumns.SYMBOL.value.name, how="inner")

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


# done and checked
def auto_select_exchange_quotes(
        q_data: pl.DataFrame, print_exchange: bool = False
) -> pl.DataFrame:
    """
    | Retain only data from the stock exchange with the highest quotes volume.

    | Filters raw quote data and return only data that stems from the exchange with the highest value for the variable
    *bidsiz* + *ofrdiz*, i.e. the highest quoted volume.

    :param q_data: polars DataFrame containing quote data with at least columns *bidsiz*, *ofrdiz* and *ex*.
    :param print_exchange: If True, print the exchange with the highest trading volume.
    :return: DataFrame containing only data from the exchange with the highest trading volume.
    """
    # check data consistency
    q_data = check_column_names(q_data)
    q_data = check_quote_data(q_data)

    # Step 1: Group by 'symbol' and 'exchange', and aggregate 'size' by summing
    aggregated_df = (
        q_data.with_columns(
            (pl.col(QColumns.OFRSIZ.value.name) + pl.col(QColumns.BIDSIZ.value.name)).alias("total_size_temp")
        )
        .group_by([QColumns.SYMBOL.value.name, QColumns.EXCHANGE.value.name])
        .agg(pl.sum("total_size_temp").alias("total_size"))
    )

    # Step 2: Find the maximum size exchange for each symbol
    max_size_df = aggregated_df.group_by(QColumns.SYMBOL.value.name).agg(
        pl.max("total_size").alias("max_size")
    )

    # Step 3: Join the maximum size with the original aggregated dataframe to filter the exchanges
    joined_df = aggregated_df.join(max_size_df, on=QColumns.SYMBOL.value.name, how="inner")

    # Step 4: Filter to keep only the exchanges with the maximum size for each symbol
    filtered_df = joined_df.filter(pl.col("total_size") == pl.col("max_size"))

    # Step 5: Join back to the original dataframe to get the original rows
    result_df = q_data.join(
        filtered_df[[QColumns.SYMBOL.value.name, QColumns.EXCHANGE.value.name]],
        on=[QColumns.SYMBOL.value.name, QColumns.EXCHANGE.value.name],
        how="inner",
    )

    # Print the exchange with the highest trading volume
    if print_exchange:
        for row in filtered_df.iter_rows():
            symbol, exchange, _, _ = row
            print(f"Symbol: {symbol}, Selected Exchange: {EXCHANGES[exchange]}")

    return result_df.sort(QColumns.TIME.value.name)


# done and checked
def no_zero_prices(t_data: pl.DataFrame) -> pl.DataFrame:
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


# done and checked
def no_zero_quotes(df: pl.DataFrame) -> pl.DataFrame:
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
    timezone: str | None = None,
) -> pl.DataFrame:
    """
    | Filter Raw Trade DataFrame to contain only data from regular trading hours. The function functions exactly the same
    if the datetime column also has a timezone associated with it.

    :param t_data: DataFrame containing raw trade data.
    :param market_open: Tuple containing the opening time of the market. First entry represents the hour, second the minute.
    :param market_close: Tuple containing the closing time of the market. First entry represents the hour, second the minute.
    :param timezone: Timezone of the data. If None, the data is assumed to be in UTC.
    :return: DataFrame containing only data from regular trading hours.
    """

    # check data consistency
    #t_data = check_column_names(t_data)
    #t_data = check_trade_data(t_data)

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
    if timezone:
        # adjust the timezone leaving the timestamp unchanged
        df_filtered = df_filtered.with_columns(
            pl.col(TColumns.TIME.value.name).dt.replace_time_zone(timezone)
        )

    return df_filtered.sort(TColumns.TIME.value.name)


# done and checked
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


def gather_prices(data: pl.DataFrame) -> pl.DataFrame:
    """
    | Convenience function to gather data from one Dataframe with at least a column 'dt', and d columns with prices.
    The function returns a DataFrame with three columns: 'dt', 'symbol', and 'price'.

    :param data: DataFrame a column dt and d columns with prices.
    :return: DataFrame with columns 'dt', 'symbol', and 'price'.
    """

    # Check if the input is a polars DataFrame
    if not isinstance(data, pl.DataFrame):
        raise ValueError("Input must be a polars DataFrame.")

    # Check for the 'dt' column
    if "dt" not in data.columns:
        raise ValueError("Data must contain a dt column.")

    melted_data = data.unpivot(index="dt",
                               variable_name="symbol",
                               value_name="price")

    melted_data = melted_data.sort(["dt", "symbol"])

    return melted_data


def spread_prices(data: pd.DataFrame) -> pl.DataFrame:
    """
    | Convenience function to spread data from a DataFrame with columns 'dt', 'symbol', and 'price' into a DataFrame
    where each symbol has its own column.

    :param data: DataFrame with columns 'dt', 'symbol', and 'price'.
    :return: DataFrame where each symbol has its own column.
    """

    if not isinstance(data, pl.DataFrame):
        raise ValueError("Input must be a polars DataFrame.")

    required_columns = ["dt", "symbol", "price"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in data: {', '.join(missing_columns)}. These columns must be present.")

    # Pivot the data to spread SYMBOL into separate columns
    spread_data = data.pivot(
        index="dt",
        on="symbol",
        values="price"
    )

    # Sort the resulting DataFrame by 'DT' to ensure chronological order
    spread_data = spread_data.sort("dt")

    return spread_data


# done and checked
def trade_condition(
        t_data: pl.DataFrame,
        valid_conds: List[str] = ('', '@', 'E', '@E', 'F', 'FI', '@F', '@FI', 'I', '@I')
) -> pl.DataFrame:
    """
    | Filter the Raw Trade data to retain only the data with valid trade conditions.

    :param t_data: DataFrame containing raw trade data.
    :param valid_conds: List of valid trade conditions.
    :return:
    """

    # check data consistency
    t_data = check_column_names(t_data)
    t_data = check_trade_data(t_data)

    if TColumns.CONDITION.value.name not in t_data.columns:
        raise KeyError("Data must contain a cond column.")

    # fill missing values with '' for unknown
    t_data = t_data.with_columns(
        pl.col(TColumns.CONDITION.value.name).fill_null("").alias(TColumns.CONDITION.value.name)
    )

    # filter the data
    t_data = t_data.filter(
        pl.col(TColumns.CONDITION.value.name).str.replace_all(r"\s", "").is_in(valid_conds)
    )

    return t_data.sort(TColumns.TIME.value.name)


def refresh_time(
        p_data: List[pl.DataFrame] | dict[str, pl.DataFrame],
        sort=False,
        criterion='squared duration'):
    """
    | Synchronize (multiple) irregular timeseries by refresh time.

    | This function implements the refresh time synchronization scheme proposed by Harris et al. (1995).
    It picks the so-called refresh times at which all assets have traded at least once since the last refresh time point.
    For example, the first refresh time corresponds to the first time at which all stocks have traded.
    The subsequent refresh time is defined as the first time when all stocks have traded again.
    This process is repeated until the end of one time series is reached.

    References
    ----------
    Harris, F.H.D., McInish, T.H., Shoesmith, G.L., & Wood, R.A. (1995). Cointegration, error correction, and price \
    discovery on informationally linked security markets. Journal of Financial and Quantitative Analysis, 30(4), \
    563-579. `doi:10.2307/2331277 <https://doi.org/10.2307/2331277>`_.

    :param p_data: Dictionary or list of Polars DataFrames containing the data to be synchronized.
    :param sort: If True, sort the data according to the criterion.
    :param criterion: The criterion to use for sorting the data. Must be either 'squared duration' or 'duration'.
    :return: Synchronized data.
    """

    # Input validation
    if not isinstance(p_data, (list, dict)) or len(p_data) < 1:
        raise ValueError("p_data must be a list or dictionary with at least one element")

    if isinstance(p_data, dict):
        data_items = list(p_data.items())
        data_names = [name for name, _ in data_items]
        p_data_list: List[pl.DataFrame] = [df for _, df in data_items]
    else:
        p_data_list: List[pl.DataFrame] = p_data
        data_names = None

    # Check that all elements of p_data are Polars DataFrames
    if not all(isinstance(df, pl.DataFrame) for df in p_data_list):
        raise TypeError("All elements of p_data must be Polars DataFrames")

    # Check that each DataFrame has exactly two columns and contains data for a single day
    for df in p_data_list:
        if df.shape[1] != 2:
            raise ValueError("Each DataFrame must have exactly two columns")
        if TColumns.TIME.value.name not in df.columns:
            raise ValueError("Each DataFrame must have a dt column")
        # TODO: Shouldn't we check that the date is the same for all timeseries?
        dates = df.select(pl.col(TColumns.TIME.value.name).dt.date()).unique().to_series()
        if len(dates) > 1:
            raise ValueError("All DataFrames must contain data for a single day")

    # check there are at least two series
    if len(p_data_list) < 1:
        raise ValueError("At least two series are required for synchronization")
    if len(p_data_list) == 1:
        return p_data_list[0]

    if sort and not data_names:
        raise TypeError("When using sort, please provide p_data as a dictionary with names")

    # sort the data if required
    if sort:
        # Define the sorting criterion function
        if criterion == 'squared duration':
            def compute_criterion(df):
                durations = df.sort(TColumns.TIME.value.name).select(
                    (pl.col(TColumns.TIME.value.name).shift(-1) - pl.col(TColumns.TIME.value.name))
                )[:-1][TColumns.TIME.value.name].to_list()
                return sum([delta.total_seconds()**2 for delta in durations])
        elif criterion == 'duration':
            def compute_criterion(df):
                durations = df.sort(TColumns.TIME.value.name).select(
                    (pl.col(TColumns.TIME.value.name).shift(-1) - pl.col(TColumns.TIME.value.name))
                )[:-1][TColumns.TIME.value.name].to_list()
                return sum([delta.total_seconds() for delta in durations])
        else:
            raise ValueError("Criterion must be either 'squared duration' or 'duration'")

        # Compute the criterion for each series
        criteria = {name: compute_criterion(df) for name, df in zip(data_names, p_data_list)}
        sorted_names = sorted(criteria, key=criteria.get)
        # Sort the data accordingly
        p_data_list = [p_data[name] for name in sorted_names]
        data_names = sorted_names

    if not data_names:
        # check if names are unique
        if len({col for df in p_data_list for col in df.columns if col != TColumns.TIME.value.name}) == len(p_data_list):
            data_names = [col for df in p_data_list for col in df.columns if col != TColumns.TIME.value.name]
        else:
            data_names = ['price_' + str(i) for i in range(len(p_data_list))]

    # Rename the columns
    for i, df in enumerate(p_data_list):
        df = df.sort(TColumns.TIME.value.name)
        value_col = [col for col in df.columns if col != TColumns.TIME.value.name][0]
        df = df.rename({value_col: data_names[i]})
        p_data_list[i] = df

    # Merge all dataframes on TColumns.TIME.value.name using an outer join
    merged_df = p_data_list[0]
    for df in p_data_list[1:]:
        merged_df = merged_df.join(df, on=TColumns.TIME.value.name, how='full', coalesce=True)

    # Sort merged dataframe by TColumns.TIME.value.name
    merged_df = merged_df.sort(TColumns.TIME.value.name)


    # generate column with the 1 for non-null values and 0 for null values
    cum_counts = [
        merged_df[col].is_not_null().cast(pl.Int8).alias(f'mark_{col}')
        for col in data_names
    ]

    # get refresh times
    # TODO: This can be optimized further with cython or numba perhaps
    merged_df = merged_df.with_columns(cum_counts)
    data_updates = merged_df[[f'mark_{col}' for col in data_names]].to_dicts()
    refresh_times = []
    updates = {key: False for key in data_updates[0].keys()}
    prev_refresh = False
    for d in data_updates:
        if prev_refresh:
            updates = d.copy()
        else:
            updates = {key: any([updates[key], d[key]]) for key in updates}
        if all(updates.values()):
            prev_refresh = True
        else:
            prev_refresh = False
        refresh_times.append(prev_refresh)
    merged_df = merged_df.with_columns(pl.Series("refresh", refresh_times))
    merged_df = merged_df.filter(pl.col("refresh"))
    merged_df = merged_df[[TColumns.TIME.value.name] + data_names]

    return merged_df


def remove_large_spread(
        q_data: pl.DataFrame,
        maximum_spread: float = 50,
        time_zone: str | None = None
) -> pl.DataFrame:
    """
    | Delete entries for which the spread is more than **maxi** times the median spread.

    :param q_data: DataFrame containing quote data.
    :param maximum_spread: Maximum spread allowed.
    :param time_zone: Timezone of the data. If None, the data is assumed to be in UTC.
    :return: DataFrame with rows containing large spreads removed.
    """

    q_data = check_column_names(q_data)
    q_data = check_quote_data(q_data)

    # Extract DATE and Calculate SPREAD
    q_data = q_data.with_columns([
        pl.col(QColumns.TIME.value.name).dt.date().alias('DATE'),
        (pl.col(QColumns.OFR.value.name) - pl.col(QColumns.BID.value.name)).alias('SPREAD')
    ])

    # Calculate the median spread per day
    median_spread = q_data.group_by('DATE').agg([
        pl.col('SPREAD').median().alias('SPREAD_MEDIAN')
    ])

    # Merge the median spread with the original data
    q_data = q_data.join(median_spread, on='DATE', how='left')
    threshold = pl.col('SPREAD_MEDIAN') * maximum_spread

    # Filter the data
    q_data = q_data.filter(pl.col('SPREAD') < threshold)

    # Drop Auxiliary Columns
    q_data = q_data.drop(['DATE', 'SPREAD', 'SPREAD_MEDIAN'])

    return q_data.sort(QColumns.TIME.value.name)


def merge_trades_same_timestamp(
        t_data: pl.DataFrame,
        selection: str = "median"
):
    """
    | Merge trade entries that have the same timestamp to a single one.

    :param t_data: DataFrame containing raw trade data.
    :param selection: Method to select the price. Can be either 'median' or 'max_volume' or 'weighted_average'.
    :return: DataFrame with merged trade entries.
    """

    t_data = check_column_names(t_data)
    t_data = check_trade_data(t_data)

    # SValidate 'selection' Parameter
    valid_selections = ["median", "max_volume", "weighted_average"]
    if selection not in valid_selections:
        raise ValueError(f"Selection has to be one of {valid_selections}, but got '{selection}'")
    pass



