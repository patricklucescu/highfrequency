from enum import Enum
from collections import namedtuple
import polars as pl

# Define named tuple with dtype defaulting to None
ColumnInfo = namedtuple('ColumnInfo', ['name', 'dtype'])
ColumnInfo.__new__.__defaults__ = (None,)  # Set default for dtype to None


class TColumns(Enum):
    """
    | Enum class to map the trade data names to the classes as well as data types.
    """
    TIME = ColumnInfo(name='dt', dtype=pl.Datetime)
    PRICE = ColumnInfo(name='price', dtype=pl.Float64)
    SIZE = ColumnInfo(name='size', dtype=[pl.Float64, pl.Int64])
    SYMBOL = ColumnInfo(name='symbol', dtype=pl.String)
    CONDITION = ColumnInfo(name='cond', dtype=pl.String)
    EXCHANGE = ColumnInfo(name='ex', dtype=pl.String)
    CORR = ColumnInfo(name='corr', dtype=[pl.Float64, pl.Int64])

class QColumns(Enum):
    """
    | Enum class to map the quote data names to the classes as well as data types.
    """
    TIME = ColumnInfo(name='dt', dtype=pl.Datetime)
    BID = ColumnInfo(name='bid', dtype=pl.Float64)
    OFR = ColumnInfo(name='ofr', dtype=pl.Float64)
    BIDSIZ = ColumnInfo(name='bidsiz', dtype=[pl.Float64, pl.Int64])
    OFRSIZ = ColumnInfo(name='ofrsiz', dtype=[pl.Float64, pl.Int64])
    SYMBOL = ColumnInfo(name='symbol', dtype=pl.String)
    EXCHANGE = ColumnInfo(name='ex', dtype=pl.String)


# RTAQ format
column_name_mapper = {
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