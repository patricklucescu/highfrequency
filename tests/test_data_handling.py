import pytest
import polars as pl

from highfrequency.data_handling import *

@pytest.mark.datahandling
class TestInternalDataHandling:

    def test_remove_negative_spread(self):
        df = pl.DataFrame({
            'dt': ["2018-01-02 05:01:21.479", "2018-01-02 05:01:22.479"],
            'bid': [1.1, 2.1],
            'ofr': [2.1, 1.1],
            'bidsiz': [1, 2],
            'ofrsiz': [2, 1],
            'symbol': ['AAPL', 'AAPL'],
            'ex': ['NASDAQ', 'NASDAQ']
        })
        df = df.with_columns(
            pl.col('dt').str.to_datetime("%Y-%m-%d %H:%M:%S%.6f")
        )
        expected_df = pl.DataFrame({
            'dt': ["2018-01-02 05:01:21.479"],
            'bid': [1.1],
            'ofr': [2.1],
            'bidsiz': [1],
            'ofrsiz': [2],
            'symbol': ['AAPL'],
            'ex': ['NASDAQ']
        })
        expected_df = expected_df.with_columns(
            pl.col('dt').str.to_datetime("%Y-%m-%d %H:%M:%S%.6f")
        )

        df = remove_negative_spread(df)
        assert df.equals(expected_df)

    def test_select_exchange(self):
        pass