import pytest
import polars as pl

from highfrequency.internal_data_handling import check_quote_data, check_trade_data, check_column_names

@pytest.mark.database
class TestInternalDataHandling:
    def test_quote_is_polars_dataframe(self):

        # test dataframe
        df = 'Not a DataFrame'
        with pytest.raises(TypeError):
            df = check_quote_data(df)

    def test_column_names_quotes(self):

        # test column names
        df = pl.DataFrame(
            {'DT': [1],
             'BID': [1],
             'ASK': [1],
             'BIDSIZE': [1],
             'ASKSIZE': [1],
             'SYM_ROOT': [1],
             'TR_SCOND': [1],
                'CR': [1],
             })
        df = check_column_names(df)
        assert df.columns == ['dt', 'bid', 'ofr', 'bidsiz', 'ofrsiz', 'symbol', 'cond', 'corr']
        
    def test_column_names_trades(self):

        # test column names
        df = pl.DataFrame(
            {'DT': [1],
             'PRICE': [1],
             'SIZE': [1],
             'SYMBOL': [1],
             'TR_SCOND': [1],
             'EX': [1],
             'CORR': [1],
             })
        df = check_column_names(df)
        assert df.columns == ['dt', 'price', 'size', 'symbol', 'cond', 'ex', 'corr']

    def test_check_quote_data(self):

        # test dt column
        df = pl.DataFrame({'time': [1, 2, 3], 'bid': [1, 2, 3], 'ofr': [1, 2, 3]})
        try:
            df = check_quote_data(df)
        except KeyError as e:
            assert str(e) == "'Column dt not found in DataFrame columns.'"

        # test dt column type
        df = pl.DataFrame({'dt': [1, 2, 3], 'bid': [1, 2, 3], 'ofr': [1, 2, 3]})
        try:
            df = check_quote_data(df)
        except TypeError as e:
            assert str(e) == "Column dt should be of type Datetime."


        # test bidsize column
        df = pl.DataFrame({'dt': ["2018-01-02 05:01:21.479"], 'ofr': [1]})
        df = df.with_columns(
            pl.col('dt').str.to_datetime("%Y-%m-%d %H:%M:%S%.6f")
        )
        try:
            df = check_quote_data(df)
        except KeyError as e:
            assert str(e) == "'Column bid not found in DataFrame columns.'"

        # test ofr column
