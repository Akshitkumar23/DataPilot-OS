import pandas as pd
import numpy as np
import logging
import sqlite3
import duckdb
from typing import Union, List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class DataEngineeringEngine:
    """
    DataEngineeringEngine provides production-ready helper functions for data import,
    merging, reshaping, column operations, and datetime operations.
    """
    
    @staticmethod
    def load_data(filepath: str, format_type: str, **kwargs) -> pd.DataFrame:
        """
        Data import wrappers for CSV, Excel, Parquet, JSON, Feather, SQL, DuckDB.
        """
        format_type = format_type.lower()
        try:
            if format_type == 'csv':
                return pd.read_csv(filepath, **kwargs)
            elif format_type == 'excel':
                return pd.read_excel(filepath, **kwargs)
            elif format_type == 'parquet':
                return pd.read_parquet(filepath, **kwargs)
            elif format_type == 'json':
                return pd.read_json(filepath, **kwargs)
            elif format_type == 'feather':
                return pd.read_feather(filepath, **kwargs)
            elif format_type == 'sheets':
                if "docs.google.com/spreadsheets" in filepath and "/edit" in filepath:
                    filepath = filepath.split("/edit")[0] + "/export?format=csv"
                return pd.read_csv(filepath, **kwargs)
            elif format_type == 'sql':
                conn = sqlite3.connect(filepath)
                query = kwargs.get('query', "SELECT * FROM data")
                df = pd.read_sql(query, conn)
                conn.close()
                return df
            elif format_type == 'duckdb':
                query = kwargs.get('query')
                if not query:
                    raise ValueError("A 'query' must be provided for duckdb format.")
                conn = duckdb.connect(database=filepath)
                df = conn.execute(query).df()
                conn.close()
                return df
            else:
                raise ValueError(f"Unsupported format_type: {format_type}")
        except Exception as e:
            logger.error(f"Error loading data from {filepath} with format {format_type}: {e}")
            raise

    @staticmethod
    def merge_datasets(
        left: pd.DataFrame, 
        right: pd.DataFrame, 
        on: Union[str, List[str]], 
        how: str = 'inner', 
        **kwargs
    ) -> pd.DataFrame:
        """Merge two datasets."""
        try:
            return pd.merge(left, right, on=on, how=how, **kwargs)
        except Exception as e:
            logger.error(f"Error merging datasets: {e}")
            raise

    @staticmethod
    def concat_datasets(objs: List[pd.DataFrame], axis: int = 0, **kwargs) -> pd.DataFrame:
        """Concatenate a list of datasets."""
        try:
            return pd.concat(objs, axis=axis, **kwargs)
        except Exception as e:
            logger.error(f"Error concatenating datasets: {e}")
            raise

    @staticmethod
    def join_datasets(left: pd.DataFrame, right: pd.DataFrame, how: str = 'left', **kwargs) -> pd.DataFrame:
        """Join two datasets on their index."""
        try:
            return left.join(right, how=how, **kwargs)
        except Exception as e:
            logger.error(f"Error joining datasets: {e}")
            raise

    @staticmethod
    def melt_dataframe(
        df: pd.DataFrame, 
        id_vars: Union[str, List[str]], 
        value_vars: Optional[Union[str, List[str]]] = None, 
        **kwargs
    ) -> pd.DataFrame:
        """Melt a DataFrame into a format where one or more columns are identifier variables."""
        try:
            return pd.melt(df, id_vars=id_vars, value_vars=value_vars, **kwargs)
        except Exception as e:
            logger.error(f"Error melting dataframe: {e}")
            raise

    @staticmethod
    def pivot_dataframe(
        df: pd.DataFrame, 
        index: Union[str, List[str]], 
        columns: Union[str, List[str]], 
        values: Union[str, List[str]], 
        **kwargs
    ) -> pd.DataFrame:
        """Create a spreadsheet-style pivot table as a DataFrame."""
        try:
            return df.pivot_table(index=index, columns=columns, values=values, **kwargs)
        except Exception as e:
            logger.error(f"Error pivoting dataframe: {e}")
            raise

    @staticmethod
    def unpivot_dataframe(
        df: pd.DataFrame, 
        id_vars: Union[str, List[str]], 
        value_vars: Optional[Union[str, List[str]]] = None, 
        **kwargs
    ) -> pd.DataFrame:
        """Unpivot a DataFrame (alias for melt)."""
        return DataEngineeringEngine.melt_dataframe(df, id_vars, value_vars, **kwargs)

    @staticmethod
    def wide_to_long(
        df: pd.DataFrame, 
        stubnames: Union[str, List[str]], 
        i: Union[str, List[str]], 
        j: str, 
        **kwargs
    ) -> pd.DataFrame:
        """Wide panel to long format."""
        try:
            return pd.wide_to_long(df, stubnames=stubnames, i=i, j=j, **kwargs)
        except Exception as e:
            logger.error(f"Error in wide_to_long: {e}")
            raise

    @staticmethod
    def rename_columns(df: pd.DataFrame, columns: Dict[str, str]) -> pd.DataFrame:
        """Rename columns of a DataFrame."""
        try:
            return df.rename(columns=columns)
        except Exception as e:
            logger.error(f"Error renaming columns: {e}")
            raise

    @staticmethod
    def split_column(df: pd.DataFrame, column: str, delimiter: str, new_columns: List[str]) -> pd.DataFrame:
        """Split a string column into multiple columns based on a delimiter."""
        try:
            df_new = df.copy()
            df_new[new_columns] = df_new[column].str.split(delimiter, expand=True)
            return df_new
        except Exception as e:
            logger.error(f"Error splitting column {column}: {e}")
            raise

    @staticmethod
    def combine_columns(df: pd.DataFrame, columns: List[str], new_column: str, separator: str = '_') -> pd.DataFrame:
        """Combine multiple columns into a single string column."""
        try:
            df_new = df.copy()
            df_new[new_column] = df_new[columns].astype(str).agg(separator.join, axis=1)
            return df_new
        except Exception as e:
            logger.error(f"Error combining columns {columns}: {e}")
            raise

    @staticmethod
    def regex_extract(df: pd.DataFrame, column: str, pattern: str, new_column: str) -> pd.DataFrame:
        """Extract capture groups in the regex pat as columns in a DataFrame."""
        try:
            df_new = df.copy()
            df_new[new_column] = df_new[column].str.extract(pattern, expand=False)
            return df_new
        except Exception as e:
            logger.error(f"Error in regex_extract for column {column}: {e}")
            raise

    @staticmethod
    def replace_values(df: pd.DataFrame, column: str, to_replace: Any, value: Any) -> pd.DataFrame:
        """Replace values in a column."""
        try:
            df_new = df.copy()
            df_new[column] = df_new[column].replace(to_replace, value)
            return df_new
        except Exception as e:
            logger.error(f"Error replacing values in column {column}: {e}")
            raise

    @staticmethod
    def extract_datetime_features(df: pd.DataFrame, datetime_column: str, country: str = 'US') -> pd.DataFrame:
        """
        Extract datetime features from a datetime column:
        Year, Quarter, Month, Weekday, Hour, Business Day, Holiday flags.
        """
        import holidays
        try:
            df_new = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_new[datetime_column]):
                df_new[datetime_column] = pd.to_datetime(df_new[datetime_column])
            
            series = df_new[datetime_column]
            df_new[f'{datetime_column}_year'] = series.dt.year
            df_new[f'{datetime_column}_quarter'] = series.dt.quarter
            df_new[f'{datetime_column}_month'] = series.dt.month
            df_new[f'{datetime_column}_weekday'] = series.dt.weekday
            df_new[f'{datetime_column}_hour'] = series.dt.hour
            
            # Business day: weekday < 5 (Monday=0, Sunday=6)
            df_new[f'{datetime_column}_is_business_day'] = series.dt.weekday < 5
            
            # Holiday flag
            country_holidays = holidays.country_holidays(country)
            df_new[f'{datetime_column}_is_holiday'] = series.dt.date.apply(lambda d: d in country_holidays)
            
            return df_new
        except Exception as e:
            logger.error(f"Error extracting datetime features from {datetime_column}: {e}")
            raise


# Global Singleton Instance
data_engineering = DataEngineeringEngine()
