import logging
from typing import List, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedSQLEngine:
    """
    Advanced SQL Engine for generating complex DuckDB SQL queries.
    """

    @staticmethod
    def generate_window_sql(table_name: str, partition_col: str, order_col: str, function_name: str) -> str:
        """
        Generates DuckDB SQL queries with window functions like ROW_NUMBER(), RANK(), etc.

        Args:
            table_name (str): The name of the table.
            partition_col (str): The column to partition by.
            order_col (str): The column to order by.
            function_name (str): The window function to use (e.g., 'ROW_NUMBER()', 'RANK()').

        Returns:
            str: The generated SQL query.

        Raises:
            ValueError: If invalid input parameters are provided.
        """
        if not all([table_name, partition_col, order_col, function_name]):
            logger.error("All parameters must be provided to generate window SQL.")
            raise ValueError("All parameters must be provided to generate window SQL.")
        
        valid_functions = ['ROW_NUMBER', 'RANK', 'DENSE_RANK', 'LAG', 'LEAD', 'NTILE']
        base_func = function_name.split('(')[0].upper()
        if base_func not in valid_functions:
            logger.warning(f"Window function {function_name} might not be a standard DuckDB window function.")

        sql = f"""
        SELECT *,
               {function_name} OVER (PARTITION BY {partition_col} ORDER BY {order_col}) AS window_result
        FROM {table_name};
        """
        logger.info(f"Generated window SQL for table {table_name}")
        return sql.strip()

    @staticmethod
    def generate_cte_sql(table_name: str, cte_queries: List[Dict[str, str]]) -> str:
        """
        Generates chained CTE SQL queries.

        Args:
            table_name (str): The final table to select from (usually the last CTE name or a target table).
            cte_queries (List[Dict[str, str]]): A list of dictionaries containing 'name' and 'query' for each CTE.

        Returns:
            str: The generated SQL query.
            
        Raises:
            ValueError: If cte_queries is empty or improperly formatted.
        """
        if not cte_queries:
            logger.error("No CTE queries provided.")
            raise ValueError("No CTE queries provided.")

        cte_strings = []
        for cte in cte_queries:
            if 'name' not in cte or 'query' not in cte:
                logger.error("CTE definition missing 'name' or 'query'.")
                raise ValueError("Each CTE must have a 'name' and 'query'.")
            cte_strings.append(f"{cte['name']} AS (\n    {cte['query']}\n)")

        cte_clause = "WITH " + ",\n".join(cte_strings)
        
        final_select = f"SELECT * FROM {table_name};"
        
        sql = f"{cte_clause}\n{final_select}"
        logger.info(f"Generated CTE SQL with {len(cte_queries)} CTEs.")
        return sql

    @staticmethod
    def generate_case_when_sql(table_name: str, target_col: str, conditions: List[Tuple[str, str]]) -> str:
        """
        Generates SQL CASE WHEN statements.

        Args:
            table_name (str): The name of the table.
            target_col (str): The new column name for the CASE WHEN result.
            conditions (List[Tuple[str, str]]): A list of tuples containing (condition, result).

        Returns:
            str: The generated SQL query.
            
        Raises:
            ValueError: If conditions list is empty.
        """
        if not conditions:
            logger.error("Conditions must be provided for CASE WHEN.")
            raise ValueError("Conditions must be provided for CASE WHEN.")

        when_clauses = "\n            ".join([f"WHEN {cond} THEN {res}" for cond, res in conditions])
        
        sql = f"""
        SELECT *,
            CASE
            {when_clauses}
            ELSE NULL
            END AS {target_col}
        FROM {table_name};
        """
        logger.info(f"Generated CASE WHEN SQL for table {table_name}")
        return sql.strip()

    @staticmethod
    def generate_relational_join_sql(table1: str, table2: str, join_type: str, on_keys: Tuple[str, str]) -> str:
        """
        Generates relational JOIN SQL queries.

        Args:
            table1 (str): The first table name.
            table2 (str): The second table name.
            join_type (str): The type of join (e.g., 'INNER', 'LEFT', 'RIGHT', 'FULL').
            on_keys (Tuple[str, str]): The keys to join on from table1 and table2 respectively.

        Returns:
            str: The generated SQL query.
            
        Raises:
            ValueError: If an invalid join type is provided.
        """
        valid_joins = ['INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN', 'INNER', 'LEFT', 'RIGHT', 'FULL']
        if join_type.upper() not in valid_joins:
            logger.error(f"Invalid join type: {join_type}")
            raise ValueError(f"Invalid join type: {join_type}. Must be one of {valid_joins}.")

        if "JOIN" not in join_type.upper():
            join_clause = f"{join_type.upper()} JOIN"
        else:
            join_clause = join_type.upper()

        if len(on_keys) != 2:
            logger.error("on_keys must contain exactly two keys.")
            raise ValueError("on_keys must contain exactly two keys.")

        key1, key2 = on_keys
        sql = f"""
        SELECT *
        FROM {table1}
        {join_clause} {table2}
        ON {table1}.{key1} = {table2}.{key2};
        """
        logger.info(f"Generated {join_clause} SQL for {table1} and {table2}")
        return sql.strip()


# Global Singleton Instance
sql_engine = AdvancedSQLEngine()
