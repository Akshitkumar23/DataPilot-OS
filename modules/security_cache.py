import ast
import hashlib
from typing import Tuple, Optional, Any, Dict
import pandas as pd
import logging

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class ASTSecurityGuard:
    """
    Validates Python code by inspecting the Abstract Syntax Tree (AST) 
    to block forbidden imports and calls.
    """
    
    def __init__(self) -> None:
        """Initialize the ASTSecurityGuard class."""
        self.forbidden_modules = {
            "os", "sys", "subprocess", "socket", "shutil", "ctypes", "urllib",
            "requests", "pickle", "pty", "platform", "signal", "tempfile",
            "importlib", "threading", "compile", "code", "builtins"
        }
        self.forbidden_calls = {
            "eval", "exec", "open", "__import__", "getattr", "compile", "interact"
        }
        self.allowed_modules = {
            "pandas", "numpy", "seaborn", "matplotlib", 
            "sklearn", "scikit-learn", "statsmodels", "duckdb"
        }

    def validate_code(self, code_str: str) -> Tuple[bool, Optional[str]]:
        """
        Performs AST inspection to block forbidden imports/calls.

        Args:
            code_str (str): The Python code string to validate.

        Returns:
            Tuple[bool, Optional[str]]: A tuple containing a boolean indicating if the code is valid,
                                        and an error message if invalid (or None if valid).
        """
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            logger.error(f"Syntax error in code: {e}")
            return False, f"SyntaxError: {str(e)}"

        for node in ast.walk(tree):
            # Check for forbidden imports (e.g., import os)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    base_module = alias.name.split('.')[0]
                    if base_module in self.forbidden_modules:
                        msg = f"Forbidden import detected: {alias.name}"
                        logger.warning(msg)
                        return False, msg
            
            # Check for forbidden from-imports (e.g., from os import system)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    base_module = node.module.split('.')[0]
                    if base_module in self.forbidden_modules:
                        msg = f"Forbidden import detected: {node.module}"
                        logger.warning(msg)
                        return False, msg

            # Check for forbidden function calls (e.g., eval(), exec(), open())
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.forbidden_calls:
                        msg = f"Forbidden function call detected: {node.func.id}"
                        logger.warning(msg)
                        return False, msg
                elif isinstance(node.func, ast.Attribute):
                    # Check for calls like builtins.eval() or os.system()
                    if isinstance(node.func.value, ast.Name):
                        if node.func.value.id in self.forbidden_modules:
                            msg = f"Forbidden module call detected: {node.func.value.id}.{node.func.attr}"
                            logger.warning(msg)
                            return False, msg
                        
            # Prevent access to dunder methods/attributes that could bypass security
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith('__') and node.attr.endswith('__'):
                    msg = f"Forbidden attribute access detected: {node.attr}"
                    logger.warning(msg)
                    return False, msg

        return True, None


class ExecutionCache:
    """
    Caches and retrieves past Python/SQL execution results based on 
    dataset signature and query string.
    """
    
    def __init__(self) -> None:
        """Initialize the ExecutionCache class."""
        self._cache: Dict[str, Any] = {}

    def _generate_key(self, df: pd.DataFrame, query_string: str) -> str:
        """
        Generates a hash based on DataFrame shape, column names, and the query string.
        """
        shape_str = f"{df.shape[0]}x{df.shape[1]}"
        cols_str = ",".join(map(str, df.columns.tolist()))
        
        signature = f"{shape_str}|{cols_str}|{query_string}"
        return hashlib.sha256(signature.encode('utf-8')).hexdigest()

    def get(self, df: pd.DataFrame, query_string: str) -> Tuple[bool, Any]:
        """
        Retrieves a cached result if it exists.

        Args:
            df (pd.DataFrame): The context DataFrame.
            query_string (str): The executed code or SQL query.

        Returns:
            Tuple[bool, Any]: A tuple containing a boolean indicating a cache hit,
                              and the cached result (or None if miss).
        """
        try:
            key = self._generate_key(df, query_string)
            if key in self._cache:
                logger.info(f"Cache hit for query: {query_string[:50]}...")
                return True, self._cache[key]
            
            logger.info(f"Cache miss for query: {query_string[:50]}...")
            return False, None
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return False, None

    def set(self, df: pd.DataFrame, query_string: str, result: Any) -> bool:
        """
        Caches the result of an execution.

        Args:
            df (pd.DataFrame): The context DataFrame.
            query_string (str): The executed code or SQL query.
            result (Any): The execution result to cache.

        Returns:
            bool: True if caching was successful, False otherwise.
        """
        try:
            key = self._generate_key(df, query_string)
            self._cache[key] = result
            logger.info(f"Cached result for query: {query_string[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False

    def clear(self) -> None:
        """
        Clears the execution cache.
        """
        self._cache.clear()
        logger.info("Execution cache cleared.")


# Global Singleton Instance
security_guard = ASTSecurityGuard()


# Global Singleton Instance
execution_cache = ExecutionCache()
