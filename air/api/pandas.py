"""
This module provides a DataFrame client capable of
executing queries on pandas DataFrames and returning the results efficiently.
"""

import json
import os
import traceback
from typing import Any

import pandas as pd


class PandasAPI:
    """
    The class for Dataframe client
    """

    def __init__(self, df_metadata_list: list[dict]) -> None:
        """
        Initializes the dataframe API object.

        Args:
            df_metadata_list: list of (name, file_path) tuples of dataframes.
        """

        self.dataframes = {}

        # Load each table into a DataFrame and store it in the dictionary
        for table in df_metadata_list:
            table_name, file_path = table["name"], table["file_path"]

            # Determine the file extension
            file_extension = os.path.splitext(file_path)[1].lower()

            # Load the file into a DataFrame based on its extension
            if file_extension == ".csv":
                df = pd.read_csv(file_path)
            # elif file_extension in ['.xls', '.xlsx']:
            #     df = pd.read_excel(file_path)
            elif file_extension == ".parquet":
                df = pd.read_parquet(file_path)
            else:
                print(f"Unsupported file type: {file_extension}")
                continue

            # Store the DataFrame in the dictionary
            self.dataframes["default_" + table_name] = df

    async def execute_query(
        self, query: str = "", params: dict = {}
    ) -> tuple[list[Any] | str, bool]:
        """
        Executes a dataframe query safely.

        Parameters:
        ----------
        query : str
            The query string to evaluate.
        params : Optional
            Additional parameters for the query.

        Returns:
        -------
        tuple[Any, bool]
            A tuple containing the query results as a list of tuples and a success flag.
            Returns (None, False) if execution fails.
        """

        return_stringified_result = params.get("return_stringified_result", False)

        query_results = []
        try:
            # Create a local namespace with only the pandas module
            local_namespace = self.dataframes.copy()
            local_namespace["pd"] = pd

            # Whitelist safe built-in functions
            safe_builtins = {
                "bool": bool,
                "int": int,
                "float": float,
                "str": str,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "len": len,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "sorted": sorted,
                "map": map,
                "filter": filter,
                "zip": zip,
                "range": range,
                "enumerate": enumerate,
                "all": all,
                "any": any,
            }
            restricted_globals = {"__builtins__": safe_builtins}

            query_results = eval(query, restricted_globals, local_namespace)

            if isinstance(query_results, pd.Series):
                query_results = query_results.to_frame()

            if not isinstance(query_results, pd.DataFrame):  # if it is a scalar value
                query_results = pd.DataFrame([query_results])

            if return_stringified_result:
                # convert result df to a json string and return
                return [json.dumps(query_results.to_json(orient="index"))], True

            return [tuple(row) for row in query_results.itertuples(index=False)], True

        except Exception:
            print("The following query has failed:", query)
            error_traceback = traceback.format_exc()
            print(error_traceback)
            return error_traceback, False
