import pandas as pd
import json
from agents.agent_base import BaseAgent
from tools.cleaning_tools import inspect_metadata, get_column_stats, impute_missing, drop_column

CLEANING_TOOLS_DECLARATIONS = [
    {
        "name": "inspect_metadata",
        "description": "Returns shape, data types, and null counts of the dataset.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "get_column_stats",
        "description": "Returns distribution or unique values for a specific column.",
        "parameters": {
            "type": "object",
            "properties": {
                "col": {"type": "string", "description": "The column name to analyze."}
            },
            "required": ["col"]
        }
    },
    {
        "name": "impute_missing",
        "description": "Fills missing values (NaNs) in a column using a specific strategy.",
        "parameters": {
            "type": "object",
            "properties": {
                "col": {"type": "string", "description": "The column name."},
                "strategy": {"type": "string", "enum": ["mean", "median", "mode"], "description": "The imputation strategy."}
            },
            "required": ["col", "strategy"]
        }
    },
    {
        "name": "drop_column",
        "description": "Removes a column from the dataset.",
        "parameters": {
            "type": "object",
            "properties": {
                "col": {"type": "string", "description": "The column name to drop."}
            },
            "required": ["col"]
        }
    }
]

SYSTEM_PROMPT = """You are 'The Auditor', a Data Cleaning Agent.
Your goal is to ensure the dataset is technically sound.
You must:
1. Inspect the metadata to understand the structure.
2. Identify missing values, wrong data types, or high cardinality columns.
3. Decide which columns to drop or impute.
4. Once finished, say 'CLEANING_COMPLETE' and provide a summary of your actions.

You have access to the dataset through the provided tools."""

class DataCleanerAgent(BaseAgent):
    def __init__(self, df):
        super().__init__(
            name="The Auditor",
            role="Data Cleaner",
            system_prompt=SYSTEM_PROMPT,
            tools_declarations=CLEANING_TOOLS_DECLARATIONS
        )
        self.df = df
        self.actions_taken = []

    def execute_tool(self, func_name, args):
        if func_name == "inspect_metadata":
            return inspect_metadata(self.df)
        elif func_name == "get_column_stats":
            return get_column_stats(self.df, args["col"])
        elif func_name == "impute_missing":
            self.df, msg = impute_missing(self.df, args["col"], args["strategy"])
            self.actions_taken.append(msg)
            return msg
        elif func_name == "drop_column":
            self.df, msg = drop_column(self.df, args["col"])
            self.actions_taken.append(msg)
            return msg
        return f"Unknown tool: {func_name}"
