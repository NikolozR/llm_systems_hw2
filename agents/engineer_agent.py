import pandas as pd
import json
from agents.agent_base import BaseAgent
from tools.engineering_tools import create_interaction, encode_categorical, correlation_analysis, select_top_features
from tools.cleaning_tools import inspect_metadata

ENGINEERING_TOOLS_DECLARATIONS = [
    {
        "name": "create_interaction",
        "description": "Creates a new numeric feature by combining two existing columns.",
        "parameters": {
            "type": "object",
            "properties": {
                "col1": {"type": "string"},
                "col2": {"type": "string"},
                "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]}
            },
            "required": ["col1", "col2", "operation"]
        }
    },
    {
        "name": "encode_categorical",
        "description": "Encodes a categorical column into numeric values.",
        "parameters": {
            "type": "object",
            "properties": {
                "col": {"type": "string"},
                "method": {"type": "string", "enum": ["label", "onehot"]}
            },
            "required": ["col", "method"]
        }
    },
    {
        "name": "correlation_analysis",
        "description": "Analyzes correlation of features with the target column.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {"type": "string"}
            },
            "required": ["target"]
        }
    },
    {
        "name": "select_top_features",
        "description": "Keeps only the most relevant features.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "k": {"type": "integer"}
            },
            "required": ["target", "k"]
        }
    },
    {
        "name": "inspect_metadata",
        "description": "Returns shape, data types, and null counts of the dataset.",
        "parameters": {"type": "object", "properties": {}}
    }
]

SYSTEM_PROMPT = """You are 'The Architect', a Feature Engineering Agent.
Your goal: Maximize information density for the 'ArsenalWin' prediction task.

CRITICAL RULES:
1. ALWAYS call 'inspect_metadata' FIRST to see actual column names - DO NOT GUESS column names
2. Only use columns that exist in the metadata output
3. Create at least ONE interaction feature using existing columns
4. Encode categorical columns (like 'Opponent', 'Venue', 'Weather')
5. Select top k features using 'select_top_features' (k should be 8-12)
6. End with 'ENGINEERING_COMPLETE' and summarize your actions

Workflow:
1. inspect_metadata() - see what columns exist
2. create_interaction() - combine numeric columns (e.g., Possession * ShotsOnTarget)
3. encode_categorical() - encode each categorical column
4. select_top_features() - keep most predictive features
5. Report completion"""

class FeatureEngineerAgent(BaseAgent):
    def __init__(self, df, cleaner_summary):
        super().__init__(
            name="The Architect",
            role="Feature Engineer",
            system_prompt=SYSTEM_PROMPT + f"\n\nCleaner Summary: {cleaner_summary}",
            tools_declarations=ENGINEERING_TOOLS_DECLARATIONS
        )
        self.df = df
        self.actions_taken = []

    def execute_tool(self, func_name, args):
        if func_name == "create_interaction":
            self.df, msg = create_interaction(self.df, args["col1"], args["col2"], args["operation"])
            self.actions_taken.append(msg)
            return msg
        elif func_name == "encode_categorical":
            self.df, msg = encode_categorical(self.df, args["col"], args.get("method", "label"))
            self.actions_taken.append(msg)
            return msg
        elif func_name == "select_top_features":
            self.df, msg = select_top_features(self.df, args["target"], args["k"])
            self.actions_taken.append(msg)
            return msg
        elif func_name == "inspect_metadata":
            return inspect_metadata(self.df)
        return f"Unknown tool: {func_name}"
