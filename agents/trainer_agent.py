import pandas as pd
import json
from agents.agent_base import BaseAgent
from tools.training_tools import execute_python_code

TRAINING_TOOLS_DECLARATIONS = [
    {
        "name": "execute_python_code",
        "description": "Executes Python code for training and evaluation. Returns logs and metrics.",
        "parameters": {
            "type": "object",
            "properties": {
                "code_string": {"type": "string", "description": "The full Python script to execute."}
            },
            "required": ["code_string"]
        }
    }
]

SYSTEM_PROMPT = """You are 'The Coder', a Model Training Agent.
Your goal: Train an XGBoost model on 'data/engineered_data.csv' to predict 'ArsenalWin'.

IMPORTANT: All required libraries (pandas, sklearn, xgboost) are already installed.

WORKFLOW:
1. Before calling execute_python_code, briefly explain what hyperparameters you're testing and why
2. Generate Python code and call execute_python_code
3. Analyze the results (Accuracy and F1 Score)
4. If results are unsatisfactory, explain what you'll change and why, then try again
5. After your final attempt, say 'TRAINING_COMPLETE' with the best metrics achieved

Your code must:
- Load 'data/engineered_data.csv'
- Split 80/20 train/test with random_state=42
- Train XGBoost model
- Print exactly: Accuracy: X.XX and F1 Score: X.XX

You have full creative control over hyperparameters. Make intelligent decisions based on results.
"""

class ModelTrainerAgent(BaseAgent):
    def __init__(self, engineered_summary):
        super().__init__(
            name="The Coder",
            role="Model Trainer",
            system_prompt=SYSTEM_PROMPT + f"\n\nEngineering Summary: {engineered_summary}",
            tools_declarations=TRAINING_TOOLS_DECLARATIONS
        )

    def execute_tool(self, func_name, args):
        if func_name == "execute_python_code":
            return execute_python_code(args["code_string"])
        return f"Unknown tool: {func_name}"
