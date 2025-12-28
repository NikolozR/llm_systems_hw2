# Multi-Agent AutoML Team

An autonomous AI Data Science Team consisting of three specialized agents that collaborate to process datasets, make semantic decisions, and train machine learning models.

## Overview

This system implements a sequential pipeline where three LLM agents work together:
1. **The Auditor (Data Cleaner)**: Audits data quality and handles missing values/outliers
2. **The Architect (Feature Engineer)**: Creates new features and selects the most relevant ones
3. **The Coder (Model Trainer)**: Trains XGBoost models with iterative hyperparameter optimization

## Features

- 🤖 **LLM-Driven Decisions**: Agents make semantic decisions about data cleaning, feature engineering, and model tuning
- 🔄 **Feedback Loop**: Model trainer iteratively optimizes hyperparameters based on performance
- 📊 **Conversational Logging**: Clear, readable logs showing agent reasoning and decision-making
- 🔧 **Portable**: Auto-detects virtual environments (works on Windows, Mac, Linux)

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd llm_systems_hw2
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your API key:
Create a `.env` file in the project root:
```
LLM_HW_API_KEY=your_google_genai_api_key_here
```

## Usage

Run the complete pipeline:
```bash
python main.py
```

Generate sample data (optional):
```bash
python generate_sample_data.py
```

## Project Structure

```
llm_systems_hw2/
├── agents/
│   ├── agent_base.py          # Base agent class with tool-calling logic
│   ├── cleaner_agent.py       # Data cleaning agent
│   ├── engineer_agent.py      # Feature engineering agent
│   └── trainer_agent.py       # Model training agent
├── tools/
│   ├── cleaning_tools.py      # Data cleaning utilities
│   ├── engineering_tools.py   # Feature engineering utilities
│   └── training_tools.py      # Model training utilities
├── data/
│   ├── raw_data.csv          # Input dataset
│   ├── clean_data.csv        # Cleaned dataset
│   └── engineered_data.csv   # Engineered features
├── main.py                    # Pipeline orchestration
├── generate_sample_data.py   # Sample data generator
├── FINAL_REPORT.md           # Generated report
└── requirements.txt          # Python dependencies
```

## How It Works

### Phase 1: Data Cleaning
The Auditor inspects the dataset and:
- Identifies missing values and data quality issues
- Imputes missing values using appropriate strategies (mean, median, mode)
- Drops unusable columns
- Outputs: `clean_data.csv` and a summary report

### Phase 2: Feature Engineering
The Architect receives the clean data and:
- Creates interaction features (e.g., Possession × ShotsOnTarget)
- Encodes categorical variables (one-hot or label encoding)
- Performs feature selection to keep the most predictive features
- Outputs: `engineered_data.csv` and a strategy report

### Phase 3: Model Training
The Coder trains models with:
- XGBoost classifier with custom hyperparameters
- Iterative optimization based on Accuracy and F1 Score
- Automatic hyperparameter tuning (max 3-4 attempts)
- Outputs: Final metrics and `FINAL_REPORT.md`

## Example Output

```
🚀 Starting Multi-Agent AutoML Pipeline...

================================================================================
📊 PHASE 1: DATA CLEANING
================================================================================
🤖 The Auditor is analyzing the task...

💡 The Auditor: I will use 'inspect_metadata' tool
💡 The Auditor: I will use 'impute_missing' tool
   Parameters: col='Possession', strategy='mean'

📋 The Auditor's Summary:
────────────────────────────────────────────────────────────────────────────────
CLEANING_COMPLETE
I imputed missing values in 'Possession' using the mean...
────────────────────────────────────────────────────────────────────────────────

✅ Saved cleaned data to: data/clean_data.csv
🔄 Handoff to Feature Engineer: 3 cleaning actions performed
```