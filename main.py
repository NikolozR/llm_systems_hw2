import pandas as pd
import os
from dotenv import load_dotenv
from agents.cleaner_agent import DataCleanerAgent
from agents.engineer_agent import FeatureEngineerAgent
from agents.trainer_agent import ModelTrainerAgent

load_dotenv()

def main():
    print("🚀 Starting Multi-Agent AutoML Pipeline...")
    

    raw_data_path = 'data/raw_data.csv'
    if not os.path.exists(raw_data_path):
        print("Creating sample data...")
        import generate_sample_data
    
    df = pd.read_csv(raw_data_path)
    
    # --- Agent 1: Data Cleaner ---
    print("\n" + "="*80)
    print("📊 PHASE 1: DATA CLEANING")
    print("="*80)
    print("Task: Audit data quality and handle missing values/outliers")
    
    cleaner = DataCleanerAgent(df)
    cleaner_report = cleaner.run("Please audit and clean the raw dataset.")
    
    clean_data_path = 'data/clean_data.csv'
    cleaner.df.to_csv(clean_data_path, index=False)
    print(f"\n✅ Saved cleaned data to: {clean_data_path}")
    print(f"\n🔄 Handoff to Feature Engineer: {len(cleaner.actions_taken)} cleaning actions performed")
    
    # --- Agent 2: Feature Engineer ---
    print("\n" + "="*80)
    print("🔧 PHASE 2: FEATURE ENGINEERING")
    print("="*80)
    print("Task: Create new features and select the most relevant ones")
    
    engineer = FeatureEngineerAgent(cleaner.df, cleaner_report)
    engineer_report = engineer.run("Please perform feature engineering and selection on the clean data.")
    
    engineered_data_path = 'data/engineered_data.csv'
    engineer.df.to_csv(engineered_data_path, index=False)
    print(f"\n✅ Saved engineered data to: {engineered_data_path}")
    print(f"\n🔄 Handoff to Model Trainer: {len(engineer.actions_taken)} engineering actions performed")
    
    # --- Agent 3: Model Trainer ---
    print("\n" + "="*80)
    print("🎯 PHASE 3: MODEL TRAINING")
    print("="*80)
    print("Task: Train XGBoost model with iterative hyperparameter optimization")
    
    trainer = ModelTrainerAgent(engineer_report)
    trainer_report = trainer.run("Please train an XGBoost model on 'data/engineered_data.csv' to predict 'ArsenalWin'.")
    
    print(f"\n✅ Model training complete!")
    
    # Generate Final Report
    generate_final_report(cleaner_report, engineer_report, trainer_report)

def generate_final_report(r1, r2, r3):
    report_content = f"""# Multi-Agent AutoML Final Report

## Agent 1: The Auditor (Data Cleaner)
{r1}

## Agent 2: The Architect (Feature Engineer)
{r2}

## Agent 3: The Coder (Model Trainer)
{r3}
"""
    with open("FINAL_REPORT.md", "w") as f:
        f.write(report_content)
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)
    print(f"📄 Final report saved to: FINAL_REPORT.md")
    print(f"📊 Data files: clean_data.csv → engineered_data.csv")
    print("="*80)

if __name__ == "__main__":
    main()
