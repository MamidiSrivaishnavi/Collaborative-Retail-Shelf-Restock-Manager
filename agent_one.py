import pandas as pd
from google.cloud import aiplatform
import os
import json

# 1. Setup the Identity (The Service Account)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "secrets.json" 

# 2. Define the Agent's Brain (Gemini)
def get_gemini_analysis(inventory_summary):
    # Initialize Vertex AI with your project ID
    aiplatform.init(project="retail-ai-system", location="us-central1")
    
    from vertexai.generative_models import GenerativeModel
    model = GenerativeModel("gemini-2.5-flash")

    prompt = f"""
    You are the Demand Forecasting Agent (Agent 1). 
    Analyze this inventory data and predict which items will hit 0 stock soon.
    
    DATA:
    {inventory_summary}
    
    OUTPUT REQUIREMENTS:
    - Identify items where (current_stock / daily_avg_sales) is less than 3 days.
    - Provide a JSON list of objects with: item_id, item_name, urgency_score (1-10), and suggested_restock_qty.
    - If no items are critical, return an empty list [].
    """
    
    response = model.generate_content(prompt)
    return response.text

# 3. The Main Execution Logic (No Hard-Coding)
def run_agent():
    print("🚀 Agent 1: Reading live inventory data...")
    df = pd.read_csv("data/inventory_live.csv")
    
    # Convert the table to text so Gemini can read it
    inventory_text = df.to_string(index=False)
    
    print("🧠 Agent 1: Analyzing with Gemini...")
    analysis = get_gemini_analysis(inventory_text)
    
    print("\n📢 AGENT 1 OUTPUT (RESTOCK REQUESTS):")
    print(analysis)

if __name__ == "__main__":
    run_agent()