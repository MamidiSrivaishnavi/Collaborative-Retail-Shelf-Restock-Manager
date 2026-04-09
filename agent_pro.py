import os
import json
import re
import pandas as pd
import numpy as np
import faiss
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sentence_transformers import SentenceTransformer
from tabulate import tabulate

# --- 1. CONFIGURATION ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Force JSON output for system stability
model = genai.GenerativeModel(
    'gemini-2.5-flash',
    generation_config={"response_mime_type": "application/json"}
)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- 2. THE ANALYTICS ENGINES ---
def calculate_demand_forecast(history_series):
    """Predicts demand using Holt-Winters Triple Exponential Smoothing"""
    try:
        # seasonal_periods=7 handles weekly shopping cycles
        hw_model = ExponentialSmoothing(
            history_series, trend='add', seasonal='add', seasonal_periods=7
        ).fit()
        return max(0, hw_model.forecast(1)[0])
    except:
        return np.mean(history_series) if len(history_series) > 0 else 0

def build_retail_memory(df):
    """FAISS Vector Memory for product relationships"""
    descriptions = df['item_name'].tolist()
    embeddings = embed_model.encode(descriptions)
    index = faiss.IndexFlatL2(384)
    index.add(np.array(embeddings).astype('float32'))
    return index, descriptions

# --- 3. THE CORE AGENT LOGIC ---
def run_retail_agent_logic():
    """Processes inventory and returns raw AI recommendations"""
    df = pd.read_csv('inventory_live.csv')
    history_df = pd.read_csv('sales_history.csv') 
    index, memory_list = build_retail_memory(df)
    
    results = []
    for _, row in df.iterrows():
        # 1. Forecast with 20% Safety Buffer
        item_history = history_df[history_df['item_id'] == row['item_id']]['sales'].values
        predicted_demand = calculate_demand_forecast(item_history)
        safety_stock_target = predicted_demand * 1.2
        
        # 2. Vector Search for related context
        query_vector = embed_model.encode([row['item_name']])
        D, I = index.search(query_vector.astype('float32'), k=2)
        related = [memory_list[i] for i in I[0] if memory_list[i] != row['item_name']]

        # 3. Gemini Reasoning
        prompt = f"""
        Analyze inventory for: {row['item_name']}
        Current Stock: {row['current_stock']}
        Target Stock (Forecast + Buffer): {safety_stock_target}
        Related Products: {related}
        
        Task: Return a JSON object with:
        "item_name": string,
        "status": "STABLE" or "URGENT",
        "order_quantity": number (Forecast minus current stock, min 0)
        """
        
        response = model.generate_content(prompt)
        results.append(response.text)
    return results

# --- 4. THE MANAGER'S DASHBOARD (UI) ---
def launch_dashboard():
    print("\n" + "="*65)
    print("🏪 RETAIL AI SYSTEM: LIVE MANAGER DASHBOARD")
    print("="*65)
    
    raw_data = run_retail_agent_logic()
    table_data = []

    for item_raw in raw_data:
        try:
            # Clean and parse JSON
            clean_json = re.search(r'\{.*\}', item_raw, re.DOTALL).group()
            data = json.loads(clean_json)
            
            # Logic-based formatting
            status_val = data['status'].upper()
            order_qty = round(float(data['order_quantity']))
            
            # Set icons based on urgency
            if order_qty > 0 or "URGENT" in status_val:
                display_status = f"⚠️  RESTOCK NEEDED"
                display_qty = f"{order_qty} units"
            else:
                display_status = f"✅  STABLE"
                display_qty = "0 (Stock OK)"

            table_data.append([data['item_name'], display_status, display_qty])
        except Exception as e:
            continue

    headers = ["Product Name", "Inventory Health", "Recommended Action"]
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
    
    now = datetime.now().strftime("%d-%m-%Y %H:%M")
    print(f"\n📅 Report Generated: {now} IST")
    print("💡 Note: Order quantities include 20% safety stock for weekend rushes.")
    print("="*65 + "\n")

if __name__ == "__main__":
    launch_dashboard()