import pandas as pd
import numpy as np

# 1. Create Live Inventory (The "Now")
inventory_data = {
    'item_id': ['OIL_02', 'MILK_03', 'RICE_01', 'SUGAR_05'],
    'item_name': ['Sunflower Oil 1L', 'Whole Milk 1L', 'Basmati Rice 5kg', 'White Sugar 1kg'],
    'current_stock': [15, 8, 45, 12]
}
pd.DataFrame(inventory_data).to_csv('inventory_live.csv', index=False)

# 2. Create Sales History (The "Past" for Holt-Winters)
# We generate 14 days of sales for each item
history_records = []
items = ['OIL_02', 'MILK_03', 'RICE_01', 'SUGAR_05']

for item in items:
    # Simulate a weekly cycle (higher sales on weekends)
    base_sales = np.array([5, 6, 5, 7, 8, 15, 12] * 2) 
    noise = np.random.randint(-2, 3, size=14)
    final_sales = base_sales + noise
    
    for day, sale in enumerate(final_sales):
        history_records.append({'item_id': item, 'day': day, 'sales': max(0, sale)})

pd.DataFrame(history_records).to_csv('sales_history.csv', index=False)
print("✅ Data generated: inventory_live.csv and sales_history.csv")