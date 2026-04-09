import pandas as pd
import numpy as np

df = pd.read_csv('retail_store_inventory.csv')
df['Date'] = pd.to_datetime(df['Date'])
latest_date = df['Date'].max()
current_df = df[df['Date'] == latest_date].copy()

urgent_clothing = []
for _, row in current_df.iterrows():
    if row['Category'] != 'Clothing':
        continue
    pid, sid = row['Product ID'], row['Store ID']
    history = df[(df['Product ID'] == pid) & (df['Store ID'] == sid)]['Units Sold'].values
    avg_sales = np.mean(history[-30:]) if len(history) > 0 else 1
    current_stock = row['Inventory Level']
    days_left = current_stock / avg_sales
    if days_left < 2.5:
        urgent_clothing.append({
            'Store': sid,
            'Product': pid,
            'Season': row['Seasonality']
        })

df_urgent = pd.DataFrame(urgent_clothing)
print("URGENT Clothing Items:")
for _, r in df_urgent.iterrows():
    print(f"{r['Store']} - {r['Product']}: {r['Season']}")
print(f"\nSeason Counts:")
counts = df_urgent['Season'].value_counts().sort_values(ascending=False)
for season, count in counts.items():
    print(f"{season}: {count}")
print(f"\nWINNER: {counts.idxmax()} with {counts.max()} items")
