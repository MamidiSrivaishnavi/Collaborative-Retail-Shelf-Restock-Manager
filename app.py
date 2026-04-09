import streamlit as st
import pandas as pd
import numpy as np
import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

# --- 1. CONFIGURATION & SETUP ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

st.set_page_config(page_title="Retail AI: Intelligence Dashboard", layout="wide")

# UPDATED CSS: Adjusted font sizes to ensure Cost (INR) is fully visible
st.markdown("""
    <style>
    /* Card Container */
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        padding: 15px !important;
        border-radius: 12px !important;
        border-left: 6px solid #1e3a8a !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
        min-height: 120px;
    }
    /* Label Text (Top) - Made slightly smaller for better fit */
    [data-testid="stMetricLabel"] p {
        color: #475569 !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        white-space: nowrap !important; /* Prevents title from wrapping */
    }
    /* Value Text (Large Number) - Adjusted size to fit large currency values */
    [data-testid="stMetricValue"] div {
        color: #0f172a !important;
        font-weight: 900 !important;
        font-size: 1.8rem !important; 
        line-height: 1.2 !important;
    }
    h1, h2, h3 { color: #1e3a8a; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedder()

# --- 2. CORE DATA LOGIC ---
@st.cache_data
def get_inventory_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    latest_date = df['Date'].max()
    current_df = df[df['Date'] == latest_date].copy()
    
    results = []
    for _, row in current_df.iterrows():
        pid, sid = row['Product ID'], row['Store ID']
        history = df[(df['Product ID'] == pid) & (df['Store ID'] == sid)]['Units Sold'].values
        avg_sales = np.mean(history[-30:]) if len(history) > 0 else 1
        current_stock = row['Inventory Level']
        days_left = current_stock / avg_sales
        status = "URGENT" if days_left < 2.5 else "STABLE"
        target_qty = avg_sales * 7
        order_qty = max(0, round(target_qty - current_stock))

        results.append({
            "Store": sid, "Product": pid, "Category": row['Category'],
            "Stock": int(current_stock), "Daily Sales": round(avg_sales, 1),
            "Status": status, "Order Qty": order_qty, "Cost (INR)": round(order_qty * row['Price'], 2)
        })
    return pd.DataFrame(results)

# --- 3. VECTOR SEARCH ENGINE (FAISS) ---
def build_search_index(data_df):
    text_data = data_df.apply(lambda x: 
        f"Store {x['Store']} has {x['Product']} ({x['Category']}). Status: {x['Status']}, Stock: {x['Stock']}.", 
        axis=1).tolist()
    embeddings = embedder.encode(text_data)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index, text_data

# --- 4. MAIN USER INTERFACE ---
st.title("🏪 Retail AI Intelligence Dashboard")
st.markdown("#### Real-time Inventory Analytics & AI Forecasting")

uploaded_file = st.sidebar.file_uploader("Upload Inventory CSV", type="csv")

if uploaded_file:
    data = get_inventory_data(uploaded_file)
    tab1, tab2 = st.tabs(["📊 Performance Dashboard", "🤖 AI Semantic Search"])
    
    with tab1:
        # Metrics section
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Products", len(data))
        m2.metric("Urgent Restocks", len(data[data['Status'] == "URGENT"]))
        m3.metric("Units to Order", int(data['Order Qty'].sum()))
        m4.metric("Cost (INR)", f"₹{data['Cost (INR)'].sum():,.0f}")

        st.subheader("📊 Restock Investment by Store")
        cost_by_store = data.groupby('Store')['Cost (INR)'].sum().reset_index()
        st.bar_chart(data=cost_by_store, x='Store', y='Cost (INR)')

        def color_status(val):
            color = '#ff4b4b' if val == 'URGENT' else '#00cc66'
            return f'background-color: {color}; color: white; font-weight: bold'

        st.subheader("🚨 Urgent Priority List")
        st.dataframe(data[data['Status'] == "URGENT"].style.applymap(color_status, subset=['Status']), use_container_width=True)
        
        st.subheader("📋 Full Inventory Audit")
        st.dataframe(data.style.applymap(color_status, subset=['Status']), use_container_width=True, height=400)

    with tab2:
        st.subheader("🔍 Ask the Retail AI")
        query = st.text_input("Example: 'Show me low stock items in Store S001'")
        
        if st.button("Generate AI Answer"):
            if query:
                with st.spinner("Searching vectors..."):
                    index, text_descriptions = build_search_index(data)
                    query_vec = embedder.encode([query])
                    dist, indices = index.search(np.array(query_vec).astype('float32'), k=3)
                    
                    context = "\n".join([text_descriptions[i] for i in indices[0]])
                    top_row_idx = indices[0][0]
                    top_row = data.iloc[top_row_idx]

                    try:
                        model = genai.GenerativeModel("gemini-1.5-flash")
                        response = model.generate_content(f"You are a retail manager. Data:\n{context}\n\nUser: {query}")
                        st.write("### AI Insights")
                        st.info(response.text)
                    except Exception:
                        st.write("### AI Insights")
                        st.warning("⚠️ Running in Offline Intelligence Mode")
                        st.info(f"""
                        **Analysis:** The most relevant item is **{top_row['Product']}** in **Store {top_row['Store']}**. 
                        Current status is **{top_row['Status']}**. Recommended order: **{top_row['Order Qty']}** units (Cost: **₹{top_row['Cost (INR)']:,.2f}**).
                        """)
            else:
                st.warning("Please enter a question first.")
else:
    st.info("👋 Please upload your .csv file to begin.")