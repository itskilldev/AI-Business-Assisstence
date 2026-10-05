# ============================================================
# AI Business Operations Assistant - Streamlit Dashboard
# File: app.py
# Run:  streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import os
from dotenv import load_dotenv
from google import genai

st.set_page_config(
    page_title="AI Business Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD ENV & CONFIGURE GEMINI
# ============================================================
load_dotenv()

GEMINI_API_KEY = "AQ.Ab8RN6K5VaOaa9Br3TO7mG_meCH-cV012yxGVW2oN42vbbgHvg"

if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        gemini_ready = True
    except Exception as e:
        client = None
        gemini_ready = False
        print(f"Gemini init error: {e}")
else:
    client = None
    gemini_ready = False


# ============================================================
# SECTION 1: DATA PROCESSING
# ============================================================

def normalize_column_name(name):
    return str(name).lower().strip().replace("_", " ").replace("-", " ")


def detect_column_types(df):
    numerical, categorical, date_columns, text_columns = [], [], [], []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            numerical.append(column)
            continue
        converted = pd.to_datetime(series, errors="coerce", dayfirst=True)
        non_empty = series.notna().sum()
        if non_empty > 0:
            date_ratio = converted.notna().sum() / non_empty
            if date_ratio >= 0.8:
                date_columns.append(column)
                continue
        unique_ratio = series.nunique() / max(len(series), 1)
        if unique_ratio < 0.5:
            categorical.append(column)
        else:
            text_columns.append(column)
    return {
        "numerical": numerical,
        "categorical": categorical,
        "date_columns": date_columns,
        "text_columns": text_columns
    }


COLUMN_KEYWORDS = {
    "date": ["order date", "transaction date", "invoice date", "sale date",
             "purchase date", "date"],
    "order_id": ["order id", "order number", "invoice no", "invoice number",
                 "transaction id"],
    "customer_id": ["customer id", "client id", "buyer id", "customer code"],
    "customer": ["customer name", "client name", "buyer name", "customer",
                 "client", "buyer"],
    "product_id": ["product id", "item id", "sku", "product code"],
    "product": ["product name", "item name", "service name", "product",
                "item", "service"],
    "category": ["product category", "category", "department", "type"],
    "sales": ["total sales", "net sales", "revenue", "sales", "amount",
              "total amount"],
    "cost": ["cost", "product cost", "unit cost", "cost price", "purchase cost"],
    "profit": ["net profit", "gross profit", "profit"],
    "expenses": ["expense", "expenses", "operating expense",
                 "operating expenses", "business expenses"],
    "quantity": ["quantity", "qty", "units sold", "units"],
    "location": ["location", "branch", "store", "city", "area"],
    "region": ["region", "zone", "territory"]
}


def map_columns(df):
    mapping = {}
    normalized_columns = {col: normalize_column_name(col) for col in df.columns}
    for standard_name, keywords in COLUMN_KEYWORDS.items():
        best_match = None
        for original_column, normalized in normalized_columns.items():
            if normalized in keywords:
                best_match = original_column
                break
        if best_match is None:
            for original_column, normalized in normalized_columns.items():
                for keyword in keywords:
                    if keyword in normalized:
                        best_match = original_column
                        break
                if best_match:
                    break
        mapping[standard_name] = best_match
    return mapping


def standardize_dataset(df, mapping):
    standardized = pd.DataFrame()
    for standard_name, original_column in mapping.items():
        if original_column is not None:
            standardized[standard_name] = df[original_column]
        else:
            standardized[standard_name] = np.nan
    return standardized


def clean_and_prepare(df):
    column_types = detect_column_types(df)
    for col in column_types["date_columns"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    mapping = map_columns(df)
    standard_df = standardize_dataset(df, mapping)
    standard_df = standard_df.dropna(axis=1, how="all")

    if "sales" in standard_df.columns:
        standard_df["sales"] = (
            standard_df["sales"].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.strip()
        )
        standard_df["sales"] = pd.to_numeric(standard_df["sales"], errors="coerce")

    if "date" in standard_df.columns:
        standard_df["date"] = pd.to_datetime(standard_df["date"], errors="coerce")

    return standard_df


# ============================================================
# SECTION 2: BUSINESS METRICS
# ============================================================

def generate_business_metrics(df):
    metrics = {}
    if "sales" in df.columns:
        metrics["total_sales"] = float(df["sales"].sum())
    if "order_id" in df.columns:
        metrics["total_orders"] = int(df["order_id"].nunique())
    if "customer_id" in df.columns:
        metrics["total_customers"] = int(df["customer_id"].nunique())
    if "sales" in df.columns and "order_id" in df.columns:
        orders = df["order_id"].nunique()
        if orders > 0:
            metrics["average_order_value"] = float(df["sales"].sum()) / orders
    if "category" in df.columns and "sales" in df.columns:
        cat = df.groupby("category")["sales"].sum().sort_values(ascending=False)
        if len(cat) > 0:
            metrics["top_category"] = {
                "name": str(cat.index[0]),
                "sales": float(cat.iloc[0])
            }
    return metrics


def monthly_business_analysis(df):
    if "date" not in df.columns or "sales" not in df.columns:
        return None
    monthly = (
        df.groupby(df["date"].dt.to_period("M"))["sales"].sum().sort_index()
    )
    if len(monthly) == 0:
        return None
    return {
        "average_monthly_sales": float(monthly.mean()),
        "highest_month": {
            "month": str(monthly.idxmax()),
            "sales": float(monthly.max())
        },
        "lowest_month": {
            "month": str(monthly.idxmin()),
            "sales": float(monthly.min())
        },
        "sales_volatility": float(monthly.std()) if len(monthly) > 1 else 0.0
    }


def detect_sales_anomalies(df):
    if "date" not in df.columns or "sales" not in df.columns:
        return []
    monthly = (
        df.groupby(df["date"].dt.to_period("M"))["sales"].sum().sort_index()
    )
    if len(monthly) < 4:
        return []
    mean, std = monthly.mean(), monthly.std()
    if std == 0 or pd.isna(std):
        return []
    anomalies = []
    for month, sales in monthly.items():
        z = (sales - mean) / std
        if abs(z) >= 2:
            anomalies.append({
                "month": str(month),
                "sales": float(sales),
                "z_score": float(z),
                "type": "unusually_high" if z > 0 else "unusually_low"
            })
    return anomalies


def generate_business_alerts(df):
    alerts = []
    for anomaly in detect_sales_anomalies(df):
        if anomaly["type"] == "unusually_low":
            alerts.append({
                "severity": "warning",
                "message": f"Sales in {anomaly['month']} were unusually low "
                           f"at {anomaly['sales']:,.2f}."
            })
        else:
            alerts.append({
                "severity": "opportunity",
                "message": f"Sales in {anomaly['month']} were unusually high "
                           f"at {anomaly['sales']:,.2f}."
            })
    return alerts


# ============================================================
# SECTION 3: AI CONTEXT & Q&A
# ============================================================

def create_ai_business_context(df):
    context = {}
    if "sales" in df.columns:
        context["total_sales"] = float(df["sales"].sum())
    if "order_id" in df.columns:
        context["total_orders"] = int(df["order_id"].nunique())
    if "customer_id" in df.columns:
        context["total_customers"] = int(df["customer_id"].nunique())
    if "category" in df.columns and "sales" in df.columns:
        cat = df.groupby("category")["sales"].sum().sort_values(ascending=False)
        context["category_performance"] = {str(k): float(v) for k, v in cat.items()}
    if "product" in df.columns and "sales" in df.columns:
        prod = df.groupby("product")["sales"].sum().sort_values(ascending=False)
        context["top_products"] = {str(k): float(v) for k, v in prod.head(10).items()}
    if "location" in df.columns and "sales" in df.columns:
        loc = df.groupby("location")["sales"].sum().sort_values(ascending=False)
        context["top_locations"] = {str(k): float(v) for k, v in loc.head(10).items()}
    if "date" in df.columns and "sales" in df.columns:
        monthly = df.groupby(df["date"].dt.to_period("M"))["sales"].sum().sort_index()
        context["monthly_sales"] = {str(k): float(v) for k, v in monthly.items()}
    return context


def build_ai_prompt(question, context):
    return f"""You are an AI Business Analyst.

Answer the user's business question using ONLY the verified business data below.
Do not invent numbers.
If the data does not contain enough information, clearly say so.

BUSINESS DATA:
{json.dumps(context, indent=2)}

USER QUESTION:
{question}

Provide:
1. Direct answer
2. Important evidence
3. Business implication
4. Recommended action
"""


def classify_question(question):
    q = question.lower()
    if "total sales" in q or "revenue" in q:
        return "sales"
    if "orders" in q:
        return "orders"
    if "customers" in q:
        return "customers"
    if "best category" in q or "top category" in q:
        return "category"
    if "best product" in q or "top product" in q:
        return "product"
    if "location" in q or "city" in q:
        return "location"
    if "trend" in q or "growth" in q:
        return "trend"
    return "ai"


def business_question(question, context):
    """Rule-based fallback for simple questions."""
    q = question.lower()
    if ("total sales" in q or "revenue" in q) and "total_sales" in context:
        return f"Total sales are **{context['total_sales']:,.2f}**."
    if "orders" in q and "total_orders" in context:
        return f"There are **{context['total_orders']:,}** unique orders."
    if "customers" in q and "total_customers" in context:
        return f"There are **{context['total_customers']:,}** unique customers."
    if ("best category" in q or "top category" in q) and "category_performance" in context:
        cats = context["category_performance"]
        best = max(cats, key=cats.get)
        return f"The best-performing category is **{best}**, with sales of {cats[best]:,.2f}."
    if ("best product" in q or "top product" in q) and "top_products" in context:
        prods = context["top_products"]
        best = max(prods, key=prods.get)
        return f"The top product is **{best}**, with sales of {prods[best]:,.2f}."
    if ("location" in q or "city" in q) and "top_locations" in context:
        locs = context["top_locations"]
        best = max(locs, key=locs.get)
        return f"The top location is **{best}**, with sales of {locs[best]:,.2f}."
    return "I don't have enough information. Try asking about sales, orders, customers, categories, products, or locations."


def ask_gemini(question, context):
    """Use Gemini AI with automatic model fallback."""
    if not gemini_ready or client is None:
        fallback = business_question(question, context)
        return fallback, "rule-based (no API key)"

    prompt = f"""You are an expert AI Business Analyst.

Answer the user's question using ONLY the verified business data below.
Do NOT invent numbers. Be specific, practical, and concise.

═══════════════════════════════════════
VERIFIED BUSINESS DATA
═══════════════════════════════════════
{json.dumps(context, indent=2)}
═══════════════════════════════════════

USER QUESTION:
{question}

Instructions:
- Give a DIRECT answer first
- Support with specific numbers from the data
- Provide business implications
- Suggest 1-2 recommended actions
- If data is insufficient, say so honestly
- Keep under 200 words
- Use markdown formatting (bold, bullet points)

Response:"""

    # Try multiple models - fallback if one is unavailable
    models_to_try = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite",
        "gemini-flash-latest",
    ]

    last_error = None
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            return response.text, f"Gemini AI ({model_name})"
        except Exception as e:
            last_error = str(e)
            continue

    # All models failed - use rule-based fallback
    fallback = business_question(question, context)
    return (
        f"{fallback}\n\n_(All Gemini models unavailable. Last error: {last_error})_",
        "rule-based (fallback)"
    )
# ============================================================
# SECTION 4: STREAMLIT UI
# ============================================================

st.title("📊 AI Business Operations Assistant")
st.markdown("Upload a business dataset and get instant insights, KPIs, and AI-powered answers.")

with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader(
        "Upload dataset",
        type=["csv", "xlsx", "xls"],
        help="CSV or Excel file with business data"
    )

    if gemini_ready:
        st.success("✨ Gemini AI: Active")
    else:
        st.warning("⚠️ Gemini AI: Not configured")

if uploaded_file is None:
    st.info("👈 Please upload a CSV or Excel file from the sidebar to begin.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 📈 Auto KPIs")
        st.write("Total sales, orders, customers, AOV — auto-calculated.")
    with c2:
        st.markdown("#### 🧠 Smart Mapping")
        st.write("Automatically recognizes columns across datasets.")
    with c3:
        st.markdown("#### 🤖 AI Q&A")
        st.write("Ask business questions in plain English.")
    st.stop()

try:
    if uploaded_file.name.endswith(".csv"):
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = pd.read_excel(uploaded_file)
    st.sidebar.success(f"✅ Loaded: {raw_df.shape[0]:,} rows × {raw_df.shape[1]} cols")
except Exception as e:
    st.error(f"❌ Failed to load file: {e}")
    st.stop()

with st.spinner("Processing data..."):
    df = clean_and_prepare(raw_df.copy())
    metrics = generate_business_metrics(df)
    ai_context = create_ai_business_context(df)
    monthly_info = monthly_business_analysis(df)
    alerts = generate_business_alerts(df)

st.subheader("📈 Key Business Metrics")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Sales", f"${metrics.get('total_sales', 0):,.2f}")
k2.metric("Total Orders", f"{metrics.get('total_orders', 0):,}")
k3.metric("Total Customers", f"{metrics.get('total_customers', 0):,}")
k4.metric("Avg Order Value", f"${metrics.get('average_order_value', 0):,.2f}")

if alerts:
    st.markdown("### 🚨 Business Alerts")
    for alert in alerts:
        if alert["severity"] == "warning":
            st.warning(alert["message"])
        else:
            st.success(alert["message"])

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🏆 Top Performers", "📉 Trends", "🔎 Data Explorer", "🤖 AI Assistant"
])

with tab1:
    st.subheader("Category Performance")
    if "category" in df.columns and "sales" in df.columns:
        cat_df = df.groupby("category")["sales"].sum().reset_index().sort_values("sales", ascending=False)
        fig = px.bar(cat_df, x="category", y="sales", color="category",
                     title="Sales by Category", text_auto=".2s")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No 'category' column detected.")

    if "region" in df.columns and "sales" in df.columns:
        reg_df = df.groupby("region")["sales"].sum().reset_index().sort_values("sales", ascending=False)
        fig = px.pie(reg_df, values="sales", names="region",
                     title="Sales by Region")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    ca, cb = st.columns(2)
    with ca:
        if "product" in df.columns and "sales" in df.columns:
            prod_df = df.groupby("product")["sales"].sum().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(prod_df, x="sales", y="product", orientation="h", title="Top 10 Products")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
    with cb:
        if "location" in df.columns and "sales" in df.columns:
            loc_df = df.groupby("location")["sales"].sum().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(loc_df, x="sales", y="location", orientation="h", title="Top 10 Locations")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    if "customer_id" in df.columns and "sales" in df.columns:
        cust_df = df.groupby("customer_id")["sales"].sum().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(cust_df, x="customer_id", y="sales", title="Top 10 Customers", text_auto=".2s")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    if "date" in df.columns and "sales" in df.columns:
        monthly = df.groupby(df["date"].dt.to_period("M"))["sales"].sum().reset_index()
        monthly["date"] = monthly["date"].astype(str)
        fig = px.line(monthly, x="date", y="sales", markers=True, title="Monthly Sales Trend")
        st.plotly_chart(fig, use_container_width=True)
        if monthly_info:
            m1, m2, m3 = st.columns(3)
            m1.metric("Avg Monthly", f"${monthly_info['average_monthly_sales']:,.2f}")
            m2.metric("Best Month", monthly_info['highest_month']['month'],
                      f"${monthly_info['highest_month']['sales']:,.2f}")
            m3.metric("Weakest Month", monthly_info['lowest_month']['month'],
                      f"${monthly_info['lowest_month']['sales']:,.2f}")
    else:
        st.info("No 'date' column detected.")

with tab4:
    st.subheader("Processed Data Preview")
    st.write(f"Shape: **{df.shape[0]:,} rows × {df.shape[1]} cols**")
    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("Column Mapping")
    mapping = map_columns(raw_df)
    mapping_df = pd.DataFrame([
        {"Standard": k, "Original": v if v else "❌ Not found"}
        for k, v in mapping.items()
    ])
    st.dataframe(mapping_df, use_container_width=True)

with tab5:
    st.subheader("🤖 AI Business Assistant")

    if gemini_ready:
        st.success("✨ **Powered by Google Gemini** — Ask anything about your business data!")
    else:
        st.warning("⚠️ Gemini not configured. Add `GEMINI_API_KEY` to `.env` file.")

    st.markdown("""
    **Try asking:**
    - What are my total sales?
    - Which category is performing best and why?
    - How can I improve my sales?
    - Compare Technology vs Furniture performance
    - What is my biggest business opportunity?
    - Are there any anomalies in my sales?
    """)

    user_q = st.text_input("Your question:", placeholder="e.g. Which category should I focus on?")

if user_q:
    with st.spinner("🤔 AI is thinking..."):
        answer, source = ask_gemini(user_q, ai_context)
    
    # Source badge
    if "Gemini" in source:
        st.success(f"✨ **Source:** {source}")
    else:
        st.warning(f"⚠️ **Source:** {source}")
    
    st.markdown("---")
    
    # Answer inside a nice container
    with st.container(border=True):
        st.markdown(answer)
    
    st.markdown("---")

    with st.expander("🔍 View AI Business Context (raw data sent to AI)"):
        st.json(ai_context)

    with st.expander("📝 View Structured Prompt"):
        sample_q = user_q if user_q else "Which category is performing best?"
        st.code(build_ai_prompt(sample_q, ai_context), language="text")

st.markdown("---")
st.caption("AI Business Operations Assistant · Streamlit + Pandas + Plotly + Gemini AI")