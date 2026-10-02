import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from claude_code.model import (
    build_customer_features,
    train_segmentation,
    assign_segments,
    add_fallback_campaign,
)


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Personalized Marketing Tool",
    page_icon="📊",
    layout="wide"
)


# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 40px;
        font-weight: 800;
        color: white;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 17px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    .metric-card {
        background: white;
        padding: 22px;
        border-radius: 18px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
        border: 1px solid #eef2f7;
        text-align: center;
    }

    .metric-label {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #111827;
    }

    .info-box {
        background-color: #f3f4f6;
        padding: 18px;
        border-radius: 15px;
        border-left: 5px solid #2563eb;
        margin-top: 15px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- HELPER FUNCTIONS ----------------
def clean_product_name(product):
    if pd.isna(product):
        return "your favorite products"

    product = str(product).strip().title()

    if len(product) > 45:
        product = product[:45] + "..."

    return product


def run_customer_segmentation(df):
    """Uses the shared logic in model.py (same code the Lambda will run)."""
    required_columns = [
        "CustomerID", "InvoiceNo", "InvoiceDate",
        "Description", "Quantity", "UnitPrice"
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Cleaning, RFM and top product
    customer_features = build_customer_features(df)

    if len(customer_features) < 4:
        raise ValueError("At least 4 customers are needed for K-Means clustering.")

    # Tidy product names for display and messages
    customer_features["Top_Product"] = customer_features["Top_Product"].map(clean_product_name)

    # Train the clustering model and assign named segments
    bundle = train_segmentation(customer_features)
    customer_features = assign_segments(customer_features, bundle)

    # Template-based campaign messages (Bedrock replaces these later)
    customer_features = add_fallback_campaign(customer_features)

    return customer_features, df


# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">AI-Driven Personalized Marketing Dashboard</div>',
    unsafe_allow_html=True
)


# ---------------- SIDEBAR ----------------
st.sidebar.title("📁 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload customer transaction CSV",
    type=["csv"]
)

run_analysis = st.sidebar.button("Run Customer Analysis")


# ---------------- FILE UPLOAD ----------------
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding="latin1")

        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        if run_analysis:
            with st.spinner("Analyzing customer behavior..."):
                customer_features, cleaned_df = run_customer_segmentation(df)

            st.session_state["customer_features"] = customer_features
            st.session_state["cleaned_df"] = cleaned_df


    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Please upload a CSV file from the sidebar to begin analysis.")


# ---------------- DASHBOARD ----------------
if "customer_features" in st.session_state:

    customer_features = st.session_state["customer_features"]
    cleaned_df = st.session_state["cleaned_df"]

    total_customers = customer_features["CustomerID"].nunique()
    total_segments = customer_features["Segment"].nunique()
    total_revenue = customer_features["Monetary"].sum()
    avg_recency = customer_features["Recency"].mean()

    st.markdown("## 📌 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Customers</div>
            <div class="metric-value">{total_customers}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Customer Segments</div>
            <div class="metric-value">{total_segments}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Revenue</div>
            <div class="metric-value">{total_revenue:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Recency</div>
            <div class="metric-value">{avg_recency:.1f} days</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "👥 Segment Analysis",
        "💬 Campaign Messages",
        "📄 Full Customer Data"
    ])

    # ---------------- TAB 1 ----------------
    with tab1:
        st.subheader("Customer Segment Distribution")

        segment_counts = customer_features["Segment"].value_counts().reset_index()
        segment_counts.columns = ["Segment", "Customer_Count"]

        fig1 = px.pie(
            segment_counts,
            names="Segment",
            values="Customer_Count",
            hole=0.45,
            title="Customer Distribution by Segment"
        )

        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Segment Count Table")
        st.dataframe(segment_counts, use_container_width=True)

    # ---------------- TAB 2 ----------------
    with tab2:
        st.subheader("RFM Segment Summary")

        segment_summary = customer_features.groupby("Segment").agg({
            "CustomerID": "count",
            "Recency": "mean",
            "Frequency": "mean",
            "Monetary": "mean"
        }).round(2)

        segment_summary = segment_summary.rename(columns={
            "CustomerID": "Customer Count",
            "Recency": "Avg Recency",
            "Frequency": "Avg Frequency",
            "Monetary": "Avg Monetary"
        })

        st.dataframe(segment_summary, use_container_width=True)

        summary_reset = segment_summary.reset_index()

        fig2 = px.bar(
            summary_reset,
            x="Segment",
            y="Avg Monetary",
            text="Avg Monetary",
            title="Average Monetary Value by Segment"
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Customer Behaviour Map")

        fig3 = px.scatter(
            customer_features,
            x="Recency",
            y="Monetary",
            size="Frequency",
            color="Segment",
            hover_data=["CustomerID", "Top_Product"],
            title="Recency vs Monetary Value"
        )

        st.plotly_chart(fig3, use_container_width=True)

    # ---------------- TAB 3 ----------------
    with tab3:
        st.subheader("Personalized Campaign Output")

        final_output = customer_features[
            ["CustomerID", "Segment", "Top_Product", "Campaign_Goal", "Campaign_Message"]
        ]

        st.dataframe(final_output, use_container_width=True)

        csv = final_output.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Campaign CSV",
            data=csv,
            file_name="personalized_marketing_campaigns.csv",
            mime="text/csv"
        )

    # ---------------- TAB 4 ----------------
    with tab4:
        st.subheader("Full Customer Feature Table")

        st.dataframe(customer_features, use_container_width=True)

        full_csv = customer_features.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Full Customer Data",
            data=full_csv,
            file_name="customer_segments_full_data.csv",
            mime="text/csv"
        )