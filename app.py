import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np
import requests
import io
from docx import Document
from PyPDF2 import PdfReader
from num2words import num2words
import datetime as dt
import os
from dotenv import load_dotenv
import glob
import time
import json
from urllib.parse import urljoin
import fitz 
import pytesseract
from PIL import Image, ImageOps
from langchain_groq import ChatGroq
from typing import Optional, List, Tuple, Dict, Any
import logging
from io import BytesIO
import re
from pydantic import BaseModel, Field
import pdfplumber
import camelot
from pdf2image import convert_from_bytes
import platform
import tempfile





# Try optional AI deps (app continues even if missing)
try:
    from langchain_core.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain_huggingface import HuggingFaceEndpoint
    HAS_LANGCHAIN = True
except Exception:
    HAS_LANGCHAIN = False

BACKEND_BASE_URL = os.getenv("BANK_AUCTION_INSIGHTS_API_URL", "http://localhost:8000")
AUCTION_INSIGHTS_ENDPOINT = "/auction-insights"


    





# Load environment variables
load_dotenv()
HF_API_KEY = (os.getenv("HF_API_KEY") or "").strip()

st.set_page_config(
    page_title="Auction Portal India",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: bold;
        }
        .metric-tile {
            background: linear-gradient(135deg, #ff6b6b 0%, #ff8e53 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin: 0.5rem 0;
        }
        .analytics-header {
            font-size: 1.8rem;
            color: #2e7d32;
            margin-bottom: 1rem;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🏛️ Auction Portal")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    ["🏠 Dashboard", "🔍 Search Analytics", "📊 Basic Analytics","📈 KPI Analytics", "🤖 AI Analysis"],
    index=0
)



#####################################################################################################################################################################
########################################################################################################################################################################
# Load data
@st.cache_data
def load_auction_data():
    csv_path = r"C:\Users\Amit Sharma\ai-platform\frontend\auction_exports\combined_auctions_20250819_154419.csv"
    try:
        # Get list of CSV files
        csv_files = glob.glob("auction_exports/combined_auctions_*.csv")
        
        if not csv_files:
            st.error("❌ No CSV files found in auction_exports folder.")
            return None, None
        
        # Pick the latest file by modification time
        latest_file = max(csv_files, key=os.path.getmtime)
        df = pd.read_csv(latest_file)
        #st.success(f"✅ Loaded data from {latest_file} with {len(df)} records.")
        
        
        

        
        # Rename columns for clarity
        df = df.rename(columns={
            'Auction ID/CIN/LLPIN': 'Auction ID',
            'Bank/Organisation Name': 'Bank',
            'Location-City/District/address': 'Location',
            '_Auction date': 'Auction Date',
            '_Last Date of EMD Submission': 'EMD Submission Date',
            '_Reserve Price': '₹Reserve Price',
            'EMD Amount': '₹EMD Amount',
            'Nature of Assets': 'Nature of Assets',
            'Details URL': 'Details URL',
            'Auction Notice URL': 'Notice URL',
            'Source': 'Source',
            'Notice_date': 'Notice_date'
        })
        # Convert date columns to datetime64[ns] and create duplicate columns for filtering
        df['EMD Submission Date_dt'] = pd.to_datetime(df['EMD Submission Date'], format="%d-%m-%Y", errors='coerce')
        df['Auction Date_dt'] = pd.to_datetime(df['Auction Date'], format="%d-%m-%Y", errors='coerce')
        df['Notice_date'] = pd.to_datetime(df['Notice_date'], format="%d/%m/%Y", errors='coerce')

        # Convert date columns to datetime64[ns] and format as strings for display
        df['EMD Submission Date'] = pd.to_datetime(df['EMD Submission Date'], format="%d-%m-%Y", errors='coerce')
        df['Auction Date'] = pd.to_datetime(df['Auction Date'], format="%d-%m-%Y", errors='coerce')

        # Convert to string format to avoid Arrow conversion issues (only date part)
        df['EMD Submission Date'] = df['EMD Submission Date'].dt.strftime('%d-%m-%Y')
        df['Auction Date'] = df['Auction Date'].dt.strftime('%d-%m-%Y')

        # Use tz-naive date for "today" (as datetime object for consistency in calculations)
        today_date = pd.Timestamp.now(tz=None).date()

        # Calculate days_until_submission safely
        if 'days_until_submission' not in df.columns:
            df['days_until_submission'] = df['EMD Submission Date'].apply(
                lambda x: (pd.to_datetime(x).date() - today_date).days if pd.notna(x) and x != '' else -999
            )
        # Clean numeric columns
        df['₹Reserve Price'] = pd.to_numeric(df['₹Reserve Price'].astype(str).str.replace(r'[,₹\s]', '', regex=True), errors='coerce')
        df['₹EMD Amount'] = pd.to_numeric(df['₹EMD Amount'].astype(str).str.replace(r'[,₹\s]', '', regex=True), errors='coerce')

        # Calculate EMD % and categorize
        # Calculate EMD %
        df['EMD %'] = (df['₹EMD Amount'] / df['₹Reserve Price'] * 100).round(2)

        # Define bins and labels
        bins = [-float("inf"), 5, 10, 15, 20, float("inf")]
        labels = ["<5%", "5-10%", "10-15%", "15-20%", ">20%"]

        # Categorize into bins
        df['EMD % Category'] = pd.cut(df['EMD %'], bins=bins, labels=labels, right=False)
      

        if df['EMD Submission Date'].isna().any():
            pass
            #st.warning("⚠️ Some EMD Submission Dates could not be parsed and are set to NaT. These rows may have invalid data.")

        return df, csv_path
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        return None, None








#####################################################################################################################################################################
########################################################################################################################################################################







# Load data
df, latest_csv = load_auction_data()

# Dashboard Page
if page == "🏠 Dashboard" and df is not None:
    st.markdown('<div class="main-header">🏛️ Auction Portal India</div>', unsafe_allow_html=True)
    #st.markdown(f"**Last Updated:** {latest_csv.split('_')[-1].split('.')[0] if latest_csv else 'Unknown'}")


    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_auctions = len(df)
        st.metric("Total Auctions", total_auctions)
    
    with col2:
        invalid_count = df['EMD Submission Date'].isna().sum()
        st.metric("Invalid EMD Dates", invalid_count)
    
    with col3:
        active_auctions = len(df[df['days_until_submission'] >= 0])
        st.metric("Active Auctions", active_auctions)

    from babel.numbers import format_currency
    import pandas as pd

    def format_indian_currency(value):
        if pd.isna(value) or value <= 0:
            return "N/A"
        # Convert to lakhs or crores
        if value >= 10000000:  # 1 crore = 10 million
            formatted = value / 10000000
            return f"{formatted:.2f} cr"
        elif value >= 100000:  # 1 lakh = 100,000
            formatted = value / 100000
            return f"{formatted:.2f} lakhs"
        else:
            return f"{value:.2f}"

    with col4:
        avg_reserve = df[df['days_until_submission'] >= 0]['₹Reserve Price'].mean()
        formatted_value = format_indian_currency(avg_reserve)
        st.metric("Avg Reserve Price of active auctions", formatted_value)
    # Display filtered data
    filtered_df = df[df['days_until_submission'] >= 0]
    if not filtered_df.empty:
        st.dataframe(filtered_df[['Auction ID', 'Bank', 'Location', 'Auction Date', 'EMD Submission Date',
                                 '₹Reserve Price', '₹EMD Amount', 'EMD %', 'EMD % Category', 'Nature of Assets'
                                 ,   'days_until_submission']],
                     use_container_width=True)
        st.write(f"**Total Auctions (Today or Future):** {len(filtered_df)}")
 
    else:
        st.info("✅ No auctions found for day or future dates.")
        








#####################################################################################################################################################################
########################################################################################################################################################################








# Search Analytics Page
elif page == "🔍 Search Analytics" and df is not None:
    st.markdown('<div class="main-header">🔍 Search Analytics</div>', unsafe_allow_html=True)
    #st.markdown(f"**Last Updated:** {latest_csv.split('_')[-1].split('.')[0] if latest_csv else 'Unknown'}")
    

   

    filtered_df = df[df['days_until_submission'] >= 0].copy()
    

    # Location Filter
    use_location_filter = st.checkbox("Use Location Filter", value=False)
    if use_location_filter:
        unique_locations = sorted(filtered_df['Location'].dropna().unique())
        selected_locations = st.multiselect(
            "Select Locations",
            options=unique_locations,
            default=None
        )
        if selected_locations:
            filtered_df = filtered_df[filtered_df['Location'].isin(selected_locations)]

    # Range Slider for days_until_submission
    use_days_filter = st.checkbox("Use Days Until Submission Filter", value=False)
    if use_days_filter and not filtered_df.empty:
        min_days = int(filtered_df['days_until_submission'].min())
        max_days = int(filtered_df['days_until_submission'].max())
        days_range = st.slider(
            "Filter by Days Until Submission",
            min_value=min_days,
            max_value=max_days,
            value=(min_days, max_days)
        )
        filtered_df = filtered_df[
            (filtered_df['days_until_submission'] >= days_range[0]) &
            (filtered_df['days_until_submission'] <= days_range[1])
        ]

    # Checkbox and Date Input for EMD Submission Date
    use_date_filter = st.checkbox("Use EMD Submission Date Filter", value=False)
    if use_date_filter:
        selected_date = st.date_input("Select EMD Submission Date", value=pd.Timestamp.now(tz=None).date(), disabled=not use_date_filter)
        filtered_df = filtered_df[filtered_df['EMD Submission Date_dt'].dt.date == selected_date]

   # EMD % Filter
    use_emd_percent_filter = st.checkbox("Use EMD % Filter", value=False)
    if use_emd_percent_filter:
        emd_options = ["<5%", "5-10%", "10-15%", "15-20%", ">20%"]
        selected_emd = st.multiselect(
            "Select EMD % Category",
            options=emd_options,
            default=None
        )
        if selected_emd:
            mask = filtered_df['EMD % Category'].str.contains('|'.join(selected_emd), na=False).fillna(False)
            filtered_df = filtered_df[mask]

    # Drop rows with any NaN values across all columns
    #filtered_df = filtered_df.dropna()

    if not filtered_df.empty:
        st.dataframe(filtered_df[['Auction ID', 'Bank', 'Location', 'Auction Date', 'EMD Submission Date',
                                 '₹Reserve Price', '₹EMD Amount', 'EMD %', 'EMD % Category', 'Nature of Assets'
                                 , 'days_until_submission']],
                     use_container_width=True)
        st.write(f"**Total Auctions:** {len(filtered_df)}")
    else:
        st.info("✅ No auctions found with the selected filters.")










#####################################################################################################################################################################
########################################################################################################################################################################


# Basic Analytics Page
elif page == "📊 Basic Analytics" and df is not None:
    st.markdown('<div class="main-header">📊 Basic Analytics</div>', unsafe_allow_html=True)
    
    # Inject custom CSS for improved metric tiles
    st.markdown("""
        <style>
            .metric-grid {
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                padding: 15px;
            }
            .metric-tile {
                background-color: #ffffff;
                border-radius: 10px;
                padding: 15px;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                transition: transform 0.2s, box-shadow 0.2s;
                border: 1px solid #e0e0e0;
                flex: 1;
                min-width: 200px;
            }
            .metric-tile:hover {
                transform: translateY(-5px);
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
            }
            .metric-tile h3 {
                font-size: 1.5em;
                margin: 0;
                color: #1a73e8;
                font-weight: bold;
            }
            .metric-tile p {
                font-size: 0.9em;
                margin: 5px 0 0 0;
                color: #5f6368;
                font-weight: 500;
            }
        </style>
    """, unsafe_allow_html=True)

    # Row 1: Total Auctions and Active Auctions
    col1_row1, col2_row1 = st.columns(2)
    with col1_row1:
        st.markdown("""
            <div class="metric-tile">
                <h3>{}</h3>
                <p>Total Auctions</p>
            </div>
        """.format(len(df)), unsafe_allow_html=True)
    with col2_row1:
        active_auctions = len(df[df['days_until_submission'] >= 0])
        st.markdown("""
            <div class="metric-tile">
                <h3>{}</h3>
                <p>Active Auctions</p>
            </div>
        """.format(active_auctions), unsafe_allow_html=True)

    # Row 2: Avg Reserve Price (All) and Avg Reserve Price of Active Auctions
    col1_row2, col2_row2 = st.columns(2)
    with col1_row2:
        from babel.numbers import format_currency
        import textwrap
        def format_indian_currency(value):
            if pd.isna(value) or value <= 0:
                return "N/A"
            if value >= 10000000:  # 1 crore = 10 million
                formatted = value / 10000000
                return f"{formatted:.2f} cr"
            elif value >= 100000:  # 1 lakh = 100,000
                formatted = value / 100000
                return f"{formatted:.2f} lakhs"
            else:
                return f"{value:.2f}"
        
        avg_reserve_all = df['₹Reserve Price'].mean()
        formatted_value_all = format_indian_currency(avg_reserve_all)
        st.markdown("""
            <div class="metric-tile">
                <h3>{}</h3>
                <p>Avg Reserve Price (All)</p>
            </div>
        """.format(formatted_value_all), unsafe_allow_html=True)
    with col2_row2:
        avg_reserve_active = df[df['days_until_submission'] >= 0]['₹Reserve Price'].mean()
        formatted_value_active = format_indian_currency(avg_reserve_active)
        st.markdown("""
            <div class="metric-tile">
                <h3>{}</h3>
                <p>Avg Reserve Price of Active Auctions</p>
            </div>
        """.format(formatted_value_active), unsafe_allow_html=True)

    # Row 3: Sum of Reserve Price (All) and Sum of Reserve Price of Active Auctions
    col1_row3, col2_row3 = st.columns(2)
    with col1_row3:
        sum_reserve_all = df['₹Reserve Price'].sum()
        formatted_value_sum_all = format_indian_currency(sum_reserve_all)
        st.markdown("""
            <div class="metric-tile">
                <h3>{}</h3>
                <p>Sum of Reserve Price (All)</p>
            </div>
        """.format(formatted_value_sum_all), unsafe_allow_html=True)
    with col2_row3:
        sum_reserve_active = df[df['days_until_submission'] >= 0]['₹Reserve Price'].sum()
        formatted_value_sum_active = format_indian_currency(sum_reserve_active)
        st.markdown("""
            <div class="metric-tile">
                <h3>{}</h3>
                <p>Sum of Reserve Price of Active Auctions</p>
            </div>
        """.format(formatted_value_sum_active), unsafe_allow_html=True)

    # Row 5: Min and Max of Reserve Price of Active Auctions
    col1_row5, col2_row5 = st.columns(2)
    with col1_row5:
        min_reserve_active = df[df['days_until_submission'] >= 0]['₹Reserve Price']
        if not min_reserve_active.empty:
            min_reserve_active = min_reserve_active[min_reserve_active > 0].min() if (min_reserve_active > 0).any() else float('nan')
        else:
            min_reserve_active = float('nan')
        formatted_min_active = format_indian_currency(min_reserve_active)
        st.markdown("""
            <div class="metric-tile">
                <h3>{}</h3>
                <p>Min of Reserve Price of Active Auctions</p>
            </div>
        """.format(formatted_min_active), unsafe_allow_html=True)
    with col2_row5:
        max_reserve_active = df[df['days_until_submission'] >= 0]['₹Reserve Price'].max()
        formatted_max_active = format_indian_currency(max_reserve_active)
        st.markdown("""
            <div class="metric-tile">
                <h3>{}</h3>
                <p>Max of Reserve Price of Active Auctions</p>
            </div>
        """.format(formatted_max_active), unsafe_allow_html=True)

    st.markdown("---")

    # Top 5 Banks with Min and Max Reserve Price as a DataFrame
    top_banks = df['Bank'].value_counts().head(5).index
    active_df = df[df['days_until_submission'] >= 0]
    bank_stats = []
    for bank in top_banks:
        bank_data = active_df[active_df['Bank'] == bank]['₹Reserve Price']
        min_price = bank_data[bank_data > 0].min() if (bank_data > 0).any() else float('nan')
        max_price = bank_data[bank_data > 0].max() if (bank_data > 0).any() else float('nan')
        bank_stats.append({
            'Bank': bank,
            'Min Reserve Price': min_price,
            'Max Reserve Price': max_price
        })
    bank_df = pd.DataFrame(bank_stats)
    bank_df['Min Reserve Price'] = bank_df['Min Reserve Price'].apply(format_indian_currency)
    bank_df['Max Reserve Price'] = bank_df['Max Reserve Price'].apply(format_indian_currency)
    st.subheader("📈 Top 5 Banks by Reserve Price ")
    st.dataframe(bank_df)

    st.markdown("---")
    # Chart 1: Top 10 Banks by Auction Count
    st.subheader("📈 Top 10 Banks by Auction Count")
    bank_counts = df['Bank'].value_counts().head(10)
    fig1 = px.bar(
        x=bank_counts.values,
        y=bank_counts.index,
        orientation='h',
        title="Top 10 Banks by Auction Count",
        labels={'x': 'Number of Auctions', 'y': 'Bank'},
        color=bank_counts.values,
        color_continuous_scale='viridis'
    )
    fig1.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2: Average Reserve Price by Location (Top 10)
    st.subheader("💰 Top 10 Locations by Average Reserve Price")
    location_avg = df.groupby('Location')['₹Reserve Price'].mean().sort_values(ascending=False).head(10)
    fig2 = px.bar(
        x=location_avg.apply(format_indian_currency),
        y=location_avg.index,
        orientation='h',
        title="Top 10 Locations by Average Reserve Price",
        labels={'x': 'Average Reserve Price', 'y': 'Location'},
        color=location_avg.values,
        color_continuous_scale='plasma'
    )
    fig2.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    # Chart 3: EMD Percentage Distribution
    st.subheader("📊 EMD Percentage Distribution")
    emd_dist = df['EMD %'].dropna()
    fig3 = px.histogram(
        emd_dist,
        title="Distribution of EMD Percentages",
        labels={'value': 'EMD %', 'count': 'Frequency'},
        nbins=50,
        color_discrete_sequence=['#ff6b6b']
    )
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

    # Chart 4: Auctions Over Time
    st.subheader("📅 Auction Trends Over Time")
    if not df['Auction Date_dt'].isna().all():
        df_time = df.dropna(subset=['Auction Date_dt']).copy()
        df_time['Month'] = df_time['Auction Date_dt'].dt.to_period('M').dt.to_timestamp()
        monthly_auctions = df_time.groupby('Month').size().reset_index(name='Count')
        
        fig4 = px.line(
            monthly_auctions,
            x='Month',
            y='Count',
            title="Number of Auctions per Month",
            labels={'Count': 'Number of Auctions', 'Month': 'Month'}
        )
        fig4.update_traces(line_color='#2e7d32', line_width=3)
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No valid auction dates available for time trend analysis.")

    # Chart 5: Reserve Price vs EMD Amount Scatter
    st.subheader("💸 Reserve Price vs EMD Amount")
    scatter_data = df[df['days_until_submission'] >= 0].dropna(subset=['₹Reserve Price', '₹EMD Amount'])
    if not scatter_data.empty:
        fig5 = px.scatter(
            scatter_data,
            x='₹Reserve Price',
            y='₹EMD Amount',
            title="Reserve Price vs EMD Amount",
            labels={'x': 'Reserve Price (₹)', 'y': 'EMD Amount (₹)'},
            opacity=0.6,
            color='EMD %',
            color_continuous_scale='viridis'
        )
        combined_text = scatter_data.apply(lambda row: f"{format_indian_currency(row['₹Reserve Price'])} / {format_indian_currency(row['₹EMD Amount'])}", axis=1)
        fig5.update_traces(text=combined_text, textposition='top center')
        fig5.update_layout(height=500)
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("No valid price data available for scatter plot.")

#####################################################################################################################################################################
########################################################################################################################################################################      

# Sidebar Navigation (update the radio options)


# ... (existing code for other pages)

# KPI Analytics Page
elif page == "📈 KPI Analytics" and df is not None:
    st.markdown('<div class="main-header">📈 KPI Analytics</div>', unsafe_allow_html=True)
    
    # Filter for active auctions
    active_df = df[df['days_until_submission'] >= 0]
    active_df1=active_df[active_df["Source"]!="Albion"]
    
    if not active_df.empty:

        # notice compliance rate (proxy)
        total_auctions = len(active_df1)
        compliant_auctions = len(active_df1[active_df1['Notice URL'] != 'URL 2_if available'])
        notice_compliance_rate = (compliant_auctions / total_auctions * 100) if total_auctions > 0 else 0


        # Compute Disclosure Timeliness (proxy: auction_date - emd_submission_date)
        active_df1['timeliness_days'] = (active_df1['Auction Date_dt'] - active_df['Notice_date']).dt.days
        min_days = active_df1['timeliness_days'].min()
        median_days = active_df1['timeliness_days'].median()
        p95_days = active_df1['timeliness_days'].quantile(0.95)
        
        # Compute Data Quality Error Rate
        error_rate = (active_df.isna().any(axis=1).sum() / len(active_df)) * 100
        
        # Display in cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
                <div class="metric-tile">
                    <h3>Disclosure Timeliness (Days)</h3>
                    <p>Min: {min_days}, Median: {median_days}, P95: {p95_days}</p>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div class="metric-tile">
                    <h3>Notice Compliance Rate (Proxy)</h3>
                    <p>{notice_compliance_rate:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="metric-tile">
                    <h3>Data Quality Error Rate</h3>
                    <p>{error_rate:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.write(f"**Active Auctions Analyzed:** {len(active_df)}")
    else:
        st.info("No active auctions available for KPI calculation.")
    
    st.markdown("---")




#####################################################################################################################################################################
########################################################################################################################################################################  

# AI Analysis Page

def display_insights(insights: dict):
    st.success("Insights generated successfully!")

    # Summary section
    st.markdown("### Auction Summary")
    st.markdown(f"**Corporate Debtor:** {insights.get('corporate_debtor', '')}")
    st.markdown(f"**Auction Date:** {insights.get('auction_date', '')}")
    st.markdown(f"**Auction Time:** {insights.get('auction_time', '')}")
    st.markdown(f"**Inspection Date:** {insights.get('inspection_date', '')}")
    st.markdown(f"**Inspection Time:** {insights.get('inspection_time', '')}")
    st.markdown(f"**Auction Platform:** {insights.get('auction_platform', '')}")
    st.markdown(f"**Contact Email:** {insights.get('contact_email', '')}")
    st.markdown(f"**Contact Mobile:** {insights.get('contact_mobile', '')}")

    # Assets
    assets = insights.get("assets", [])
    if assets:
        st.markdown("### Assets Information")
        for asset in assets:
            st.markdown(f"**Block Name:** {asset.get('block_name', '')}")
            st.markdown(f"**Description:** {asset.get('asset_description', '')}")
            st.markdown(f"**Auction Time:** {asset.get('auction_time', '')}")
            st.markdown(f"**Reserve Price:** {asset.get('reserve_price', '')}")
            st.markdown(f"**EMD Amount:** {asset.get('emd_amount', '')}")
            st.markdown(f"**Incremental Bid Amount:** {asset.get('incremental_bid_amount', '')}")
            st.markdown("---")

    # Financial Terms
    financial = insights.get("financial_terms", {})
    if financial:
        st.markdown("### Financial Terms")
        st.markdown(f"**EMD Amount:** {financial.get('emd', '')}")
        bid_increments = financial.get("bid_increments", [])
        if bid_increments:
            st.markdown("**Bid Increments:**")
            for inc in bid_increments:
                st.markdown(f"- {inc}")

    # Timeline
    timeline = insights.get("timeline", {})
    if timeline:
        st.markdown("### Timeline")
        st.markdown(f"**Auction Date:** {timeline.get('auction_date', '')}")
        st.markdown(f"**Inspection Period:** {timeline.get('inspection_period', '')}")

    # Ranking
    ranking = insights.get("ranking", {})
    if ranking:
        st.markdown("### Auction Ranking")
        st.markdown(f"**Legal Compliance Score:** {ranking.get('legal_compliance_score', 0)}")
        st.markdown(f"**Economical Score:** {ranking.get('economical_score', 0)}")
        st.markdown(f"**Market Trends Score:** {ranking.get('market_trends_score', 0)}")
        st.markdown(f"**Final Score:** {ranking.get('final_score', 0)}")
        st.markdown(f"**Risk Summary:** {ranking.get('risk_summary', '')}")
        references = ranking.get("reference_summary", [])
        if references:
            st.markdown("**Reference Summary:**")
            for ref in references:
                st.markdown(f"- {ref}")

# Load data
df, latest_csv = load_auction_data()

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


# Text Processing Functions

def is_pdf_scanned(pdf_bytes: bytes) -> bool:
    """
    Determine if a PDF is scanned by checking for the absence of text
    and the presence of images.
    """
    try:
        # Check for extractable text using pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            has_text = any(page.extract_text() for page in pdf.pages)
            if has_text:
                return False  

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        has_images = any(page.get_images(full=True) for page in doc)
       
        return has_images
    except Exception as e:
        print(f"[ERROR] is_pdf_scanned failed: {e}")
        return False
   
def extract_json_from_text(text: str) -> dict:
    """
    Attempts to extract the first valid JSON object from a text string.
    Returns an empty dict if extraction fails.
    """
    try:
        # Match JSON object from first '{' to last '}'
        match = re.search(r"{.*}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        print(f"[ERROR] JSON extraction failed: {e}")
    return {}
   

    return clean_assets(all_assets)


def extract_assets_from_text(text: str) -> list:
    assets = []
    reserve_price, emd_amount, incremental_bid = "", "", ""

    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("reserve price"):
            reserve_price = line.split(":", 1)[-1].strip()
        elif line.lower().startswith("emd amount"):
            emd_amount = line.split(":", 1)[-1].strip()
        elif line.lower().startswith("incremental bid amount"):
            incremental_bid = line.split(":", 1)[-1].strip()

    current_asset = {
        "block_name": "",
        "asset_description": "",
        "reserve_price": reserve_price,
        "emd_amount": emd_amount,
        "incremental_bid_amount": incremental_bid,
    }
    assets.append(current_asset)
    return assets


def format_tables_as_markdown(tables: List[List[List[str]]]):
    markdown = ""
    for table in tables:
        if not table or len(table) < 2:
            continue
        headers = [h.strip() if h else "" for h in table[0]]
        markdown += "\n| " + " | ".join(headers) + " |\n"
        markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in table[1:]:
            row = [cell.strip().replace("\n", " ") if cell else "" for cell in row]
            while len(row) < len(headers):
                row.append("")
            markdown += "| " + " | ".join(row) + " |\n"
    return markdown

def extract_tables_with_camelot(pdf_bytes: bytes, page_number: int = None) -> List[List[List[str]]]:
    """
    Extracts tables from a specific page of a PDF using Camelot.
    Tries both 'lattice' and 'stream' flavors.
    """
    tables = []
    page_str = "all" if page_number is None else str(page_number)
   
    try:
        # Try 'lattice' for tables with clear lines
        print(f"[INFO] Trying Camelot with 'lattice' flavor on page {page_str}...")
        lattice_tables = camelot.read_pdf(io.BytesIO(pdf_bytes), pages=page_str, flavor='lattice')
        if lattice_tables.n > 0:
            tables.extend([table.data for table in lattice_tables])
            print(f"[INFO] Found {lattice_tables.n} table(s) with 'lattice' flavor.")
            return tables

        # If that fails, try 'stream' for tables without clear lines
        print(f"[INFO] 'lattice' failed, trying 'stream' flavor on page {page_str}...")
        stream_tables = camelot.read_pdf(io.BytesIO(pdf_bytes), pages=page_str, flavor='stream')
        if stream_tables.n > 0:
            tables.extend([table.data for table in stream_tables])
            print(f"[INFO] Found {stream_tables.n} table(s) with 'stream' flavor.")
            return tables
           
    except Exception as e:
        print(f"[ERROR] Camelot table extraction failed: {e}")
       
    return tables

def ocr_pdf(pdf_bytes: io.BytesIO) -> Tuple[str, List]:
    """
    Runs OCR on all pages of a PDF and returns extracted text + empty table list.
    """
    ocr_text = ""
    tables = []

    try:
        images = convert_from_bytes(pdf_bytes.getvalue())
        for img in images:
            text = pytesseract.image_to_string(img)
            ocr_text += text.strip() + "\n"
    except Exception as e:
        print(f"[ERROR] OCR failed: {e}")

    return ocr_text, tables





def fetch_text_from_url(pdf_url: str) -> Tuple[str, List, bool]:
    response = requests.get(pdf_url, timeout=15)
    response.raise_for_status()
    pdf_bytes = response.content
    pdf_io = io.BytesIO(pdf_bytes)

    raw_text = ""
    tables = []
    scanned_pdf = False

    try:
        with pdfplumber.open(pdf_io) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                raw_text += page_text.strip() + "\n"

        # Checked if pdfplumber failed to extract text, and use OCR if so.
        if not raw_text.strip():
            print("[INFO] pdfplumber extracted no text. Falling back to OCR...")
            scanned_pdf = True
            raw_text, tables = ocr_pdf(pdf_io)
       
        # If no tables were found, use Camelot as a fallback.
        if not tables and not scanned_pdf:
            print("[INFO] pdfplumber did not find any tables. Trying Camelot on All Pages...")
            tables = extract_tables_with_camelot(pdf_bytes)

    except Exception as e:
        print(f"[ERROR] PDF extraction failed: {e}")
        # If pdfplumber fails entirely, we check if it's a scanned PDF
        if is_pdf_scanned(pdf_bytes):
            scanned_pdf = True
            raw_text, tables = ocr_pdf(pdf_io)
        else:
            print("[INFO] PDF extraction failed but document is not scanned. No OCR fallback.")
            raw_text = ""
            tables = []

    return raw_text.strip(), tables, scanned_pdf

def truncate_text(text: str, max_words: int = 5000) -> str:
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text

def clean_llm_output(output: str) -> str:
    import re
    match = re.search(r"{.*}", output, re.DOTALL)
    return match.group(0) if match else output

def normalize_keys(obj):
    """Lowercase keys and replace spaces with underscores for consistency."""
    if isinstance(obj, dict):
        return {k.lower().replace(" ", "_"): normalize_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [normalize_keys(i) for i in obj]
    else:
        return obj

def clean_assets(assets: list) -> list:
    """
    Cleans OCR noise from asset descriptions and formats numeric values.
    """
    cleaned_assets = []

    for asset in assets:
        cleaned_asset = asset.copy()

        # Clean asset description
        desc = asset.get("asset_description", "")
        desc = re.sub(r"http\S+|www\.\S+", "", desc)  
        desc = re.sub(r"\S+@\S+", "", desc)  
        desc = re.sub(r"\s+", " ", desc)  
        cleaned_asset["asset_description"] = desc.strip()

        # Clean numeric fields
        for field in ["reserve_price", "emd_amount", "incremental_bid_amount"]:
            value = asset.get(field, "")
            cleaned_asset[field] = value.strip()

        cleaned_assets.append(cleaned_asset)

    return cleaned_assets

def enforce_numeric_fields(asset: dict) -> dict:
    """
    Ensure Reserve Price, EMD, Incremental Bid keep ONLY the numeric Rs. value
    """
    for field in ["reserve_price", "emd_amount", "incremental_bid_amount"]:
        if asset.get(field):
            match = re.search(r"(Rs\.\s*[\d,]+/-)", asset[field])
            if match:
                asset[field] = match.group(1)  
            else:
                asset[field] = asset[field].split("(")[0].strip()
    return asset

class AuctionDetails(BaseModel):
    name_of_corporate_debtor_pdf_: str = Field(..., alias="Name of Corporate Debtor (PDF)")
    auction_notice_url: str = Field(..., alias="Auction Notice URL")
    date_of_auction_pdf: Optional[str] = Field(None, alias="Date of Auction (PDF)")
    unique_number: Optional[str] = Field(None, alias="Unique Number")
    ip_registration_number: Optional[str] = Field(None, alias="IP Registration Number")
    auction_platform: Optional[str] = Field(None, alias="Auction Platform")
    details_url: Optional[str] = Field(None, alias="Details URL")
   
    borrower_name: Optional[str] = None
   
    model_config = {
        "validate_by_name": True,   
        "extra": "ignore"           
    }
   
def generate_auction_insights(corporate_debtor: str, auction_notice_url: str, llm) -> dict:
    try:
       
        raw_text, tables, scanned_pdf = fetch_text_from_url(auction_notice_url)

        fallback_assets = extract_assets_from_text(raw_text)
        use_fallback = not tables or len(tables[0]) <= 2

        assets_for_prompt = None
        markdown_table = ""
        if use_fallback:
            logging.warning("[FALLBACK] Using extracted assets from text due to missing/bad table")
            assets_for_prompt = clean_assets(fallback_assets)
        else:
            markdown_table = format_tables_as_markdown(tables)

        if not raw_text.strip():
            return {"status": "error", "message": "No usable text found in auction notice."}

        truncated_text = truncate_text(raw_text)

       
        assets_section = (
            f"\nAssets (extracted via OCR fallback):\n{json.dumps(assets_for_prompt, indent=2)}"
            if assets_for_prompt else markdown_table
        )

       
        prompt = f"""
You are an expert financial analyst specializing in Indian auction notices. Your primary role is to audit the listing quality and risk.

Your job is to carefully extract key details from the below auction notice text.
It may contain normal paragraphs, markdown tables, or pre-parsed OCR asset JSON. Read all content carefully.

Corporate Debtor: {corporate_debtor}

Auction Notice Text:
{truncated_text}

{assets_section}

# RISK SCORING FRAMEWORK (Use this internally for scoring)

## HIGH RISK (Block/Hold - Score 0-3)
Assign this risk level if ANY of the following Critical Defects are present. If a High Risk item is found, the Legal Compliance Score MUST be in the 0-3 range.
- **Statutory Defects:** Notice has legal defects (e.g., notice period shorter than mandated; missing authorized officer name/signature/seal).
- **Critical Mismatch:** Key details (e.g., property size/reserve price) differ significantly between the official notice PDF and the listing data.
- **Missing Core Docs:** Critical artifacts are missing (Sale Notice PDF, Valuation Report, Title documents).
- **Expired Valuation:** The valuation report date is older than 6-12 months (stale valuation).
- **Extreme Price Outlier:** Reserve price is an extreme outlier (e.g., > +50% or < -40% vs. comparable properties/norms).
- **Known Litigation:** Known, unresolved litigation (lis pendens, stay order) is disclosed.
- **Process Integrity:** Frequent re-schedules (>=3) without adequate cause, OR outcome anomalies (postings contradict terms).

## AVERAGE RISK (Warn Users - Score 4-7)
Assign this risk level if NO High Risk items are present, but ANY of the following Moderate Defects are found. The Legal Compliance Score MUST be in the 4-7 range.
- **Ambiguity:** Property description is ambiguous or inconsistent across documents.
- **Missing Minor Annexures:** Minor supporting documents (e.g., uncertified translations) are missing.
- **Low Quality:** Low photo count (<=3 .photos) or poor readability of scanned documents.
- **Mild Price Outlier:** Reserve price is a mild outlier (e.g., 10-25% out of band).
- **Short EMD Window:** Tight gap between notice and EMD close date.
- **Multiple Re-Auctions:** Listing mentions multiple prior re-auctions with ad-hoc reserve changes (no method cited).
- **Non-Standard Contact:** Personal emails/phones used in notices instead of official domains.

## LOW / NO RISK (Informational - Score 8-10)
Assign this risk level if NO High Risk or Average Risk items are found. The Legal Compliance Score MUST be in the 8-10 range.
- **Minor Typos:** Only minor typos/formatting errors that don't change legal meaning.
- **Normal Dynamics:** Events like last-minute bidding or re-auction due to reserve not met (1-2 cycles).

---

Please extract the following insights and return them as a structured JSON:

1. Extract all general details (e.g. dates, contacts, platform link, etc.) exactly as written — do not infer or modify them.
2. Use the provided markdown table or OCR asset JSON (whichever is present) to populate the `"Assets"` list. **This table/JSON is the definitive source for asset details.**
3. One row = one asset. Do not duplicate or infer missing rows.
4. If values are missing, leave them blank — do not guess. **Crucially, if a value like 'Incremental Bid Amount' is not present in the source text or table, leave its field empty.**
5. For Reserve Price, EMD Amount, and Incremental Bid Amount:
- Copy the value EXACTLY as written in the notice (e.g., 'Rs. 90,00,000/-').
- Preserve the numeric formatting (commas, Rs., /-).
- DO NOT expand numbers into words (e.g., do not write 'Ninety Lakh' or 'Nine Crore').

Additional Task:
Rank the Auction using the provided **RISK SCORING FRAMEWORK** and the three components:
- Legal Compliance (Score 0-10, based on the Framework)
- Economical Point of View (Score 0-10, based on asset value and market context)
- Market Trends (Score 0-10, based on timing and location factors)

Provide:
- Individual scores for each component (0–10).
- A final score (simple average of the three components).
- A single-line summary of risk: "High Risk", "Average Risk", or "Low/No Risk" based on the highest risk category found.
- A **Reference Summary** that consists of **exactly 8 bullet points** in the JSON array. This summary must be a **detailed, evidence-based audit report** that uses plain, easy-to-understand language. **For every point, you MUST include the specific data/text from the notice that justifies the conclusion, and explicitly state the legal or market standard where applicable.**

    1. **Primary Risk & Evidence:** State the assigned risk level and the single most critical issue found. **DO NOT copy text from other points.** (Example: 'AVERAGE RISK: The contact email "anilgoel@aaainsolvency.com" is non-standard.')
    2. **Justification of Primary Risk:** Explain the risk. (Example: This email uses a non-institutional domain, raising a minor integrity concern over accountability.)
    3. **Statutory Compliance Check:** Report on legal defects by **citing the legal standard and the full period**. (Example: Statutory defects were cleared. The 21-day notice period rule is met, as the period from [Notice Date] to [Auction Date] is compliant.)
    4. **Authorization/Evidence Check:** Report on authorization and evidence by citing the document/reference. (Example: Valid authorization evidence is present, citing the NCLT order/Resolution reference from the text, ensuring the sale is legally sound.)
    5. **Artifacts Check:** Report on critical and minor documents. (Example: All critical documents are present. Minor annexures (like uncertified translations) are missing, which is an Average Risk.)
    6. **Valuation Check:** Report on price outlier and valuation currency. (Example: The Reserve Price of [Price] is acceptable based on market norms. The valuation report date of [Valuation Date] is current and not expired, meeting the 6-12 month policy window.)
    7. **Process/Timeline Check:** Report on EMD window/re-auctions. (Example: The EMD window from [Notice Date] to [EMD Date] is adequate. No signs of multiple re-auctions or ad-hoc reserve changes were noted.)
    8. **Listing Quality Warning:** Report on photos/description/ambiguity. (Example: Listing quality is low. Property photos and detailed descriptions require significant improvement for better transparency.)

Return the result in this **exact JSON format**:

{{
    "Corporate Debtor": "...",
    "Auction Date": "...",
    "Auction Time": "...",
    "Last Date for EMD Submission": "...",
    "Inspection Date": "...",
    "Inspection Time": "...",
    "Property Description": "...",
    "Auction Platform": "...",
    "Contact Email": "...",
    "Contact Mobile": "...",
    "Assets": [
        {{
            "Block Name": "...",
            "Asset Description": "...",
            "Auction Time": "...",
            "Reserve Price": "...",
            "EMD Amount": "...",
            "Incremental Bid Amount": "..."
        }}
    ],
    "Ranking": {{
        "Legal Compliance Score": 0,
        "Economical Score": 0,
        "Market Trends Score": 0,
        "Final Score": 0,
        "Risk Summary": "...",
        "Reference Summary": [
            "...",
            "...",
            "...",
            "...",
            "...",
            "...",
            "...",
            "..."
        ]
    }}
}}
"""
   
        logging.info(f"[INFO] Prompt length: {len(prompt.split())} words")

        response = llm.invoke(prompt, max_tokens=2048, temperature=0.2, top_p=0.9)
        logging.info(f"[INFO] Raw LLM response: {response.content[:500]} ...")

        parsed = extract_json_from_text(response.content)
        normalized = normalize_keys(parsed)

        return {
            "status": "success",
            "scanned_pdf": scanned_pdf,
            "insights": normalized
        }

    except Exception as e:
        error_msg = f"An error occurred during insight generation: {str(e)}"
        logging.error(f"[ERROR] generate_auction_insights failed: {error_msg}")
        return {"status": "error", "message": error_msg}

if page == "🤖 AI Analysis":
    st.markdown('<div class="main-header">🤖 AI Analysis</div>', unsafe_allow_html=True)

    if df is None:
        st.error("No auction data loaded")
        st.stop()

    emd_col = 'EMD Submission Date'
    id_col = 'Auction ID'
    bank_col = 'Bank' 
    notice_url_col = 'Notice URL'

    # Check for the existence of required columns
    if emd_col not in df.columns or id_col not in df.columns or bank_col not in df.columns or notice_url_col not in df.columns:
        missing_cols = [col for col in [emd_col, id_col, bank_col, notice_url_col] if col not in df.columns]
        st.error(f"Required columns not found. Missing: {missing_cols}. Please check your raw data file.")
        st.stop()

    # Convert the EMD column to datetime objects
    df[emd_col] = pd.to_datetime(df[emd_col], format='%d-%m-%Y', errors='coerce')

    # Filter for active auctions
    today = datetime.now().date()
    df_active = df[
        (df[emd_col].dt.date >= today) &  
        (~df[notice_url_col].str.contains('URL 2_if available', case=False, na=False)) 
    ].copy()
    if df_active.empty:
        st.warning("No active auctions found in the dataset.")
        st.stop()
   
    auction_ids = df_active[id_col].dropna().unique()
    selected_id = st.selectbox("Select Auction ID (from CIN/LLPIN)", options=[""] + list(auction_ids))
   
    if selected_id:
        # Use df_active to find the selected row, and use the variable names
        selected_row = df_active[df_active[id_col] == selected_id]
       
        if selected_row.empty:
            st.warning("Selected Auction ID not found in the filtered active data.")
            st.stop()

        auction_data = selected_row.iloc[0].to_dict()

        # Use the variable names to get data from the dictionary
        corporate_debtor = auction_data.get(bank_col, '')
        auction_notice_url = auction_data.get(notice_url_col, '')

        if not corporate_debtor or not auction_notice_url:
            st.warning("Corporate Debtor name or Auction Notice URL missing for selected Auction ID.")
            st.stop()

        @st.cache_resource
        def initialize_llm():
            groq_api_key = st.secrets["GROQ_API_KEY"]
            return ChatGroq(
                model="deepseek-r1-distill-llama-70b",
                temperature=0,
                api_key=groq_api_key,
            )

        llm = initialize_llm()

        if st.button("Generate Insights", use_container_width=True):
            if not llm:
                st.error("LLM failed to initialize. Check GROQ_API_KEY secret in Streamlit Cloud.")
                st.stop()

            with st.spinner("Generating insights (This may take up to 30 seconds for PDF processing and LLM analysis)..."):
                try:
                    insights_result = generate_auction_insights(corporate_debtor, auction_notice_url, llm)

                    if insights_result["status"] == "success":
                        insight_data = insights_result["insights"]
                        if isinstance(insight_data, dict):
                            if "assets" in insight_data and isinstance(insight_data["assets"], list):
                                insight_data["assets"] = [
                                    enforce_numeric_fields(asset) for asset in insight_data["assets"]
                                ]
                               
                            display_insights(insight_data)
                        else:
                            st.markdown(insight_data)
                           
                    else:
                        st.error("Analysis Failed")
                        st.error(insights_result["message"])
                       
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")

