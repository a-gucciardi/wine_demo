import streamlit as st
import pandas as pd
import requests
from io import StringIO
from datetime import datetime
import numpy as np

today = datetime.today().date()
LOCAL = False  # True if running locally, False for production
GITHUB_RAW_URL = f"https://raw.githubusercontent.com/a-gucciardi/wine_demo/refs/heads/main/code/streamlit/formatted_infection_risks_{today}.csv"

# --- Page Configuration ---
st.set_page_config(
    page_title="Infection Risk Predictions Viewer",
    page_icon="🍇",
    layout="wide",
)

# --- App Title ---
st.title("Vineyard Infection Risk Predictions Viewer")

# --- Data Loading ---
@st.cache_data
def load_data_from_url(url):
    """
    Loads risk prediction data from a raw GitHub URL using the requests library.
    Returns a pandas DataFrame with the formatted CSV structure.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # This will raise an exception for bad status codes (4xx or 5xx)

        # Use StringIO to treat the string data as a file
        csv_file = StringIO(response.text)

        df = pd.read_csv(csv_file)

        # --- Data Validation and Transformation for New Format ---
        # Check if we have the expected columns for the new format
        if 'RagioneSociale' not in df.columns or 'NomeVigneto' not in df.columns:
            st.error("Error: The CSV file must contain 'RagioneSociale' and 'NomeVigneto' columns.")
            return None

        # Get risk columns (columns containing "Risk" or "Probabilita" in their name)
        metadata_cols = ['RagioneSociale', 'NomeVigneto']
        risk_cols = [col for col in df.columns if 'Risk' in col or 'Probabilita' in col]

        if not risk_cols:
            st.error("Error: No risk or probability columns found in the CSV file.")
            st.write("Available columns:", list(df.columns))
            return None

        # The risk values are already as percentages, convert to decimal for internal processing
        for col in risk_cols:
            # Convert to numeric, handling any non-numeric values
            df[col] = pd.to_numeric(df[col], errors='coerce') / 100

        # Check for any rows with all NaN risk values and report
        nan_rows = df[risk_cols].isna().all(axis=1).sum()
        if nan_rows > 0:
            st.warning(f"Warning: {nan_rows} rows have invalid risk data and will be excluded from analysis.")

        return df

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from URL: {e}")
        st.info("Please make sure the GitHub URL is correct and the repository is public.")
        return None
    except Exception as e:
        st.error(f"An error occurred while processing the data: {e}")
        return None


if not LOCAL :
    df_original = load_data_from_url(GITHUB_RAW_URL)
else:
    df_original = pd.read_csv("formatted_infection_risks_2025-07-25.csv")
    # The new format has RagioneSociale, NomeVigneto and risk percentage columns
    if 'RagioneSociale' not in df_original.columns or 'NomeVigneto' not in df_original.columns:
        st.error("Error: The CSV file must contain 'RagioneSociale' and 'NomeVigneto' columns.")

    # Get risk columns (columns containing "Risk" or "Probabilita" in their name)
    metadata_cols = ['RagioneSociale', 'NomeVigneto']
    risk_cols = [col for col in df_original.columns if 'Risk' in col or 'Probabilita' in col]

    if not risk_cols:
        st.error("Error: No risk or probability columns found in the CSV file.")
        st.write("Available columns:", list(df_original.columns))

    # The risk values are already as percentages, convert to decimal for internal processing
    for col in risk_cols:
        # Convert to numeric, handling any non-numeric values
        df_original[col] = pd.to_numeric(df_original[col], errors='coerce') / 100

    # Check for any rows with all NaN risk values and report
    nan_rows = df_original[risk_cols].isna().all(axis=1).sum()
    if nan_rows > 0:
        st.warning(f"Warning: {nan_rows} rows have invalid risk data and will be excluded from analysis.")


# Get risk columns for filtering and display
metadata_cols = ['RagioneSociale', 'NomeVigneto']
risk_cols = [col for col in df_original.columns if 'Risk' in col or 'Probabilita' in col]

# --- Sidebar for Filtering ---
st.sidebar.header("Filter Options")

# Filter by Company/RagioneSociale (multiselect)
if 'RagioneSociale' in df_original.columns:
    company_values = sorted(df_original['RagioneSociale'].dropna().unique())
    selected_companies = st.sidebar.multiselect(
        "Select Company/Ragione Sociale",
        options=company_values,
        default=[]
    )
else:
    selected_companies = []
    st.sidebar.warning("'RagioneSociale' column not found for filtering.")

# Filter by Vineyard Name (multiselect)
if 'NomeVigneto' in df_original.columns:
    vineyard_names = df_original['NomeVigneto'].unique()
    selected_vineyards = st.sidebar.multiselect(
        "Select Vineyard(s)",
        options=sorted(vineyard_names),
        default=[]
    )
else:
    selected_vineyards = []
    st.sidebar.warning("'NomeVigneto' column not found for filtering.")

# Filter by Risk Type
if risk_cols:
    # Create a more user-friendly selection for risk types
    risk_type_mapping = {}
    for col in risk_cols:
        if 'Oidium_Risk' in col:
            risk_type_mapping['Oidium Risk'] = col
        elif 'Peronospora_Risk' in col:
            risk_type_mapping['Peronospora Risk'] = col
        elif 'Avg_Risk' in col:
            risk_type_mapping['Average Risk'] = col
        elif 'Probabilita_OIDIO' in col:
            risk_type_mapping['Oidium Probability'] = col
        elif 'Probabilita_PERONOSPORA' in col:
            risk_type_mapping['Peronospora Probability'] = col
    
    selected_risk_types = st.sidebar.multiselect(
        "Select Risk Type(s)",
        options=list(risk_type_mapping.keys()),
        default=[]
    )
else:
    selected_risk_types = []
    risk_type_mapping = {}

# Filter by Risk Score (slider)
if risk_cols:
    # Calculate max risk across all risk columns for filtering
    max_risk_overall = df_original[risk_cols].max().max()
    selected_risk_range = st.sidebar.slider(
        "Filter by Maximum Risk (any type)",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.05
    )
else:
    selected_risk_range = (0.0, 1.0)
    st.sidebar.warning("Risk columns not found for risk filtering.")

# --- Applying Filters ---
df_filtered = df_original.copy()

if selected_companies:
    df_filtered = df_filtered[df_filtered['RagioneSociale'].isin(selected_companies)]

if selected_vineyards:
    df_filtered = df_filtered[df_filtered['NomeVigneto'].isin(selected_vineyards)]

# Filter by selected risk types and reorder columns as requested
if selected_risk_types:
    selected_risk_cols = [risk_type_mapping[risk_type] for risk_type in selected_risk_types if risk_type in risk_type_mapping]
    if selected_risk_cols:
        display_cols = metadata_cols + selected_risk_cols
    else:
        display_cols = df_filtered.columns.tolist()
else:
    # If no risk types selected, show all in the new requested order:
    # Average Risk, Oidium Risk, Oidium Probability, Peronospora Risk, Peronospora Probability
    selected_risk_cols = []
    ordered_cols = []
    
    # Find each column type and add in the new requested order
    avg_risk = next((col for col in risk_cols if 'Avg_Risk' in col), None)
    oidium_risk = next((col for col in risk_cols if 'Oidium_Risk' in col), None)
    oidium_prob = next((col for col in risk_cols if 'Probabilita_OIDIO' in col), None)
    peronospora_risk = next((col for col in risk_cols if 'Peronospora_Risk' in col), None)
    peronospora_prob = next((col for col in risk_cols if 'Probabilita_PERONOSPORA' in col), None)
    
    # Add columns in the new requested order: Avg, Oidium Risk, Oidium Prob, Peronospora Risk, Peronospora Prob
    if avg_risk:
        ordered_cols.append(avg_risk)
    if oidium_risk:
        ordered_cols.append(oidium_risk)
    if oidium_prob:
        ordered_cols.append(oidium_prob)
    if peronospora_risk:
        ordered_cols.append(peronospora_risk)
    if peronospora_prob:
        ordered_cols.append(peronospora_prob)
    
    selected_risk_cols = ordered_cols
    display_cols = metadata_cols + ordered_cols

# Filter by risk score - keep rows where max risk across selected risk types is within range
if selected_risk_cols:
    # Ensure all selected risk columns have numeric values
    for col in selected_risk_cols:
        df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
    
    # Drop rows with NaN values in risk columns
    df_filtered = df_filtered.dropna(subset=selected_risk_cols)
    
    if len(df_filtered) > 0:
        df_filtered['max_risk'] = df_filtered[selected_risk_cols].max(axis=1)
        df_filtered = df_filtered[
            (df_filtered['max_risk'] >= selected_risk_range[0]) &
            (df_filtered['max_risk'] <= selected_risk_range[1])
            ]
        df_filtered = df_filtered.drop('max_risk', axis=1)

# --- Displaying the Data ---
st.header("Risk Prediction Results")
st.write(f"Displaying {len(df_filtered)} of {len(df_original)} total vineyards.")

# Format risk columns as percentages for display
df_display = df_filtered[display_cols].copy()
for col in selected_risk_cols:
    if col in df_display.columns:
        # Handle NaN values and convert to percentage string
        df_display[col] = df_display[col].apply(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A"
        )

# Create a function to highlight high risk values
def highlight_high_risk(val):
    if isinstance(val, str) and val.endswith('%'):
        try:
            risk_value = float(val.replace('%', ''))
            if risk_value > 70:
                return 'background-color: #ffe6e6; color: #990000; font-weight: bold'
        except:
            pass
    return ''

# Apply styling to the dataframe
styled_df = df_display.style.applymap(highlight_high_risk, subset=[col for col in selected_risk_cols if col in df_display.columns])

st.dataframe(styled_df, use_container_width=True)

if st.checkbox("Show raw data for filtered results"):
    st.write(df_filtered[display_cols])

# --- Summary Statistics ---
st.header("Summary Statistics")

if selected_risk_cols:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("By Risk Type")
        for risk_col in selected_risk_cols:
            avg_risk = df_filtered[risk_col].mean()
            # Clean up column names for display
            if 'Oidium_Risk' in risk_col:
                risk_name = 'Oidium Risk'
            elif 'Peronospora_Risk' in risk_col:
                risk_name = 'Peronospora Risk'
            elif 'Avg_Risk' in risk_col:
                risk_name = 'Average Risk'
            elif 'Probabilita_OIDIO' in risk_col:
                risk_name = 'Oidium Probability'
            elif 'Probabilita_PERONOSPORA' in risk_col:
                risk_name = 'Peronospora Probability'
            else:
                risk_name = risk_col.replace('_', ' ')
            st.metric(f"Average {risk_name}", f"{avg_risk:.1%}")

        with col2:
            st.subheader("By Company")
        if 'RagioneSociale' in df_filtered.columns and len(df_filtered) > 0:
            company_stats = df_filtered.groupby('RagioneSociale')[selected_risk_cols].mean()
            # Show top 5 companies by average risk
            for company in company_stats.index[:5]:
                avg_risk = company_stats.loc[company].mean()
                st.metric(f"{company[:20]}..." if len(company) > 20 else company, f"{avg_risk:.1%}")

# --- Risk Distribution Chart ---
if len(selected_risk_cols) > 0:
    st.header("Risk Distribution")

    # Create a chart showing risk distribution by type
    if len(selected_risk_cols) == 1:
        # Single risk type - show histogram
        risk_col = selected_risk_cols[0]
        # Clean up column names for display
        if 'Oidium_Risk' in risk_col:
            risk_name = 'Oidium Risk'
        elif 'Peronospora_Risk' in risk_col:
            risk_name = 'Peronospora Risk'
        elif 'Avg_Risk' in risk_col:
            risk_name = 'Average Risk'
        elif 'Probabilita_OIDIO' in risk_col:
            risk_name = 'Oidium Probability'
        elif 'Probabilita_PERONOSPORA' in risk_col:
            risk_name = 'Peronospora Probability'
        else:
            risk_name = risk_col.replace('_', ' ')
        
        # Create histogram data
        hist_data = df_filtered[risk_col].value_counts().sort_index()
        st.bar_chart(hist_data)
        st.write(f"Distribution of {risk_name} levels across vineyards")
    else:
        # Multiple risk types - show comparison
        chart_data = df_filtered[selected_risk_cols].copy()
        # Clean up column names for chart
        new_column_names = []
        for col in chart_data.columns:
            if 'Oidium_Risk' in col:
                new_column_names.append('Oidium Risk')
            elif 'Peronospora_Risk' in col:
                new_column_names.append('Peronospora Risk')
            elif 'Avg_Risk' in col:
                new_column_names.append('Average Risk')
            elif 'Probabilita_OIDIO' in col:
                new_column_names.append('Oidium Probability')
            elif 'Probabilita_PERONOSPORA' in col:
                new_column_names.append('Peronospora Probability')
            else:
                new_column_names.append(col.replace('_', ' '))
        chart_data.columns = new_column_names
        
        # Show average values as bar chart
        avg_data = chart_data.mean().sort_values(ascending=False)
        st.bar_chart(avg_data)
        st.write("Average risk levels by type across all filtered vineyards")
