import streamlit as st
import pandas as pd
import requests
from io import StringIO
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Infection Risk Predictions Viewer",
    page_icon="🍇",
    layout="wide",
)

# --- App Title ---
st.title("Vineyard Infection Risk Predictions Viewer")
today = datetime.today().date()
GITHUB_RAW_URL = f"https://raw.githubusercontent.com/a-gucciardi/wine_demo/refs/heads/main/code/streamlit/infection_risk_7days_{today}.csv"


# --- Data Loading ---
@st.cache_data
def load_data_from_url(url):
    """
    Loads prediction data from a raw GitHub URL using the requests library.
    Returns a pandas DataFrame transformed to long format.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # This will raise an exception for bad status codes (4xx or 5xx)

        # Use StringIO to treat the string data as a file
        csv_file = StringIO(response.text)

        df = pd.read_csv(csv_file)

        # --- Data Validation and Transformation ---
        # Check if we have the expected columns
        if 'IdVigneto' not in df.columns or 'Infection' not in df.columns:
            st.error("Error: The CSV file must contain 'IdVigneto' and 'Infection' columns.")
            return None

        # Get date columns (all columns that are not metadata)
        metadata_cols = ['IdVigneto', 'Codice', 'NomeVigneto', 'Infection']
        date_cols = [col for col in df.columns if col not in metadata_cols]

        if not date_cols:
            st.error("Error: No date columns found in the CSV file.")
            return None

        # Transform from wide to long format
        df_long = pd.melt(
            df,
            id_vars=metadata_cols,
            value_vars=date_cols,
            var_name='PredictionDate',
            value_name='RiskPercentage'
        )

        # Convert PredictionDate to datetime
        df_long['PredictionDate'] = pd.to_datetime(df_long['PredictionDate'])

        # Clean the risk percentage values (remove % and convert to float)
        df_long['RiskPercentage'] = df_long['RiskPercentage'].astype(str).str.replace('%', '').astype(float) / 100

        # Create separate columns for each disease type
        df_pivot = df_long.pivot_table(
            index=['IdVigneto', 'Codice', 'NomeVigneto', 'PredictionDate'],
            columns='Infection',
            values='RiskPercentage',
            aggfunc='first'
        ).reset_index()

        # Flatten column names
        df_pivot.columns.name = None

        # Rename disease columns to match old format
        if 'Oidium' in df_pivot.columns:
            df_pivot['Oidium_risk'] = df_pivot['Oidium']
            df_pivot.drop('Oidium', axis=1, inplace=True)

        if 'Peronospora' in df_pivot.columns:
            df_pivot['Peronospora_risk'] = df_pivot['Peronospora']
            df_pivot.drop('Peronospora', axis=1, inplace=True)

        # Rename IdVigneto to match old format
        df_pivot.rename(columns={'IdVigneto': 'Vigneto_IdVigneto'}, inplace=True)

        return df_pivot

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from URL: {e}")
        st.info("Please make sure the GitHub URL is correct and the repository is public.")
        return None
    except Exception as e:
        st.error(f"An error occurred while processing the data: {e}")
        return None


# --- Main App Logic ---
df_original = load_data_from_url(GITHUB_RAW_URL)

# The rest of the app will only proceed if the data has been successfully loaded.
if df_original is not None:
    # --- Sidebar for Filtering ---
    st.sidebar.header("Filter Options")

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

    # Filter by Date Range
    min_date = df_original['PredictionDate'].min().date()
    max_date = df_original['PredictionDate'].max().date()
    selected_date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Filter by Risk Score (slider)
    if 'Oidium_risk' in df_original.columns and 'Peronospora_risk' in df_original.columns:
        selected_risk_range = st.sidebar.slider(
            "Filter by Highest Risk (Oidium or Peronospora)",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.05
        )
    else:
        selected_risk_range = (0.0, 1.0)
        st.sidebar.warning("Risk columns not found for filtering.")

    # --- Applying Filters ---
    df_filtered = df_original.copy()

    if selected_vineyards:
        df_filtered = df_filtered[df_filtered['NomeVigneto'].isin(selected_vineyards)]

    if len(selected_date_range) == 2:
        start_date = pd.to_datetime(selected_date_range[0])
        end_date = pd.to_datetime(selected_date_range[1])
        df_filtered = df_filtered[
            (df_filtered['PredictionDate'] >= start_date) &
            (df_filtered['PredictionDate'] <= end_date)
            ]

    if 'Oidium_risk' in df_filtered.columns and 'Peronospora_risk' in df_filtered.columns:
        # Filter rows where at least one disease has risk within the selected range
        df_filtered = df_filtered[
            ((df_filtered['Oidium_risk'] >= selected_risk_range[0]) &
             (df_filtered['Oidium_risk'] <= selected_risk_range[1])) |
            ((df_filtered['Peronospora_risk'] >= selected_risk_range[0]) &
             (df_filtered['Peronospora_risk'] <= selected_risk_range[1]))
            ]

    # --- Displaying the Data ---
    st.header("Prediction Results")
    st.write(f"Displaying {len(df_filtered)} of {len(df_original)} total predictions.")

    # Format risk columns as percentages for display
    df_display = df_filtered.copy()
    if 'Oidium_risk' in df_display.columns:
        df_display['Oidium_risk'] = (df_display['Oidium_risk'] * 100).round(1).astype(str) + '%'
    if 'Peronospora_risk' in df_display.columns:
        df_display['Peronospora_risk'] = (df_display['Peronospora_risk'] * 100).round(1).astype(str) + '%'

    st.dataframe(df_display.sort_values(by="Vigneto_IdVigneto"), use_container_width=True)

    if st.checkbox("Show raw data for filtered results"):
        st.write(df_filtered)

    # --- Summary Statistics ---
    st.header("Summary Statistics")
    col1, col2 = st.columns(2)

    with col1:
        if 'Oidium_risk' in df_filtered.columns:
            st.metric("Average Oidium Risk", f"{df_filtered['Oidium_risk'].mean():.1%}")
            st.metric("Max Oidium Risk", f"{df_filtered['Oidium_risk'].max():.1%}")

    with col2:
        if 'Peronospora_risk' in df_filtered.columns:
            st.metric("Average Peronospora Risk", f"{df_filtered['Peronospora_risk'].mean():.1%}")
            st.metric("Max Peronospora Risk", f"{df_filtered['Peronospora_risk'].max():.1%}")