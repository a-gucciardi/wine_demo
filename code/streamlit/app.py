import streamlit as st
import pandas as pd
import requests
from io import StringIO
from datetime import datetime
import numpy as np

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
    Returns a pandas DataFrame with the new format.
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

        # Clean the percentage values in date columns (remove % and convert to float)
        for col in date_cols:
            df[col] = df[col].astype(str).str.replace('%', '').astype(float) / 100

        # Add date columns as datetime for filtering
        df['date_columns'] = [date_cols] * len(df)

        return df

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
    # Get date columns for filtering and display
    metadata_cols = ['Ente', 'IdVigneto', 'Codice', 'NomeVigneto', 'Infection']
    date_cols = [col for col in df_original.columns if col not in metadata_cols and col != 'date_columns']

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

    # Filter by Infection Type
    if 'Infection' in df_original.columns:
        infection_types = df_original['Infection'].unique()
        selected_infections = st.sidebar.multiselect(
            "Select Infection Type(s)",
            options=sorted(infection_types),
            default=[]
        )
    else:
        selected_infections = []
        st.sidebar.warning("'Infection' column not found for filtering.")

    # Filter by Date Range
    if date_cols:
        date_objects = [pd.to_datetime(col).date() for col in date_cols]
        min_date = min(date_objects)
        max_date = max(date_objects)
        selected_date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    else:
        selected_date_range = ()

    # Filter by Risk Score (slider)
    if date_cols:
        # Calculate max risk across all date columns for filtering
        max_risk_overall = df_original[date_cols].max().max()
        selected_risk_range = st.sidebar.slider(
            "Filter by Maximum Risk (any day)",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.05
        )
    else:
        selected_risk_range = (0.0, 1.0)
        st.sidebar.warning("Date columns not found for risk filtering.")

    # --- Applying Filters ---
    df_filtered = df_original.copy()

    if selected_vineyards:
        df_filtered = df_filtered[df_filtered['NomeVigneto'].isin(selected_vineyards)]

    if selected_infections:
        df_filtered = df_filtered[df_filtered['Infection'].isin(selected_infections)]

    # Filter by date range - only show columns within selected range
    if len(selected_date_range) == 2 and date_cols:
        start_date = pd.to_datetime(selected_date_range[0])
        end_date = pd.to_datetime(selected_date_range[1])

        # Filter date columns to only show those within range
        filtered_date_cols = [col for col in date_cols
                              if start_date <= pd.to_datetime(col) <= end_date]
        display_cols = metadata_cols + filtered_date_cols
    else:
        display_cols = df_filtered.columns.tolist()
        if 'date_columns' in display_cols:
            display_cols.remove('date_columns')
        filtered_date_cols = date_cols

    # Filter by risk score - keep rows where max risk across selected dates is within range
    if filtered_date_cols:
        df_filtered['max_risk'] = df_filtered[filtered_date_cols].max(axis=1)
        df_filtered = df_filtered[
            (df_filtered['max_risk'] >= selected_risk_range[0]) &
            (df_filtered['max_risk'] <= selected_risk_range[1])
            ]
        df_filtered = df_filtered.drop('max_risk', axis=1)

    # --- Displaying the Data ---
    st.header("Prediction Results")
    st.write(f"Displaying {len(df_filtered)} of {len(df_original)} total predictions.")

    # Format risk columns as percentages for display
    df_display = df_filtered[display_cols].copy()
    for col in filtered_date_cols:
        if col in df_display.columns:
            df_display[col] = (df_display[col] * 100).round(1).astype(str) + '%'

    st.dataframe(df_display.sort_values(by=["IdVigneto", "Infection"]), use_container_width=True)

    if st.checkbox("Show raw data for filtered results"):
        st.write(df_filtered[display_cols])

    # --- Summary Statistics ---
    st.header("Summary Statistics")

    if filtered_date_cols:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("By Infection Type")
            if 'Infection' in df_filtered.columns:
                infection_stats = df_filtered.groupby('Infection')[filtered_date_cols].mean()
                for infection in infection_stats.index:
                    avg_risk = infection_stats.loc[infection].mean()
                    st.metric(f"Average {infection} Risk", f"{avg_risk:.1%}")

        with col2:
            st.subheader("By Date")
            daily_avg = df_filtered[filtered_date_cols].mean()
            for date_col in filtered_date_cols[:3]:  # Show first 3 dates
                st.metric(f"Average Risk {date_col}", f"{daily_avg[date_col]:.1%}")

    # --- Risk Trends Chart ---
    if len(filtered_date_cols) > 1:
        st.header("Risk Trends")

        # Create a chart showing average risk by infection type over time
        chart_data = df_filtered.groupby('Infection')[filtered_date_cols].mean().T
        chart_data.index = pd.to_datetime(chart_data.index)

        st.line_chart(chart_data)