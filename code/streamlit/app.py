import streamlit as st
import pandas as pd
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Infection Risk Predictions Viewer",
    page_icon="🍇",
    layout="wide",
)

# --- App Title ---
st.title("🍇 Vineyard Infection Risk Predictions Viewer")

# --- File Uploader ---
# This widget allows users to upload a CSV file.
uploaded_file = st.file_uploader(
    "Upload your infection risk predictions CSV file",
    type=["csv"]
)

# --- Main App Logic ---
# The app will only proceed if a file has been successfully uploaded.
if uploaded_file is not None:
    try:
        # Read the uploaded CSV file into a pandas DataFrame
        df_original = pd.read_csv(uploaded_file)

        # --- Data Validation and Transformation ---
        # Ensure the 'PredictionDate' column exists and convert it to datetime
        if 'PredictionDate' in df_original.columns:
            df_original['PredictionDate'] = pd.to_datetime(df_original['PredictionDate'])
        else:
            st.error("Error: The uploaded CSV must contain a 'PredictionDate' column.")
            st.stop()  # Stop execution if the required column is missing

    except Exception as e:
        st.error(f"An error occurred while reading or processing the file: {e}")
        st.stop()

    # --- Sidebar for Filtering ---
    st.sidebar.header("Filter Options")

    # Filter by Vineyard Name (multiselect)
    if 'NomeVigneto' in df_original.columns:
        vineyard_names = df_original['NomeVigneto'].unique()
        selected_vineyards = st.sidebar.multiselect(
            "Select Vineyard(s)",
            options=sorted(vineyard_names),
            default=[]  # Empty default means show all
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
            value=(0.0, 1.0),  # Default is the full range
            step=0.05
        )
    else:
        selected_risk_range = (0.0, 1.0)
        st.sidebar.warning("Risk columns not found for filtering.")

    # --- Applying Filters ---
    df_filtered = df_original.copy()

    # Apply vineyard filter
    if selected_vineyards:
        df_filtered = df_filtered[df_filtered['NomeVigneto'].isin(selected_vineyards)]

    # Apply date range filter
    if len(selected_date_range) == 2:
        start_date = pd.to_datetime(selected_date_range[0])
        end_date = pd.to_datetime(selected_date_range[1])
        df_filtered = df_filtered[
            (df_filtered['PredictionDate'] >= start_date) &
            (df_filtered['PredictionDate'] <= end_date)
            ]

    # Apply risk score filter
    if 'Oidium_risk' in df_filtered.columns and 'Peronospora_risk' in df_filtered.columns:
        df_filtered = df_filtered[
            (df_filtered['Oidium_risk'] >= selected_risk_range[0]) |
            (df_filtered['Peronospora_risk'] >= selected_risk_range[0])
            ]
        df_filtered = df_filtered[
            (df_filtered['Oidium_risk'] <= selected_risk_range[1]) |
            (df_filtered['Peronospora_risk'] <= selected_risk_range[1])
            ]

    # --- Displaying the Data ---
    st.header("Prediction Results")
    st.write(
        f"Displaying {len(df_filtered)} of {len(df_original)} total predictions from uploaded file: `{uploaded_file.name}`")

    # Display the filtered dataframe
    st.dataframe(df_filtered.sort_values(by="Vigneto_IdVigneto"), use_container_width=True)

    # --- Show Raw Data (optional) ---
    if st.checkbox("Show raw data for filtered results"):
        st.write(df_filtered)

else:
    # This message is shown when the app first loads, before any file is uploaded.
    st.info("Please upload a CSV file to view the predictions.")

