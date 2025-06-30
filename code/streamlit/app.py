import streamlit as st
import pandas as pd
import requests
from io import StringIO

# --- Page Configuration ---
st.set_page_config(
    page_title="Infection Risk Predictions Viewer",
    page_icon="🍇",
    layout="wide",
)

# --- App Title ---
st.title("Vineyard Infection Risk Predictions Viewer")
GITHUB_RAW_URL = "https://raw.githubusercontent.com/a-gucciardi/wine_demo/refs/heads/main/code/streamlit/infection_risk_predictions.csv"


# --- Data Loading ---
# This function now loads data directly from the GitHub URL using the robust requests method.
@st.cache_data
def load_data_from_url(url):
    """
    Loads prediction data from a raw GitHub URL using the requests library.
    Returns a pandas DataFrame.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # This will raise an exception for bad status codes (4xx or 5xx)

        # Use StringIO to treat the string data as a file
        csv_file = StringIO(response.text)

        df = pd.read_csv(csv_file)

        # --- Data Validation and Transformation ---
        if 'PredictionDate' in df.columns:
            df['PredictionDate'] = pd.to_datetime(df['PredictionDate'])
            return df
        else:
            st.error("Error: The CSV file from the URL must contain a 'PredictionDate' column.")
            return None

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
    st.write(f"Displaying {len(df_filtered)} of {len(df_original)} total predictions.")
    st.dataframe(df_filtered.sort_values(by="Vigneto_IdVigneto"), use_container_width=True)

    if st.checkbox("Show raw data for filtered results"):
        st.write(df_filtered)
