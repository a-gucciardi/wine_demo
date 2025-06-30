import streamlit as st
import pandas as pd
import os

# --- Page Configuration ---
# This must be the first Streamlit command in your script.
st.set_page_config(
    page_title="Infection Risk Predictions",
    page_icon="🍇",
    layout="wide",  # Use the full page width
)

# --- App Title ---
st.title("Vineyard Infection Risk Predictions")


# --- Data Loading ---
# We use a function with caching to prevent reloading the data on every interaction.
@st.cache_data
def load_data(file_path):
    """
    Loads the prediction data from a CSV file.
    Returns a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        # In a real scenario, you'd have an error. For this demo, we create a dummy file.
        st.error(f"Error: The data file was not found at '{file_path}'. A dummy dataset is being shown.")
        dummy_data = {
            'Vigneto_IdVigneto': [101, 102, 101, 103, 102],
            'Codice': ['V01', 'V02', 'V01', 'V03', 'V02'],
            'NomeVigneto': ['Sunrise Field', 'West Valley', 'Sunrise Field', 'Hilltop Plot', 'West Valley'],
            'PredictionDate': ['2025-07-01', '2025-07-01', '2025-07-02', '2025-07-02', '2025-07-03'],
            'Oidium_pred': [1, 0, 1, 0, 1],
            'Oidium_risk': [0.78, 0.32, 0.91, 0.45, 0.65],
            'Peronospora_pred': [0, 1, 0, 1, 1],
            'Peronospora_risk': [0.21, 0.88, 0.15, 0.76, 0.81]
        }
        # FIX: Convert date column for dummy data to ensure consistent data types
        df_dummy = pd.DataFrame(dummy_data)
        df_dummy['PredictionDate'] = pd.to_datetime(df_dummy['PredictionDate'])
        return df_dummy

    try:
        df = pd.read_csv(file_path)
        # Convert date column to datetime objects for better sorting/filtering
        df['PredictionDate'] = pd.to_datetime(df['PredictionDate'])
        return df
    except Exception as e:
        st.error(f"An error occurred while reading the data file: {e}")
        return pd.DataFrame()  # Return empty dataframe on error


# Load the data
CSV_FILE = 'infection_risk_predictions.csv'
df_original = load_data(CSV_FILE)

if not df_original.empty:
    # --- Sidebar for Filtering ---
    st.sidebar.header("Filter Options")

    # Filter by Vineyard Name (multiselect)
    vineyard_names = df_original['NomeVigneto'].unique()
    selected_vineyards = st.sidebar.multiselect(
        "Select Vineyard(s)",
        options=sorted(vineyard_names),
        default=[]  # Empty default means show all
    )

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
    min_risk, max_risk = 0.0, 1.0
    selected_risk_range = st.sidebar.slider(
        "Filter by Highest Risk (Oidium or Peronospora)",
        min_value=min_risk,
        max_value=max_risk,
        value=(min_risk, max_risk),  # Default is the full range
        step=0.05
    )

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

    # Display the filtered dataframe
    # st.dataframe allows for interactive sorting by clicking on headers.
    st.dataframe(df_filtered.sort_values(by="Vigneto_IdVigneto"), use_container_width=True)

    # --- Show Raw Data (optional) ---
    if st.checkbox("Show raw data for filtered results"):
        st.write(df_filtered)

else:
    st.warning("Could not load prediction data.")