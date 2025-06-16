# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import plotly.express as px

# Optional: To suppress warnings for cleaner output
import warnings

from Tools.scripts.dutree import display

warnings.filterwarnings('ignore')

# Set aesthetic for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Load the dataset
# Assuming the dataset is named 'NTSB_Accident_Data.csv' and is in a 'data' folder
# Adjust the file_path variable if your file is named differently or located elsewhere
file_path = "./AviationData.csv"

try:
    df = pd.read_csv(file_path, encoding="ISO-8859-1", low_memory=False)
    print(f"Dataset loaded successfully from: {file_path}")
    print(f"Initial dataset shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}. Please check the path and filename.")
    print("Make sure your notebook is in the correct directory, or provide the full path to the file.")
    df = None  # Set df to None to avoid errors in subsequent operations if loading fails

# 2. Initial Data Inspection

if df is not None:
    print("\n--- First 5 Rows of the Dataset ---")
    print(df.head())

    print("\n--- Dataset Information (Columns, Non-Null Counts, Data Types) ---")
    df.info()

    print("\n--- Basic Statistical Summary of Numerical Columns ---")
    print(df.describe())

    print("\n--- Number of Unique Values per Column ---")
    print(df.nunique())

    print("\n--- Check for Duplicated Rows ---")
    print(f"Number of duplicated rows: {df.duplicated().sum()}")
else:
    print("Data not loaded, skipping initial inspection.")

# 3. Data Cleaning and Preprocessing

if df is not None:
    print("--- Cleaning 'Event Date' column ---")
    # Convert 'Event Date' to datetime objects, coercing errors to NaT
    df['Event.Date'] = pd.to_datetime(df['Event.Date'], errors='coerce')

    # Check for any NaT (Not a Time) values after conversion
    print(f"Number of unparseable dates (NaT values) found: {df['Event.Date'].isna().sum()}")

    # Extract Year for easier time-series analysis later
    df['Year'] = df['Event.Date'].dt.year

    # Drop rows where 'Event Date' (and thus 'Year') couldn't be parsed, as they are essential
    df.dropna(subset=['Event.Date'], inplace=True)
    print(f"Dataset shape after dropping rows with missing or invalid 'Event Date': {df.shape}")
else:
    print("Data not loaded, skipping date cleaning.")

# 3.2. Standardizing Aircraft Information

if df is not None:
    print("\n--- Cleaning 'Aircraft Make' and 'Aircraft Model' columns ---")
    for col in ['Make', 'Model']:
        if col in df.columns:
            # Convert to string, then to uppercase, strip leading/trailing whitespace
            df[col] = df[col].astype(str).str.upper().str.strip()
            # Replace multiple spaces with a single space
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            # Handle 'NAN' or 'NONE' strings that might result from missing values being converted to str
            df[col] = df[col].replace({'NAN': None, 'NONE': None, 'UNKNOWN': None})
        else:
            print(f"Warning: Column '{col}' not found in the dataset.")

    # Check for missing values in critical aircraft identification columns after initial cleaning
    print("\nMissing values in key aircraft columns after initial text standardization:")
    print(df[['Make', 'Model']].isnull().sum())

    # Drop rows where 'Aircraft Make' or 'Aircraft Model' are still missing, as they are crucial for analysis
    original_shape = df.shape[0]
    df.dropna(subset=['Make', 'Model'], inplace=True)
    print(f"Dropped {original_shape - df.shape[0]} rows due to missing Aircraft Make/Model.")
    print(f"Dataset shape after dropping rows with missing Make/Model: {df.shape}")
else:
    print("Data not loaded, skipping aircraft info cleaning.")

# 3.3. Cleaning Numerical Outcome Variables (Fatalities, Injuries)

if df is not None:
    print("\n--- Cleaning numerical outcome columns ---")
    # Columns to convert to numeric, coercing errors to NaN
    numeric_outcome_cols = [
        'Total.Fatal.Injuries', 'Total.Serious.Injuries',
        'Total.Minor.Injuries', 'Total.Uninjured'
    ]

    for col in numeric_outcome_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Fill NaN values with 0, assuming missing means 'no injuries/fatalities'
            df[col].fillna(0, inplace=True)
            # Convert to integer type after filling NaNs
            df[col] = df[col].astype(int)
        else:
            print(f"Warning: Column '{col}' not found in the dataset, skipping conversion.")

    # Create a 'Total Injuries' column by summing serious and minor injuries
    if 'Total.Serious.Injuries' in df.columns and 'Total.Minor.Injuries' in df.columns:
        df['Total.Injuries'] = df['Total.Serious.Injuries'] + df['Total.Minor.Injuries']
        print(f"Created 'Total Injuries' column.")
    else:
        # Fallback if specific injury columns are missing, sum all available numeric outcome cols
        df['Total.Injuries'] = df[[col for col in numeric_outcome_cols if col in df.columns]].sum(axis=1)
        print(f"Warning: Specific injury columns not found, 'Total Injuries' summed from available outcome cols.")

    # Rename 'Total Fatal Injuries' to 'Fatalities' for simplicity and consistency
    if 'Total.Fatal.Injuries' in df.columns:
        df.rename(columns={'Total.Fatal.Injuries': 'Fatalities'}, inplace=True)
        print(f"Renamed 'Total Fatal Injuries' to 'Fatalities'.")
    else:
        print("Warning: 'Total Fatal Injuries' column not found for renaming.")

    print("\nMissing values in key numerical outcome columns after cleaning (should be 0):")
    print(df[['Fatalities', 'Total.Injuries']].isnull().sum())
else:
    print("Data not loaded, skipping numerical outcome cleaning.")

# 3.4. Inspecting and Cleaning Damage Information (Aircraft Damage)

if df is not None:
    if 'Aircraft.damage' in df.columns:
        print("\nUnique values in 'Aircraft Damage' before final cleaning:")
        print(df['Aircraft.damage'].value_counts(dropna=False))

        # Standardize values to uppercase and replace common representations of missing/unknown
        df['Aircraft.damage'] = df['Aircraft.damage'].astype(str).str.upper().str.strip()
        df['Aircraft.damage'] = df['Aircraft.damage'].replace({'NAN': None, 'UNKN': None,
                                                               'UNKNOWN': None, '(NONE)': None})

        # Fill any remaining NaNs for 'Aircraft Damage' with 'UNKNOWN'
        df['Aircraft.damage'].fillna('UNKNOWN', inplace=True)

        print("\nUnique values in 'Aircraft Damage' after final cleaning:")
        print(df['Aircraft.damage'].value_counts(dropna=False))
    else:
        print("Warning: 'Aircraft Damage' column not found in the dataset, skipping cleaning.")
else:
    print("Data not loaded, skipping damage info cleaning.")

# 3.5. Final Data Check After Cleaning

if df is not None:
    print("\n--- Dataset Info After All Cleaning Steps ---")
    df.info()

    print("\n--- First 5 Rows After All Cleaning Steps ---")
    print(df.head())
else:
    print("Data not loaded, skipping final check.")

# 4. Feature Engineering and Risk Metric Definition

# 4.1. Defining Risk Metrics & Creating Features
if df is not None:
    print("--- Creating Feature Engineering for Risk Metrics ---")

    # Create a binary column: 1 if an incident had any fatalities, 0 otherwise
    df['Incident_Has_Fatalities'] = df['Fatalities'].apply(lambda x: 1 if x > 0 else 0)
    print("Created 'Incident_Has_Fatalities' column.")

    # Create a binary column: 1 if an incident resulted in 'DESTROYED' or 'SUBSTANTIAL' damage, 0 otherwise
    if 'Aircraft.damage' in df.columns:
        df['Incident_Has_High_Damage'] = df['Aircraft.damage'].apply(
            lambda x: 1 if x in ['DESTROYED', 'SUBSTANTIAL'] else 0
        )
        print("Created 'Incident_Has_High_Damage' column.")
    else:
        print("Warning: 'Aircraft Damage' column not found, 'Incident_Has_High_Damage' not created.")
        df['Incident_Has_High_Damage'] = 0  # Default to 0 if the source column is missing
else:
    print("Data not loaded, skipping feature engineering.")

# 5. Aggregating Data by Aircraft Type

if df is not None:
    print("\n--- Aggregating Data by Aircraft Make for Risk Summary ---")

    # Group by 'Aircraft Make' and calculate the desired risk metrics
    aircraft_risk_summary = df.groupby('Make').agg(
        Total_Incidents=('Event.Id', 'count'),  # Count of all incidents for the make
        Total_Fatalities=('Fatalities', 'sum'),  # Sum of all fatalities for the make
        Total_Injuries=('Total.Injuries', 'sum'),  # Sum of all injuries for the make
        Incidents_With_Fatalities=('Incident_Has_Fatalities', 'sum'),  # Count of incidents with at least one fatality
        Incidents_With_High_Damage=('Incident_Has_High_Damage', 'sum')
        # Count of incidents with destroyed/substantial damage
    ).reset_index()

    # Calculate the Fatality Rate per incident
    # This represents the percentage of incidents that resulted in one or more fatalities
    aircraft_risk_summary['Fatality_Rate_Per_Incident'] = (
            (aircraft_risk_summary['Incidents_With_Fatalities'] / aircraft_risk_summary['Total_Incidents']) * 100
    ).fillna(0)  # Fill NaN with 0 for aircraft makes with no incidents (though filtered later)

    # Calculate the High Damage Rate per incident
    # This represents the percentage of incidents that resulted in substantial or destroyed damage
    aircraft_risk_summary['High_Damage_Rate_Per_Incident'] = (
            (aircraft_risk_summary['Incidents_With_High_Damage'] / aircraft_risk_summary['Total_Incidents']) * 100
    ).fillna(0)  # Fill NaN with 0 for aircraft makes with no incidents

    # Filter out aircraft makes with very few incidents, as their rates might be statistically unreliable
    # A threshold of 20 incidents ensures a more robust rate calculation
    min_incidents_threshold = 20
    aircraft_risk_summary_filtered = aircraft_risk_summary[
        aircraft_risk_summary['Total_Incidents'] >= min_incidents_threshold
        ].copy()  # Use .copy() to avoid SettingWithCopyWarning

    print(f"\n--- Aircraft Risk Summary (Filtered for Makes with >= {min_incidents_threshold} Incidents) ---")
    print(f"Number of aircraft makes after filtering: {aircraft_risk_summary_filtered.shape[0]}")

    print("\n--- Top 10 Aircraft Makes by Total Incidents ---")
    print(aircraft_risk_summary_filtered.sort_values(by='Total_Incidents', ascending=False).head(10))

    print("\n--- Top 10 Aircraft Makes by Lowest Fatality Rate per Incident ---")
    # Sort by fatality rate ascending, then by total incidents descending (to prefer more frequently used low-risk aircraft)
    print(aircraft_risk_summary_filtered.sort_values(
        by=['Fatality_Rate_Per_Incident', 'Total_Incidents'], ascending=[True, False]
    ).head(10))

    print("\n--- Top 10 Aircraft Makes by Lowest High Damage Rate per Incident ---")
    # Sort by high damage rate ascending, then by total incidents descending
    print(aircraft_risk_summary_filtered.sort_values(
        by=['High_Damage_Rate_Per_Incident', 'Total_Incidents'], ascending=[True, False]
    ).head(10))
else:
    print("Data not loaded, skipping aggregation.")

# 6. Key Visualizations





# 6.1. Top Aircraft Makes by Total Incidents

if df is not None and 'aircraft_risk_summary_filtered' in locals():
    print("\n--- Visualizing Top Aircraft Makes by Total Incidents (Matplotlib/Seaborn) ---")
    top_incident_makes = aircraft_risk_summary_filtered.sort_values(
        by='Total_Incidents', ascending=False
    ).head(15)

    plt.figure(figsize=(14, 8)) # Adjust figure size for better readability
    sns.barplot(
        x='Make',
        y='Total_Incidents',
        data=top_incident_makes,
        palette='viridis'  # Using a different color palette for variety
    )
    plt.title('Top 15 Aircraft Makes by Total Incidents (1962-2023)')
    plt.xlabel('Aircraft Manufacturer')
    plt.ylabel('Number of Incidents')
    plt.xticks(rotation=45, ha='right') # Rotate labels for better fit
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.show()
else:
    print("Skipping visualization: Data not loaded or risk summary not generated.")



#6.2. Top Aircraft Makes by Lowest Fatality Rate Per Incident

if df is not None and 'aircraft_risk_summary_filtered' in locals():
    print("\n--- Visualizing Top Aircraft Makes with Lowest Fatality Rate Per Incident (Matplotlib/Seaborn) ---")
    # Sort by Fatality_Rate_Per_Incident (ascending), then by Total_Incidents (descending for tie-breaking)
    lowest_fatality_rate_makes = aircraft_risk_summary_filtered.sort_values(
        by=['Fatality_Rate_Per_Incident', 'Total_Incidents'], ascending=[True, False]
    ).head(15)

    plt.figure(figsize=(14, 8))
    sns.barplot(
        x='Make',
        y='Fatality_Rate_Per_Incident',
        data=lowest_fatality_rate_makes,
        palette='magma_r'  # Using a reverse color palette to highlight lower rates
    )
    plt.title(f'Top 15 Aircraft Makes with Lowest Fatality Rate Per Incident (Min. {min_incidents_threshold} Incidents)')
    plt.xlabel('Aircraft Manufacturer')
    plt.ylabel('Fatality Rate (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    # Note: Matplotlib/Seaborn plots are static. If you need interactive HTML, Plotly is recommended.
    # We will not save a static image here as the request implied updating the interactive section.
else:
    print("Skipping visualization: Data not loaded or risk summary not generated.")




# # 6.3. Top Aircraft Makes by Lowest High Damage Rate Per Incident
if df is not None and 'aircraft_risk_summary_filtered' in locals():
    print("\n--- Visualizing Top Aircraft Makes with Lowest High Damage Rate Per Incident (Matplotlib/Seaborn) ---")
    # Sort by High_Damage_Rate_Per_Incident (ascending), then by Total_Incidents (descending)
    lowest_damage_rate_makes = aircraft_risk_summary_filtered.sort_values(
        by=['High_Damage_Rate_Per_Incident', 'Total_Incidents'], ascending=[True, False]
    ).head(15)

    plt.figure(figsize=(14, 8))
    sns.barplot(
        x='Make',
        y='High_Damage_Rate_Per_Incident',
        data=lowest_damage_rate_makes,
        palette='plasma_r'  # Using a reverse color palette to highlight lower rates
    )
    plt.title(f'Top 15 Aircraft Makes with Lowest High Damage Rate Per Incident (Min. {min_incidents_threshold} Incidents)')
    plt.xlabel('Aircraft Manufacturer')
    plt.ylabel('High Damage Rate (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    # Note: Matplotlib/Seaborn plots are static. For interactive HTML plots, Plotly remains the best choice.
else:
    print("Skipping visualization: Data not loaded or risk summary not generated.")


# #6.4. Trend of Incidents and Fatalities Over Time
if df is not None:
    print("\n--- Visualizing Trend of Aviation Incidents and Fatalities Over Time (Matplotlib/Seaborn) ---")
    # Aggregate by Year for overall trends
    yearly_summary = df.groupby('Year').agg(
        Total_Incidents=('Event.Id', 'count'),
        Total_Fatalities=('Fatalities', 'sum')
    ).reset_index()

    # Filter out very early or very late years if data is sparse or incomplete at the edges
    # Assuming full data from 1965 to 2022 for clean trends
    yearly_summary = yearly_summary[
        (yearly_summary['Year'] >= df['Year'].min() + 3) &
        (yearly_summary['Year'] <= df['Year'].max() - 1)
    ]

    plt.figure(figsize=(14, 8))
    sns.lineplot(x='Year', y='Total_Incidents', data=yearly_summary, marker='o', label='Total Incidents', color='blue')
    sns.lineplot(x='Year', y='Total_Fatalities', data=yearly_summary, marker='o', label='Total Fatalities', color='red')

    plt.title('Trend of Aviation Incidents and Fatalities Over Time (1962-2023)')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
else:
    print("Skipping visualization: Data not loaded.")


if 'aircraft_risk_summary_filtered' in locals():
    # Define the output file path for the CSV
    output_csv_path = 'C:/Users/pelicier/OneDrive/Bureau/DataBase and AI Devoir/Phase-1-project/aviation-risk-analysis/data/aircraft_risk_summary.csv'

    # Export the filtered summary DataFrame to a CSV file
    aircraft_risk_summary_filtered.to_csv(output_csv_path, index=False)
    print(f"\nFiltered aircraft risk summary exported to: {output_csv_path}")
else:
    print("Error: 'aircraft_risk_summary_filtered' DataFrame not found. Please run previous cells.")

