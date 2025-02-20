import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re

# Set page configuration
st.set_page_config(
    page_title="SpandanDashPS",
    page_icon='ndim.png',
    layout="wide"
)
# Define a password
PASSWORD = os.getenv("PASS_WORD ")


# Initialize session state for authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
def login():
    """Check user password and authenticate."""
    if st.session_state.password == PASSWORD:
        st.session_state.authenticated = True
        del st.session_state.password  # Remove password from session state for security

def logout():
    """Log out user by resetting session state."""
    st.session_state.authenticated = False

def load_data(folder_path):
    """Load all CSV files from the specified folder structure."""
    data_dict = {}
    years = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    
    for year in years:
        year_path = os.path.join(folder_path, year)
        data_dict[year] = {}
        
        for file in os.listdir(year_path):
            if file.endswith('.csv'):
                event_name = file.split('.')[0]  # Remove .csv extension
                file_path = os.path.join(year_path, file)
                try:
                    df = pd.read_csv(file_path)
                    data_dict[year][event_name] = df
                except Exception as e:
                    st.error(f"Error loading {file}: {e}")
    
    return data_dict

def clean_data(df):
    """Clean and preprocess the dataframe."""
    # Copy to avoid modifying original
    cleaned_df = df.copy()
    
    # Convert registration time to datetime
    if 'Registration Time' in cleaned_df.columns:
        cleaned_df['Registration Time'] = pd.to_datetime(cleaned_df['Registration Time'])
        cleaned_df['Registration Date'] = cleaned_df['Registration Time'].dt.date
    
    # Handle missing values
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == 'object':
            cleaned_df[col] = cleaned_df[col].fillna('Not Specified')
    
    return cleaned_df

def generate_kpis(df):
    """Generate KPIs based on the dataframe."""
    cols = st.columns(4)  # Changed from 5 to 4 columns
    
    # KPI 1: Total Registrations
    cols[0].metric("Total Registrations", len(df))
    
    # KPI 2: Total Teams
    if 'Team ID' in df.columns:
        cols[1].metric("Total Teams", df['Team ID'].nunique())
        
    # KPI 3: Gender Distribution
    if 'Gender' in df.columns:
        gender_counts = df['Gender'].value_counts()
        male_count = gender_counts.get('M', 0)
        female_count = gender_counts.get('F', 0)
        cols[2].metric("Male Participants", male_count)
        cols[3].metric("Female Participants", female_count)
    
    # Country KPI removed as requested

def plot_visualizations(df):
    """Generate various visualizations based on the dataframe."""
    row1_cols = st.columns(2)
    
    # Plot 1: Registration Trend
    if 'Registration Time' in df.columns:
        df_trend = df.copy()
        df_trend['Reg Date'] = pd.to_datetime(df_trend['Registration Time']).dt.date
        reg_trend = df_trend.groupby('Reg Date').size().reset_index(name='Count')
        
        fig = px.line(reg_trend, x='Reg Date', y='Count', 
                      title='Daily Registration Trend',
                      labels={'Count': 'Number of Registrations', 'Reg Date': 'Date'})
        row1_cols[0].plotly_chart(fig, use_container_width=True)
    
    # Plot 2: User Type Distribution
    if 'User Type' in df.columns:
        user_type_counts = df['User Type'].value_counts().reset_index()
        user_type_counts.columns = ['User Type', 'Count']
        
        fig = px.pie(user_type_counts, values='Count', names='User Type', 
                     title='User Type Distribution',
                     hole=0.4)
        row1_cols[1].plotly_chart(fig, use_container_width=True)
    
    row2_cols = st.columns(2)
    
    # Plot 3: Domain Distribution
    if 'Domain' in df.columns:
        domain_counts = df['Domain'].value_counts().reset_index()
        domain_counts.columns = ['Domain', 'Count']
        
        fig = px.bar(domain_counts, x='Domain', y='Count', 
                     title='Domain Distribution',
                     labels={'Count': 'Number of Participants', 'Domain': 'Domain'})
        row2_cols[0].plotly_chart(fig, use_container_width=True)
    
    # Plot 4: Country Distribution
    if 'Country' in df.columns:
        country_counts = df['Country'].value_counts().reset_index()
        country_counts.columns = ['Country', 'Count']
        
        fig = px.bar(country_counts.head(10), x='Country', y='Count', 
                     title='Top 10 Countries by Participation',
                     labels={'Count': 'Number of Participants', 'Country': 'Country'})
        row2_cols[1].plotly_chart(fig, use_container_width=True)
    
    # Additional plots if relevant columns exist
    if 'Course Stream' in df.columns:
        st.subheader("Education Background Analysis")
        row3_cols = st.columns(2)
        
        # Course stream distribution
        stream_counts = df['Course Stream'].value_counts().reset_index()
        stream_counts.columns = ['Course Stream', 'Count']
        
        fig = px.bar(stream_counts.head(10), x='Course Stream', y='Count', 
                     title='Top 10 Course Streams',
                     labels={'Count': 'Number of Participants', 'Course Stream': 'Stream'})
        row3_cols[0].plotly_chart(fig, use_container_width=True)
        
        # Year of study distribution if available
        if 'Year Of Study' in df.columns:
            yos_counts = df['Year Of Study'].value_counts().reset_index()
            yos_counts.columns = ['Year Of Study', 'Count']
            
            fig = px.pie(yos_counts, values='Count', names='Year Of Study', 
                         title='Year of Study Distribution')
            row3_cols[1].plotly_chart(fig, use_container_width=True)

def calculate_advanced_metrics(df):
    """Calculate and display advanced metrics."""
    st.subheader("Advanced Metrics")
    
    metrics_cols = st.columns(3)
    
    # Registration completion rate
    if 'Reg. Status' in df.columns:
        complete_reg = df[df['Reg. Status'] == 'Complete'].shape[0]
        completion_rate = (complete_reg / df.shape[0]) * 100
        metrics_cols[0].metric("Registration Completion Rate", f"{completion_rate:.1f}%")
    
    # Team size distribution if team data available
    if 'Team ID' in df.columns:
        team_sizes = df.groupby('Team ID').size().reset_index(name='Team Size')
        avg_team_size = team_sizes['Team Size'].mean()
        metrics_cols[1].metric("Average Team Size", f"{avg_team_size:.2f}")
    
    # Work experience percentage
    if 'Work Experience' in df.columns:
        exp_count = df[df['Work Experience'].notna() & (df['Work Experience'] != '')].shape[0]
        exp_percentage = (exp_count / df.shape[0]) * 100
        metrics_cols[2].metric("Participants with Work Experience", f"{exp_percentage:.1f}%")

def combine_event_data(data_dict, selected_year):
    """Combine data from all events for a selected year."""
    all_events_df = pd.DataFrame()
    
    for event_name, event_df in data_dict[selected_year].items():
        # Add event name column
        event_df_copy = event_df.copy()
        event_df_copy['Event Name'] = event_name
        all_events_df = pd.concat([all_events_df, event_df_copy])
    
    return all_events_df

def main():
    st.title("📈Spandan Registration Dashboard[Unstop]")
    st.markdown("#### Developed by [💻Pranay Sarkar](https://www.linkedin.com/in/pranay-sarkar/)")
    
    # Sidebar for navigation and filtering
    st.sidebar.header("Data Settings")
    
    # Folder path input
    folder_path = "unstopReg"
    
    
    if not os.path.isdir(folder_path):
        st.sidebar.error(f"Directory not found: {folder_path}")
        st.info("Please enter a valid directory path in the sidebar.")
        
        # Show sample data for demo
        st.subheader("Sample Data Preview")
        sample_df = pd.DataFrame({
            'Team ID': ['U14R8B9Z'],
            'Team Name': ['pulkit.pulkit.trehan'],
            'User Type': ['Leader'],
            'Name': ['PULKIT TREHAN'],
            'Email': ['pulkit.pulkit.trehan@gmail.com'],
            'Country': ['India'],
            'Registration Time': ['2024-02-14T17:45:59.000Z']
        })
        st.dataframe(sample_df)
        
        return
    
    # Load data
    try:
        data_dict = load_data(folder_path)
        if not data_dict:
            st.error("No data found in the specified directory.")
            return
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    # Year selection
    years = list(data_dict.keys())
    selected_year = st.sidebar.selectbox("Select Year", years)
    
    # Event selection with "All Events" option
    events = ["All Events"] + list(data_dict[selected_year].keys())
    selected_event = st.sidebar.selectbox("Select Event", events, index=0)  # Default to "All Events"
    
    # Get and clean selected data
    if selected_event == "All Events":
        df = combine_event_data(data_dict, selected_year)
        display_title = f"All Events ({selected_year})"
    else:
        df = data_dict[selected_year][selected_event]
        display_title = f"{selected_event} ({selected_year})"
    
    cleaned_df = clean_data(df)
    
    # Display section
    st.header(display_title)
    
    # KPI Cards
    st.subheader("Key Performance Indicators")
    generate_kpis(cleaned_df)
    
    # Data filtering options
    st.sidebar.header("Data Filters")
    
    # Add Event filter when "All Events" is selected
    filtered_df = cleaned_df.copy()
    
    if selected_event == "All Events" and 'Event Name' in filtered_df.columns:
        selected_events_filter = st.sidebar.multiselect(
            "Filter by Event",
            options=filtered_df['Event Name'].unique(),
            default=[]
        )
        if selected_events_filter:
            filtered_df = filtered_df[filtered_df['Event Name'].isin(selected_events_filter)]
    
    # Dynamic filters based on available columns
    filter_cols = st.sidebar.multiselect(
        "Select columns to filter by",
        options=[col for col in filtered_df.columns if filtered_df[col].nunique() < 20 and col != 'Event Name'],
        default=[]
    )
    
    for col in filter_cols:
        unique_values = filtered_df[col].unique()
        selected_values = st.sidebar.multiselect(
            f"Filter by {col}",
            options=unique_values,
            default=unique_values
        )
        if selected_values:
            filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
    
    # Show KPIs for filtered data if filters applied
    if len(filter_cols) > 0 or (selected_event == "All Events" and selected_events_filter):
        st.subheader("Filtered Data KPIs")
        generate_kpis(filtered_df)
    
    # Visualizations
    st.header("Visualizations")
    
    # Add event comparison chart if "All Events" is selected
    if selected_event == "All Events" and 'Event Name' in filtered_df.columns:
        st.subheader("Event Comparison")
        event_counts = filtered_df['Event Name'].value_counts().reset_index()
        event_counts.columns = ['Event', 'Registrations']
        
        fig = px.bar(event_counts, x='Event', y='Registrations',
                     title=f'Registration Count by Event ({selected_year})',
                     labels={'Registrations': 'Number of Registrations', 'Event': 'Event Name'})
        st.plotly_chart(fig, use_container_width=True)
    
    plot_visualizations(filtered_df)
    
    # Advanced metrics
    calculate_advanced_metrics(filtered_df)
    
    # Data table view
    st.header("Data Explorer")
    
    # Column selector
    all_columns = filtered_df.columns.tolist()
    selected_columns = st.multiselect(
        "Select columns to display",
        options=all_columns,
        default=all_columns[:10]  # Show first 10 columns by default
    )
    
    # Search functionality
    search_term = st.text_input("Search in data", "")
    
    # Apply search if provided
    if search_term:
        search_mask = pd.Series(False, index=filtered_df.index)
        for col in filtered_df.columns:
            if filtered_df[col].dtype == 'object':
                search_mask |= filtered_df[col].astype(str).str.contains(search_term, case=False, na=False)
        display_df = filtered_df[search_mask]
    else:
        display_df = filtered_df
    
    # Display data
    if selected_columns:
        st.dataframe(display_df[selected_columns], use_container_width=True)
    else:
        st.dataframe(display_df, use_container_width=True)
    
    # Export options
    st.download_button(
        label="Download Filtered Data as CSV",
        data=display_df.to_csv(index=False).encode('utf-8'),
        file_name=f"{selected_year}_{selected_event}_filtered.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    if not st.session_state.authenticated:
        st.title("Login")
        st.text_input("Enter Password", type="password", key="password", on_change=login)
        st.stop()
    main()
