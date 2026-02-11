import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path

# 1. Page Configuration
st.set_page_config(page_title="Patent Analysis Dashboard", layout="wide")

# 2. Load Data
@st.cache_data
def load_data():
    # Using your specific directory path
    DATA_PATH = Path(__file__).parent / "data" / "Combined_Cleaned_Patents.csv"
    df = pd.read_csv(DATA_PATH)
    df['filing_date'] = pd.to_datetime(df['filing_date'], errors='coerce')
    # Fill empty why_scored with 'N/A' to avoid errors during split
    df['why_scored'] = df['why_scored'].fillna('Uncategorized')
    return df

df = load_data()

# 3. Sidebar Navigation
st.sidebar.title("Navigation")
view_option = st.sidebar.radio(
    "Analyze Patents By:",
    options=["Assignee", "Inventors", "Filing Date", "Score", "Keywords (Why Scored)"]
)

st.title(f"Analysis: Patents by {view_option}")

# 4. Logic for Toggling Views
if view_option == "Assignee":
    st.subheader("Top Patent Holders (Assignees)")
    assignee_counts = df['assignees'].value_counts().reset_index().head(15)
    fig = px.bar(assignee_counts, x='assignees', y='count', color='count', color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)

elif view_option == "Inventors":
    st.subheader("Most Prolific Inventors")
    inventor_series = df['inventors'].str.split(',').explode().str.strip()
    inventor_counts = inventor_series.value_counts().reset_index().head(15)
    fig = px.bar(inventor_counts, x='inventors', y='count', color_discrete_sequence=['#00CC96'])
    st.plotly_chart(fig, use_container_width=True)

elif view_option == "Filing Date":
    st.subheader("Patent Filing Trends")
    df['Year'] = df['filing_date'].dt.year
    timeline = df.groupby('Year').size().reset_index(name='count')
    fig = px.line(timeline, x='Year', y='count', markers=True)
    st.plotly_chart(fig, use_container_width=True)

elif view_option == "Score":
    st.subheader("Distribution of Patent Scores")
    fig = px.histogram(df, x='score', nbins=10, color_discrete_sequence=['#AB63FA'], marginal="box")
    st.plotly_chart(fig, use_container_width=True)

elif view_option == "Keywords (Why Scored)":
    st.subheader("Keyword Frequency & Overlap Analysis")
    
    # A. Extract individual keywords
    # Split the string by comma and remove spaces
    df_kw = df.copy()
    df_kw['kw_list'] = df_kw['why_scored'].str.split(',').apply(lambda x: [i.strip() for i in x] if isinstance(x, list) else [])
    
    # Frequency Chart
    exploded_kw = df_kw.explode('kw_list')
    kw_counts = exploded_kw['kw_list'].value_counts().reset_index()
    kw_counts.columns = ['Keyword', 'Count']
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Frequency of Individual Keywords**")
        fig_bar = px.bar(kw_counts, x='Keyword', y='Count', color='Count', color_continuous_scale='Reds')
        st.plotly_chart(fig_bar, use_container_width=True)

    # B. Overlap Matrix (Heatmap)
    with col2:
        st.write("**Keyword Overlap (Co-occurrence)**")
        unique_kws = sorted([k for k in exploded_kw['kw_list'].unique() if k != 'Uncategorized'])
        
        # Build an empty matrix
        matrix = pd.DataFrame(0, index=unique_kws, columns=unique_kws)
        
        # Fill matrix: for every patent, find pairs of keywords
        for kw_list in df_kw['kw_list']:
            valid_kws = [k for k in kw_list if k in unique_kws]
            for i in valid_kws:
                for j in valid_kws:
                    matrix.loc[i, j] += 1
        
        fig_heat = px.imshow(matrix, text_auto=True, color_continuous_scale='Blues',
                             labels=dict(x="Keyword", y="Keyword", color="Overlap Count"))
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("The diagonal shows the total count for that keyword. Off-diagonal numbers show how many patents share both keywords.")

# 5. Data Table
with st.expander("View Filtered Data Table"):

    st.write(df)
