# ------------------------------
# Credit Card Fraud Detection App (Interactive, CV-ready, Fixed)
# Using DBSCAN (Unsupervised ML)
# ------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import plotly.express as px
import os
import time

# ------------------------------
# Page Configuration & Custom CSS
# ------------------------------
st.set_page_config(
    page_title="DBSCAN Fraud Detection AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    """Injects custom CSS for a modern UI."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
        
        .stApp {
            background: #0F172A;
            background: linear-gradient(to right top, #0f172a, #1e293b, #334155);
            font-family: 'Poppins', sans-serif;
        }
        .stTitle {
            font-size: 3em; color: #E2E8F0; font-weight: 700; text-align: left;
        }
        .stMarkdown p, .stMarkdown li { 
            font-size: 1.1rem; color: #94A3B8; 
        }
        [data-testid="stSidebar"] { 
            background-color: #1E293B; border-right: 1px solid #334155; 
        }
        .stButton>button {
            border-radius: 12px; border: 2px solid #38BDF8; color: #38BDF8;
            background-color: transparent; font-weight: 600; transition: all 0.3s ease-in-out;
            padding: 10px 24px; width: 100%;
        }
        .stButton>button:hover {
            background-color: #38BDF8; color: #0F172A; transform: scale(1.05);
        }
        .stTabs [data-baseweb="tab-list"] { 
            gap: 24px; 
        }
        .stTabs [data-baseweb="tab"] { 
            height: 50px; background-color: transparent; border-radius: 8px; color: #94A3B8;
        }
        .stTabs [data-baseweb="tab--selected"] { 
            background-color: #334155; color: #E2E8F0; 
        }
        [data-testid="stMetric"] { 
            background-color: #1E293B; padding: 20px; border-radius: 12px;
            border-left: 5px solid #38BDF8;
        }
        [data-testid="stMetricLabel"] { 
            font-size: 1.1em; color: #94A3B8; 
        }
        [data-testid="stFileUploader"] {
             background-color: #1E293B; padding: 20px; border-radius: 12px;
        }
        .st-expander {
            background-color: #1E293B; border-radius: 12px; border: 1px solid #334155;
        }
    </style>
    """, unsafe_allow_html=True)

load_css()

# ------------------------------
# Sidebar for DBSCAN Parameters
# ------------------------------
with st.sidebar:
    st.header("⚙️ DBSCAN Parameters")
    st.markdown("Adjust the algorithm's sensitivity.")
    eps = st.slider("Epsilon (eps)", min_value=0.5, max_value=5.0, value=2.0, step=0.1,
                    help="Max distance between two samples for one to be considered as in the neighborhood of the other. Lower values require points to be closer.")
    min_samples = st.slider("Minimum Samples", min_value=1, max_value=20, value=5, step=1,
                            help="The number of samples in a neighborhood for a point to be considered as a core point.")
    
    st.markdown("---")
    with st.expander("📘 About DBSCAN"):
        st.markdown("""
        **DBSCAN** is a density-based clustering algorithm. It's great for finding outliers (like fraud) in data that might not have a clear shape.
        - **Clusters:** Dense regions of points.
        - **Outliers (-1):** Points in low-density regions, isolated from clusters.
        """)

# ------------------------------
# Main App Interface
# ------------------------------
st.markdown('<h1 class="stTitle">💳 Credit Card Fraud Detection AI</h1>', unsafe_allow_html=True)
st.markdown("<p>Use this interactive tool to detect fraudulent transactions using the DBSCAN unsupervised algorithm. Upload your data or use a sample to see it in action.</p>", unsafe_allow_html=True)

# ------------------------------
# Step 1: Data Input Section
# ------------------------------
st.subheader("1. Provide Your Data")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None

input_col1, input_col2 = st.columns([1, 1])

with input_col1:
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

with input_col2:
    if st.button("Generate New Random Sample Dataset"):
        np.random.seed()  # Use a different seed each time
        normal_data = np.random.normal(0, 1, size=(950, 28))
        fraud_data = np.random.normal(4, 1, size=(50, 28))
        data_array = np.vstack([normal_data, fraud_data])
        columns = [f"V{i}" for i in range(1, 29)]
        st.session_state.data = pd.DataFrame(data_array, columns=columns)
        st.session_state.data['Class'] = [0]*950 + [1]*50
        st.success("New random dataset generated!")

if uploaded_file is not None:
    st.session_state.data = pd.read_csv(uploaded_file)

if st.session_state.data is None:
    st.info("Please upload a CSV or generate a sample dataset to begin analysis.")
    st.stop()

data = st.session_state.data
with st.expander("Preview Current Dataset"):
    st.dataframe(data.head())


# ------------------------------
# Step 2: Preprocessing and DBSCAN Execution
# ------------------------------
st.subheader("2. Analysis & Results")

try:
    with st.spinner('Scaling data and running DBSCAN... Please wait.'):
        time.sleep(1) # Simulate processing time

        # --- Preprocessing ---
        if 'Class' in data.columns:
            X = data.drop(['Class'], axis=1)
        else:
            X = data.copy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # --- DBSCAN Execution ---
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_scaled)
        
        # --- PCA for Visualization ---
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        plot_df = pd.DataFrame({
            'PCA Component 1': X_pca[:,0],
            'PCA Component 2': X_pca[:,1],
            'Cluster': labels.astype(str) # Convert to string for discrete color mapping
        })

    st.success("Analysis complete!")

    # ------------------------------
    # Step 3: Display Results
    # ------------------------------
    
    # --- Key Metrics ---
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_outliers = np.sum(labels == -1)

    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric(label="Detected Clusters", value=num_clusters)
    metric_col2.metric(label="Detected Outliers (Potential Fraud)", value=num_outliers)

    # --- Results Tabs ---
    tab1, tab2, tab3 = st.tabs(["📊 Visualization", "📋 Outlier Transactions", "ℹ️ Ground Truth Comparison"])

    with tab1:
        st.subheader("DBSCAN Clusters & Outliers (PCA 2D View)")
        fig = px.scatter(
            plot_df, x='PCA Component 1', y='PCA Component 2',
            color='Cluster',
            title="Interactive PCA Visualization",
            color_discrete_map={'-1': '#FF5733'}, # Highlight outliers
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#1E293B',
            font_color='white',
            legend_title_text='Cluster'
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Transactions Identified as Outliers")
        if num_outliers > 0:
            outliers_df = data.iloc[np.where(labels == -1)]
            st.dataframe(outliers_df)
            
            # Download button
            csv = outliers_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Outliers as CSV",
                data=csv,
                file_name='outliers_detected.csv',
                mime='text/csv',
            )
        else:
            st.info("No outliers were detected with the current parameters.")

    with tab3:
        st.subheader("Comparison with Actual Labels (if available)")
        if 'Class' in data.columns:
            plot_df['Actual Class'] = data['Class'].values
            fraud_outliers = plot_df[(plot_df['Cluster']=='-1') & (plot_df['Actual Class']==1)]
            
            st.info(f"**Recall:** DBSCAN correctly identified **{len(fraud_outliers)}** out of **{data['Class'].sum()}** actual fraudulent transactions as outliers.")
            
            # Add a confusion matrix or more detailed stats if desired
            comparison_df = data.copy()
            comparison_df['Predicted_Cluster'] = labels
            st.write("Data with Predicted Clusters:")
            st.dataframe(comparison_df.head())
        else:
            st.warning("No 'Class' column found in the dataset for ground truth comparison.")

except Exception as e:
    st.error(f"An error occurred during analysis: {e}")
    st.error("Please check if your CSV data is formatted correctly (all numeric columns).")
