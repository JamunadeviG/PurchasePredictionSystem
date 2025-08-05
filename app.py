import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Set page configuration
st.set_page_config(
    page_title="Purchase Prediction System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .prediction-success {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1rem 0;
    }
    
    .prediction-failure {
        background: linear-gradient(135deg, #f44336 0%, #da190b 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_trained_model():
    """Load the pre-trained model and preprocessing objects"""
    try:
        # Check if trained model exists
        if not os.path.exists('trained_model.pkl'):
            st.error("❌ Trained model not found. Please run 'python train_model.py' first to train the model.")
            st.info("💡 Click the button below to train the model now:")
            if st.button("🚀 Train Model Now"):
                with st.spinner("Training model... This may take a few minutes."):
                    import subprocess
                    result = subprocess.run(['python', 'train_model.py'], capture_output=True, text=True)
                    if result.returncode == 0:
                        st.success("✅ Model trained successfully! Please refresh the page.")
                        st.experimental_rerun()
                    else:
                        st.error(f"❌ Training failed: {result.stderr}")
            return None
        
        # Load the trained model
        with st.spinner("Loading trained model..."):
            model_data = joblib.load('trained_model.pkl')
        
        # Load original data for UI dropdowns
        original_data = pd.read_csv('pps1.csv')
        model_data['original_data'] = original_data
        
        return model_data
        
    except FileNotFoundError:
        st.error("❌ Dataset file 'pps1.csv' not found. Please ensure the file is in the same directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def create_metrics_dashboard(metrics):
    """Create a metrics dashboard"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Accuracy</h3>
            <h2>{metrics['accuracy']:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Precision</h3>
            <h2>{metrics['precision']:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Recall</h3>
            <h2>{metrics['recall']:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>F1-Score</h3>
            <h2>{metrics['f1']:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)

def create_prediction_charts(data):
    """Create visualization charts"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Category distribution
        fig1 = px.histogram(
            data, 
            x='Category', 
            color='Subscription Status',
            title="Subscription Status by Category",
            color_discrete_map={0: '#ff7f7f', 1: '#7fbf7f'}
        )
        fig1.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333333')
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Season distribution
        fig2 = px.histogram(
            data, 
            x='Season', 
            color='Subscription Status',
            title="Subscription Status by Season",
            color_discrete_map={0: '#ff7f7f', 1: '#7fbf7f'}
        )
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333333')
        )
        st.plotly_chart(fig2, use_container_width=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🛒 Purchase Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load pre-trained model
    model_data = load_trained_model()
    
    if model_data is None:
        st.stop()
    
    # Sidebar for input
    st.sidebar.header("🔧 Model Configuration")
    st.sidebar.markdown("### Customer Information")
    
    # Get unique values for dropdowns from original data
    original_data = model_data['original_data']
    
    # Simplified input fields - only the most important features
    st.sidebar.subheader("🎯 Key Customer Information")
    
    age = st.sidebar.number_input(
        "👤 Age",
        min_value=18,
        max_value=100,
        value=30,
        help="Customer's age"
    )
    
    category = st.sidebar.selectbox(
        "🛍️ Product Category",
        sorted(original_data['Category'].unique()),
        help="Select the product category"
    )
    
    purchase_amount = st.sidebar.number_input(
        "💰 Purchase Amount (USD)",
        min_value=0.0,
        max_value=1000.0,
        value=100.0,
        step=10.0,
        help="Enter the purchase amount in USD"
    )
    
    season = st.sidebar.selectbox(
        "🌤️ Season",
        sorted(original_data['Season'].unique()),
        help="Season of purchase"
    )
    
    review_rating = st.sidebar.slider(
        "⭐ Review Rating",
        min_value=1.0,
        max_value=5.0,
        value=3.5,
        step=0.1,
        help="Customer's review rating (1-5 stars)"
    )
    
    previous_purchases = st.sidebar.number_input(
        "📦 Previous Purchases",
        min_value=0,
        max_value=100,
        value=5,
        help="Number of previous purchases"
    )
    
    # Prediction button
    predict_button = st.sidebar.button("🔮 Make Prediction", use_container_width=True)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "🔮 Prediction Results", "📈 Data Insights"])
    
    with tab1:
        st.header("Model Performance Metrics")
        create_metrics_dashboard(model_data['metrics'])
        
        st.subheader("Model Information")
        model_info = pd.DataFrame({
            'Property': ['Model Type', 'Features Used', 'Training Method'],
            'Value': [model_data['metrics']['model_type'], len(model_data['feature_columns']), 'Random Forest with Class Balancing']
        })
        st.dataframe(model_info, use_container_width=True, hide_index=True)
    
    with tab2:
        st.header("Prediction Results")
        
        if predict_button:
            try:
                # Prepare input data using only the 6 key features
                # Encode categorical features
                category_encoded = model_data['encoders']['Category'].transform([category])[0]
                season_encoded = model_data['encoders']['Season'].transform([season])[0]
                
                # Create input array in the correct order: Age, Category, Purchase Amount, Season, Review Rating, Previous Purchases
                input_data = [age, category_encoded, purchase_amount, season_encoded, review_rating, previous_purchases]
                
                # Scale input data
                input_data_scaled = model_data['scaler'].transform([input_data])
                
                # Make prediction
                prediction = model_data['model'].predict(input_data_scaled)[0]
                prediction_proba = model_data['model'].predict_proba(input_data_scaled)[0]
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="prediction-success">
                            ✅ PREDICTION: Customer WILL Subscribe
                            <br>
                            Confidence: {prediction_proba[1]:.1%}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-failure">
                            ❌ PREDICTION: Customer will NOT Subscribe
                            <br>
                            Confidence: {prediction_proba[0]:.1%}
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Confidence gauge
                    confidence = max(prediction_proba)
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = confidence * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Confidence %"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Input summary - only show the features the user actually provided
                st.subheader("📋 Your Input Summary")
                input_summary = pd.DataFrame({
                    'Feature': ['👤 Age', '🛍️ Category', '💰 Purchase Amount', '🌤️ Season', 
                               '⭐ Review Rating', '📦 Previous Purchases'],
                    'Value': [f"{age} years", category, f"${purchase_amount:.2f}", season,
                             f"{review_rating:.1f}/5.0", f"{previous_purchases} purchases"]
                })
                st.dataframe(input_summary, use_container_width=True, hide_index=True)
                
                # Show a note about the model
                st.info("ℹ️ **Note:** This model focuses on the most impactful factors for subscription prediction: customer demographics, purchase behavior, and satisfaction ratings.")
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
        else:
            st.info("👈 Configure customer information in the sidebar and click 'Make Prediction' to see results.")
    
    with tab3:
        st.header("Data Insights & Visualizations")
        create_prediction_charts(original_data)
        
        # Data statistics
        st.subheader("Dataset Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Records", len(original_data))
            st.metric("Unique Categories", original_data['Category'].nunique())
        
        with col2:
            subscription_rate = (original_data['Subscription Status'] == 'Yes').mean()
            st.metric("Subscription Rate", f"{subscription_rate:.1%}")
            st.metric("Unique Seasons", original_data['Season'].nunique())

if __name__ == "__main__":
    main()
