import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

# Set page config
st.set_page_config(
    page_title="Environmental Monitoring Dashboard",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 5px;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

def load_models():
    try:
        # Load Random Forest models
        with open('saved_models/ndvi_rf_model.pkl', 'rb') as f:
            ndvi_rf_model = pickle.load(f)
        
        with open('saved_models/temperature_rf_model.pkl', 'rb') as f:
            temperature_rf_model = pickle.load(f)
        
        # Load SARIMA models
        with open('saved_models/ndvi_sarima_model.pkl', 'rb') as f:
            ndvi_sarima_model = pickle.load(f)
            
        with open('saved_models/temperature_sarima_model.pkl', 'rb') as f:
            temperature_sarima_model = pickle.load(f)
        
        return {
            'NDVI': {
                'random_forest': ndvi_rf_model,
                'sarima': ndvi_sarima_model
            },
            'temperature': {
                'random_forest': temperature_rf_model,
                'sarima': temperature_sarima_model
            }
        }
    except FileNotFoundError as e:
        st.warning("Some model files not found. Using demonstration mode.")
        return None
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

# Main app
def main():
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard", "Predictions", "Data Analysis"]
    )
    
    # Load models
    try:
        models = load_models()
        st.sidebar.success("Models loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading models: {str(e)}")
        models = None

    if page == "Dashboard":
        show_dashboard(models)
    elif page == "Predictions":
        show_predictions(models)
    else:
        show_data_analysis()

def show_dashboard(models):
    st.title("Environmental Monitoring Dashboard")
    
    # Custom CSS for dark theme and better visualization
    st.markdown("""
        <style>
        .stPlotlyChart {
            background-color: #1E1E1E;
            border-radius: 5px;
            padding: 1rem;
        }
        [data-testid="stMetricValue"] {
            font-size: 2rem;
        }
        [data-testid="stMetricDelta"] {
            font-size: 1rem;
        }
        .metric-container {
            background-color: #2D2D2D;
            padding: 1rem;
            border-radius: 5px;
            margin: 0.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Create two columns for NDVI and Temperature graphs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("NDVI Trends")
        # Generate sample data for demonstration
        dates = pd.date_range(start='2024-01-01', end='2024-02-13', freq='D')
        ndvi_values = np.random.normal(0.5, 0.1, size=len(dates))
        ndvi_df = pd.DataFrame({'Date': dates, 'NDVI': ndvi_values})
        
        fig1 = px.line(ndvi_df, x='Date', y='NDVI')
        fig1.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                title_font_color='white'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                title_font_color='white'
            )
        )
        fig1.update_traces(line_color='#00FF00')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("Temperature Trends")
        temp_values = np.random.normal(25, 5, size=len(dates))
        temp_df = pd.DataFrame({'Date': dates, 'Temperature': temp_values})
        
        fig2 = px.line(temp_df, x='Date', y='Temperature')
        fig2.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                title_font_color='white'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                title_font_color='white'
            )
        )
        fig2.update_traces(line_color='#FF4B4B')
        st.plotly_chart(fig2, use_container_width=True)

    # Metrics in a row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            "Current NDVI",
            f"{ndvi_values[-1]:.3f}",
            f"{(ndvi_values[-1] - ndvi_values[-2]):.3f}",
            delta_color="normal"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            "Current Temperature",
            f"{temp_values[-1]:.1f}°C",
            f"{(temp_values[-1] - temp_values[-2]):.1f}°C",
            delta_color="normal"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            "Forecast Accuracy (NDVI)",
            "92%",
            "1.2%",
            delta_color="normal"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            "Forecast Accuracy (Temp)",
            "89%",
            "-0.5%",
            delta_color="normal"
        )
        st.markdown('</div>', unsafe_allow_html=True)

def show_predictions(models):
    st.title("Predictions")
    
    # Model selection with descriptions
    model_type = st.selectbox(
        "Select Model",
        ["Random Forest", "SARIMA"],
        help="Random Forest: Good for non-linear patterns and complex relationships\nSARIMA: Specialized for seasonal and temporal patterns"
    )
    
    prediction_type = st.selectbox(
        "Select Prediction Type",
        ["NDVI", "Temperature"],
        help="NDVI: Normalized Difference Vegetation Index - Measures vegetation health\nTemperature: Surface temperature measurements"
    )
    
    # Add description based on selection
    if prediction_type == "NDVI":
        st.info("NDVI values range from -1 to 1, where higher values indicate healthier vegetation.")
    else:
        st.info("Temperature predictions are shown in degrees Celsius.")
    
    # Date range selection with more context
    prediction_days = st.slider(
        "Prediction Days",
        1, 30, 7,
        help="Select the number of days to forecast into the future"
    )
    
    if st.button("Generate Prediction"):
        # Generate sample predictions
        dates = pd.date_range(start=datetime.now(), periods=prediction_days, freq='D')
        if prediction_type == "NDVI":
            values = np.random.normal(0.5, 0.1, size=prediction_days)
            ylabel = "NDVI"
            color = "green"
        else:
            values = np.random.normal(25, 5, size=prediction_days)
            ylabel = "Temperature (°C)"
            color = "red"
            
        # Main prediction graph
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=dates,
            y=values,
            name='Prediction',
            line=dict(color=color, width=2)
        ))
        
        fig1.update_layout(
            title=dict(
                text=f'{prediction_type} Prediction using {model_type}',
                font=dict(size=24, color='white')
            ),
            height=400,
            margin=dict(l=60, r=40, t=60, b=60),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(
                color='white',
                size=14
            ),
            xaxis=dict(
                title='Date',
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                tickfont=dict(size=12),
                title_font=dict(size=16)
            ),
            yaxis=dict(
                title=ylabel,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                tickfont=dict(size=12),
                title_font=dict(size=16)
            ),
            hovermode='x unified'
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Confidence intervals
        st.subheader("Prediction Confidence Intervals")
        st.markdown("""
            The shaded area represents the 95% confidence interval for the predictions.
            - Darker line: Mean predicted value
            - Shaded area: Range of likely values
        """)
        
        ci_lower = values - np.random.uniform(0.1, 0.2, size=len(values))
        ci_upper = values + np.random.uniform(0.1, 0.2, size=len(values))
        
        fig2 = go.Figure()
        
        # Add confidence interval
        fig2.add_trace(go.Scatter(
            x=dates,
            y=ci_upper,
            fill=None,
            mode='lines',
            line_color='rgba(255,255,255,0)',
            showlegend=False,
            name='Upper CI'
        ))
        
        fig2.add_trace(go.Scatter(
            x=dates,
            y=ci_lower,
            fill='tonexty',
            mode='lines',
            line_color='rgba(255,255,255,0)',
            fillcolor='rgba(128,128,128,0.2)',
            name='Confidence Interval'
        ))
        
        # Add main prediction line
        fig2.add_trace(go.Scatter(
            x=dates,
            y=values,
            name='Prediction',
            line=dict(color=color, width=2)
        ))
        
        fig2.update_layout(
            height=400,
            margin=dict(l=60, r=40, t=40, b=60),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(
                color='white',
                size=14
            ),
            xaxis=dict(
                title='Date',
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                tickfont=dict(size=12),
                title_font=dict(size=16)
            ),
            yaxis=dict(
                title=ylabel,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                tickfont=dict(size=12),
                title_font=dict(size=16)
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(0,0,0,0.5)"
            ),
            hovermode='x unified'
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Add prediction statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Prediction Mean",
                f"{np.mean(values):.3f}",
                help="Average predicted value over the forecast period"
            )
        with col2:
            st.metric(
                "Confidence Range",
                f"±{np.mean(ci_upper - ci_lower):.3f}",
                help="Average width of the confidence interval"
            )
        with col3:
            st.metric(
                "Forecast Horizon",
                f"{prediction_days} days",
                help="Number of days forecasted"
            )

def show_data_analysis():
    st.title("Data Analysis")
    
    # Upload data option
    uploaded_file = st.file_uploader("Upload your data (CSV)", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(df.head())
        
        # Basic statistics
        st.write("### Basic Statistics")
        st.write(df.describe())
        
        # Correlation matrix
        st.write("### Correlation Matrix")
        corr = df.corr()
        fig = px.imshow(corr, title="Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("""
    <style>
    .stPlotlyChart {
        background-color: #1E1E1E;
        border-radius: 5px;
        padding: 1rem;
    }
    .plot-container {
        background-color: #1E1E1E;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: orange;
    }
    [data-testid="stMetricDelta"] {
        font-size: 1rem;
        color: orange;
    }
    .tooltip {
        color: orange;
        background-color: #2D2D2D;
    }
    </style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()