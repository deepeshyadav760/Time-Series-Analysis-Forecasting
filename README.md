# Time Series Analysis and Forecasting Project for Bengaluru

This repository contains the code and documentation for a time series analysis and forecasting project focused on **Bengaluru**. The project analyzes trends, seasonal patterns, and residual components in time series data related to environmental factors such as vegetation (NDVI), precipitation, and air quality (AQI). The project leverages both traditional statistical methods and machine learning techniques for forecasting.

## Overview
This project focuses on time series analysis and forecasting using various machine learning models. It is built using **Streamlit** to provide an interactive web-based API for visualizing and predicting environmental factors such as NDVI, temperature, precipitation, and air quality.

**App:** https://deepeshyadav760-time-series-analysis-forecasting-app-ytbiuh.streamlit.app/

## Features
- **Data Collection**: Fetches NDVI, temperature, precipitation, and air quality data.
- **Data Preprocessing**: Includes smoothing, rolling means, and seasonal decomposition.
- **Modeling**:
  - **Random Forest Regressor** for short-term predictions.
  - **SARIMA** for time series forecasting.
- **Streamlit API**:
  - User-friendly interface for selecting models and variable.
  - Interactive visualizations of past trends, future forecasts and doing the analysis of other data (satellite data).

## Project Structure
```
Time_Series_Project/
│── saved_models/                                       # Pre-trained models (Random Forest & SARIMA)
│   ├── ndvi_rf_model.pkl
│   ├── ndvi_sarima_model.pkl
│   ├── temperature_rf_model.pkl
│   ├── temperature_sarima_model.pkl
│── app.py                                              # Streamlit API for visualization and forecasting
│── main.py                                             # Main script for data processing and modeling
│── Bengaluru_time_series_forecasting.ipynb             # Jupyter notebook for Bengaluru analysis
│── channai_main.ipynb                                  # Jupyter notebook for Chennai analysis
│── mumbai_main copy.ipynb                              # Jupyter notebook for Mumbai analysis
│── requirements.txt                                    # Dependencies
│── README.md                                           # Project documentation
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/deepeshyadav760/Time-Series-Analysis-Forecasting.git
   cd Time_Series_Project
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
- Open the **Streamlit app** in the browser.
- Select the city and environmental variable.
- View historical trends and model forecasts.

## Technologies Used
- **Python** (pandas, numpy, scikit-learn, statsmodels)
- **Machine Learning** (Random Forest, SARIMA)
- **Streamlit** (Interactive API)
- **Jupyter Notebooks** (Data exploration & modeling)

## Contributing
Feel free to fork the repository and submit pull requests with improvements.
