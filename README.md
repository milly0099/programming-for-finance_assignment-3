# La Casa de Datos 💰

A Money Heist-themed Streamlit application for machine learning on financial data with interactive visualizations and a step-by-step ML workflow.

## Features

- **Money Heist Theme**: Stylish interface with red/black color scheme and Salvador Dalí mask visuals
- **Data Source Options**: Upload your own dataset or fetch stock data from Yahoo Finance
- **ML Models**: Linear Regression, Logistic Regression, and K-Means Clustering
- **Interactive Visualizations**: Beautiful Plotly charts for data exploration and model evaluation
- **Step-by-Step Workflow**: Guide users through the entire machine learning process

## Deployment Guide

### Option 1: Deploy on Streamlit Cloud (Recommended)

1. **Create a GitHub Repository**
   - Create a new GitHub repository and push this code to it
   - Make sure all files are included: Python scripts, .streamlit folder, assets folder, etc.

2. **Connect to Streamlit Cloud**
   - Visit [Streamlit Cloud](https://streamlit.io/cloud)
   - Sign up or log in with your GitHub account
   - Click "New app" and select your repository
   - Select "app.py" as the main file
   - Click "Deploy"

3. **Configuration**
   - The required configuration files are already included:
     - `setup.sh`: Sets up Streamlit configuration
     - `Procfile`: Specifies the command to run the app
     - `.streamlit/config.toml`: Configures the Streamlit theme and server settings

### Option 2: Deploy on Render

1. **Create a Render Account**
   - Sign up at [render.com](https://render.com)

2. **Create a New Web Service**
   - Click "New +" and select "Web Service"
   - Connect your GitHub repository
   - Use the following settings:
     - **Name**: Your choice (e.g., "la-casa-de-datos")
     - **Environment**: Python 3
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

3. **Set Environment Variables**
   - Add the following environment variable:
     - `STREAMLIT_SERVER_PORT`: `$PORT`

### Option 3: Local Deployment

1. **Clone the Repository**
   ```bash
   git clone <your-repository-url>
   cd <repository-name>
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## Requirements

The application requires the following Python packages:
- streamlit>=1.27.0
- pandas>=2.0.0
- numpy>=1.24.0
- yfinance>=0.2.30
- plotly>=5.15.0
- scikit-learn>=1.3.0
- matplotlib>=3.7.0
- requests>=2.31.0
- pillow>=10.0.0

## File Structure

- `app.py`: Main Streamlit application
- `ml_models.py`: Machine learning model implementations
- `utils.py`: Utility functions for styling and data handling
- `visuals.py`: Visualization functions for plots and charts
- `assets/`: Contains theme assets (mask.svg, welcome.svg, finish.svg)
- `.streamlit/`: Streamlit configuration files

## License

This project is open source and available under the MIT License.