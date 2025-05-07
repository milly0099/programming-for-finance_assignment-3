import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_dataset_overview(data):
    """
    Create a plot showing the overview of the dataset.
    
    Parameters:
    - data: pandas DataFrame with the dataset
    
    Returns:
    - fig: Plotly figure with dataset overview
    """
    # Calculate statistics for numeric columns
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    
    if numeric_data.empty:
        # If no numeric columns, return a message
        fig = go.Figure()
        fig.add_annotation(
            text="No numeric data available for overview",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Calculate summary statistics
    stats_df = pd.DataFrame({
        'Mean': numeric_data.mean(),
        'Median': numeric_data.median(),
        'Std': numeric_data.std(),
        'Min': numeric_data.min(),
        'Max': numeric_data.max()
    })
    
    # Reset index and rename the index column to 'Feature'
    stats_df = stats_df.reset_index()
    stats_df = stats_df.rename(columns={'index': 'Feature'})
    
    # Make sure the 'Feature' column exists
    if 'Feature' not in stats_df.columns:
        # If the index column has a different name, rename the first column to 'Feature'
        first_col_name = stats_df.columns[0]
        stats_df = stats_df.rename(columns={first_col_name: 'Feature'})
    
    # Reshape for plotting
    try:
        stats_long = pd.melt(
            stats_df, 
            id_vars=['Feature'], 
            value_vars=['Mean', 'Median', 'Std', 'Min', 'Max'],
            var_name='Statistic', 
            value_name='Value'
        )
    except KeyError:
        # If melting fails, debug and print column names
        fig = go.Figure()
        column_names = ", ".join(stats_df.columns.tolist())
        fig.add_annotation(
            text=f"Error in data structure. Columns: {column_names}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color='red')
        )
        return fig
    
    # Create grouped bar chart
    fig = px.bar(
        stats_long, 
        x='Feature', 
        y='Value', 
        color='Statistic',
        barmode='group',
        title='Numeric Features Overview',
        color_discrete_sequence=px.colors.sequential.Reds,
        height=500
    )
    
    fig.update_layout(
        xaxis_title='Feature',
        yaxis_title='Value',
        legend_title='Statistic',
        xaxis={'categoryorder': 'total descending'}
    )
    
    return fig

def plot_missing_values(data):
    """
    Create a plot showing missing values in the dataset.
    
    Parameters:
    - data: pandas DataFrame with the dataset
    
    Returns:
    - fig: Plotly figure with missing values visualization
    """
    # Calculate missing values count and percentage
    missing_count = data.isna().sum()
    missing_percent = (data.isna().sum() / len(data)) * 100
    
    # Create DataFrame for plotting
    missing_df = pd.DataFrame({
        'Feature': missing_count.index,
        'Count': missing_count.values,
        'Percent': missing_percent.values
    })
    
    # Filter features with missing values
    missing_df = missing_df[missing_df['Count'] > 0]
    
    if missing_df.empty:
        # If no missing values, return a message
        fig = go.Figure()
        fig.add_annotation(
            text="No missing values in the dataset",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color='green')
        )
        return fig
    
    # Sort by missing count
    missing_df = missing_df.sort_values('Count', ascending=False)
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=missing_df['Feature'],
            y=missing_df['Count'],
            name='Missing Count',
            marker_color='red'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=missing_df['Feature'],
            y=missing_df['Percent'],
            name='Missing Percent',
            mode='lines+markers',
            yaxis='y2',
            marker_color='black',
            line=dict(width=2)
        )
    )
    
    fig.update_layout(
        title='Missing Values by Feature',
        xaxis_title='Feature',
        yaxis_title='Missing Count',
        yaxis2=dict(
            title='Missing Percent (%)',
            overlaying='y',
            side='right'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        height=500
    )
    
    return fig

def plot_feature_importance(data, selected_features):
    """
    Create a correlation heatmap for selected features.
    
    Parameters:
    - data: pandas DataFrame with the dataset
    - selected_features: List of selected feature names
    
    Returns:
    - fig: Plotly figure with correlation heatmap
    """
    # Calculate correlation matrix for selected features
    correlation = data[selected_features].corr()
    
    # Create heatmap
    fig = px.imshow(
        correlation,
        x=correlation.columns,
        y=correlation.columns,
        color_continuous_scale='Reds',
        title='Feature Correlation Heatmap',
        text_auto=True,
        aspect='auto'
    )
    
    fig.update_layout(
        height=600,
        width=700
    )
    
    return fig

def plot_train_test_split(test_size):
    """
    Create a pie chart showing the train/test split ratio.
    
    Parameters:
    - test_size: Proportion of test data
    
    Returns:
    - fig: Plotly figure with train/test split visualization
    """
    # Calculate train size
    train_size = 1 - test_size
    
    # Create data for pie chart
    labels = ['Training Data', 'Testing Data']
    values = [train_size, test_size]
    
    # Create pie chart
    fig = px.pie(
        values=values,
        names=labels,
        title='Train/Test Split Ratio',
        color_discrete_sequence=['#e50914', '#221f1f'],
        hole=0.4
    )
    
    fig.update_traces(
        textinfo='percent+label',
        textfont_size=14
    )
    
    return fig

def plot_evaluation_metrics(evaluation, model_type):
    """
    Create a visualization of model evaluation metrics.
    
    Parameters:
    - evaluation: Dict with evaluation metrics
    - model_type: Type of ML model
    
    Returns:
    - fig: Plotly figure with evaluation metrics visualization
    """
    fig = go.Figure()  # Initialize figure first to avoid unbound variable
    
    if model_type == "Linear Regression":
        # Extract metrics
        metrics = {
            'MSE': evaluation.get('mse', 0),
            'RMSE': evaluation.get('rmse', 0),
            'MAE': evaluation.get('mae', 0),
            'R² Score': evaluation.get('r2', 0)
        }
        
        # Add bars for metrics
        fig.add_trace(
            go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                marker_color=['#e50914', '#e50914', '#e50914', '#e50914']
            )
        )
        
        # Add text annotations
        for i, (metric, value) in enumerate(metrics.items()):
            fig.add_annotation(
                x=i,
                y=value + max(metrics.values()) * 0.05,
                text=f"{value:.4f}",
                showarrow=False
            )
        
        # Update layout
        fig.update_layout(
            title='Regression Model Evaluation Metrics',
            xaxis_title='Metric',
            yaxis_title='Value',
            height=500
        )
    
    elif model_type == "Logistic Regression":
        # Extract metrics
        metrics = {
            'Accuracy': evaluation.get('accuracy', 0),
            'Precision': evaluation.get('precision', 0),
            'Recall': evaluation.get('recall', 0),
            'F1 Score': evaluation.get('f1', 0),
            'AUC': evaluation.get('auc', 0)
        }
        
        # Add radar chart
        fig.add_trace(
            go.Scatterpolar(
                r=list(metrics.values()),
                theta=list(metrics.keys()),
                fill='toself',
                name='Metrics',
                line_color='red',
                fillcolor='rgba(229, 9, 20, 0.5)'
            )
        )
        
        # Update layout
        fig.update_layout(
            title='Classification Model Evaluation Metrics',
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            height=600
        )
    
    # For other model types (e.g., K-Means)
    else:
        fig.add_annotation(
            text=f"Metrics visualization not available for {model_type}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
    
    return fig
