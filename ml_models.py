import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, silhouette_score
)
import plotly.express as px
import plotly.graph_objects as go

def preprocess_data(data, handle_missing_method='Fill with mean', remove_outliers=True, target_column=None, model_type=None):
    """
    Preprocess the financial data.
    
    Parameters:
    - data: pandas DataFrame with the raw data
    - handle_missing_method: Method to handle missing values
    - remove_outliers: Whether to remove outliers
    - target_column: Name of the target column for regression
    - model_type: Type of ML model to be used
    
    Returns:
    - preprocessed_data: pandas DataFrame with preprocessed data
    """
    # Make a copy of the data to avoid modifying the original
    preprocessed_data = data.copy()
    
    # Convert date columns to datetime if they exist
    date_cols = preprocessed_data.select_dtypes(include=['object']).columns
    for col in date_cols:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                preprocessed_data[col] = pd.to_datetime(preprocessed_data[col])
            except:
                pass
    
    # Handle missing values
    if handle_missing_method == 'Drop rows with missing values':
        preprocessed_data = preprocessed_data.dropna()
    else:
        numeric_cols = preprocessed_data.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if preprocessed_data[col].isna().sum() > 0:
                if handle_missing_method == 'Fill with mean':
                    preprocessed_data[col] = preprocessed_data[col].fillna(preprocessed_data[col].mean())
                elif handle_missing_method == 'Fill with median':
                    preprocessed_data[col] = preprocessed_data[col].fillna(preprocessed_data[col].median())
                elif handle_missing_method == 'Fill with zero':
                    preprocessed_data[col] = preprocessed_data[col].fillna(0)
        
        # For categorical columns, fill with mode
        cat_cols = preprocessed_data.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if preprocessed_data[col].isna().sum() > 0:
                preprocessed_data[col] = preprocessed_data[col].fillna(preprocessed_data[col].mode()[0])
    
    # Remove outliers if requested
    if remove_outliers and model_type != "K-Means Clustering":
        numeric_cols = preprocessed_data.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            # Skip target column for outlier removal in regression models
            if model_type in ["Linear Regression", "Logistic Regression"] and col == target_column:
                continue
                
            # Calculate IQR
            Q1 = preprocessed_data[col].quantile(0.25)
            Q3 = preprocessed_data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Filter outliers
            preprocessed_data = preprocessed_data[(preprocessed_data[col] >= lower_bound) & 
                                                (preprocessed_data[col] <= upper_bound)]
    
    return preprocessed_data

def engineer_features(data, selected_features, target_column=None, model_type=None, binarize_target=False, binarize_threshold=None, n_clusters=None):
    """
    Engineer features for the machine learning model.
    
    Parameters:
    - data: pandas DataFrame with preprocessed data
    - selected_features: List of features to include
    - target_column: Name of the target column (for regression models)
    - model_type: Type of ML model to be used
    - binarize_target: Whether to binarize the target (for Logistic Regression)
    - binarize_threshold: Threshold for binarization
    - n_clusters: Number of clusters (for K-Means)
    
    Returns:
    - feature_data: Dict or DataFrame with engineered features and metadata
    """
    feature_data = {}
    
    # Create a copy of data to avoid modifying the original
    data_copy = data.copy()
    
    # First check and convert timestamp/datetime columns
    for col in selected_features:
        # Skip if column doesn't exist
        if col not in data_copy.columns:
            continue
            
        # Skip empty columns
        if data_copy[col].empty:
            continue
            
        try:
            # Check if column contains datetime objects or pandas Timestamp objects
            # Use safe checking methods
            first_non_null_val = data_copy[col].dropna().iloc[0] if not data_copy[col].dropna().empty else None
            
            if pd.api.types.is_datetime64_any_dtype(data_copy[col]) or (first_non_null_val is not None and isinstance(first_non_null_val, pd.Timestamp)):
                # Convert to Unix timestamp (float)
                data_copy[col] = data_copy[col].map(lambda x: x.timestamp() if hasattr(x, 'timestamp') else (np.nan if pd.isna(x) else x))
        except (IndexError, AttributeError, TypeError) as e:
            # If any error occurs, try to handle as gracefully as possible
            print(f"Warning: Error processing column {col}: {str(e)}")
            # Skip this column if it causes issues
            if col in selected_features:
                selected_features.remove(col)
    
    if model_type == "K-Means Clustering":
        # For clustering, we only need the selected features - first convert to a DataFrame to make it easier to handle
        X_df = data_copy[selected_features].copy()
        
        # Ensure all data is numeric
        for col in X_df.columns:
            # Try to convert any remaining non-numeric types
            try:
                if pd.api.types.is_datetime64_any_dtype(X_df[col]) or X_df[col].apply(lambda x: isinstance(x, pd.Timestamp)).any():
                    # Convert timestamps to float (seconds since epoch)
                    X_df[col] = X_df[col].apply(lambda x: x.timestamp() if isinstance(x, pd.Timestamp) else (pd.Timestamp(x).timestamp() if isinstance(x, str) else x))
            except Exception as e:
                print(f"Warning: Could not convert column {col} to numeric: {str(e)}")
                # Replace with mean or zeros if conversion fails
                if X_df[col].dtype.kind in 'ifc':  # integer, float, complex
                    X_df[col] = X_df[col].fillna(X_df[col].mean())
                else:
                    X_df[col] = 0
        
        # Convert to numpy array
        X = X_df.values.astype(float)
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        feature_data = {
            'data': X_scaled,
            'feature_names': selected_features,
            'scaler': scaler,
            'n_clusters': n_clusters
        }
    
    else:  # Linear or Logistic Regression
        # Create DataFrame copies to make it easier to handle data
        X_df = data_copy[selected_features].copy()
        
        # Ensure all features are numeric
        for col in X_df.columns:
            # Try to convert any remaining non-numeric types
            try:
                if pd.api.types.is_datetime64_any_dtype(X_df[col]) or X_df[col].apply(lambda x: isinstance(x, pd.Timestamp)).any():
                    # Convert timestamps to float (seconds since epoch)
                    X_df[col] = X_df[col].apply(lambda x: x.timestamp() if isinstance(x, pd.Timestamp) else (pd.Timestamp(x).timestamp() if isinstance(x, str) else x))
            except Exception as e:
                print(f"Warning: Could not convert column {col} to numeric: {str(e)}")
                # Replace with mean or zeros if conversion fails
                if X_df[col].dtype.kind in 'ifc':  # integer, float, complex
                    X_df[col] = X_df[col].fillna(X_df[col].mean())
                else:
                    X_df[col] = 0
        
        # Convert to numpy arrays
        X = X_df.values.astype(float)
        y = data_copy[target_column].values
        
        # Handle timestamp in target column if needed
        try:
            first_target_val = data_copy[target_column].dropna().iloc[0] if not data_copy[target_column].dropna().empty else None
            
            if pd.api.types.is_datetime64_any_dtype(data_copy[target_column]) or (first_target_val is not None and isinstance(first_target_val, pd.Timestamp)):
                y = np.array([x.timestamp() if hasattr(x, 'timestamp') else (np.nan if pd.isna(x) else x) for x in y])
        except (IndexError, AttributeError, TypeError) as e:
            print(f"Warning: Error processing target column {target_column}: {str(e)}")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # For Logistic Regression, we may need to binarize the target
        if model_type == "Logistic Regression" and binarize_target:
            y = (y > binarize_threshold).astype(int)
        
        feature_data = {
            'X': X_scaled,
            'y': y,
            'feature_names': selected_features,
            'target_column': target_column,
            'scaler': scaler
        }
    
    return feature_data

def split_data(feature_data, test_size=0.2, random_state=42, model_type=None, target_column=None):
    """
    Split the data into training and testing sets.
    
    Parameters:
    - feature_data: Dict or DataFrame with engineered features
    - test_size: Proportion of the data to include in the test split
    - random_state: Random seed for reproducibility
    - model_type: Type of ML model to be used
    - target_column: Name of the target column
    
    Returns:
    - split_result: Dict with train/test data
    """
    if model_type == "K-Means Clustering":
        # For clustering, we don't need a train/test split, but return all data
        return {
            'data': feature_data['data'],
            'feature_names': feature_data['feature_names'],
            'n_clusters': feature_data['n_clusters']
        }
    
    else:  # Linear or Logistic Regression
        X = feature_data['X']
        y = feature_data['y']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_data['feature_names'],
            'target_column': feature_data['target_column']
        }

def train_model(X_train, y_train=None, model_type=None, params=None):
    """
    Train the machine learning model.
    
    Parameters:
    - X_train: Training features
    - y_train: Training target (not used for K-Means)
    - model_type: Type of ML model to train
    - params: Model parameters
    
    Returns:
    - model: Trained model
    - evaluation: Dict with evaluation metrics and visualizations
    """
    if model_type == "Linear Regression":
        # Initialize and train Linear Regression model
        # Note: 'normalize' parameter was deprecated in scikit-learn 1.0 and removed in 1.2
        model = LinearRegression(
            fit_intercept=params.get('fit_intercept', True)
        )
        model.fit(X_train, y_train)
        
        # Get model coefficients
        coefficients = model.coef_
        
        # Prepare evaluation metrics (will be calculated in evaluation phase)
        evaluation = {
            'coefficients': coefficients,
            'feature_names': params.get('feature_names', [f'Feature {i}' for i in range(X_train.shape[1])])
        }
    
    elif model_type == "Logistic Regression":
        # Initialize and train Logistic Regression model
        model = LogisticRegression(
            C=params.get('C', 1.0),
            max_iter=params.get('max_iter', 100),
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Get model coefficients
        coefficients = model.coef_[0]
        
        # Prepare evaluation metrics (will be calculated in evaluation phase)
        evaluation = {
            'coefficients': coefficients,
            'feature_names': params.get('feature_names', [f'Feature {i}' for i in range(X_train.shape[1])])
        }
    
    elif model_type == "K-Means Clustering":
        # Initialize and train K-Means model
        model = KMeans(
            n_clusters=params.get('n_clusters', 3),
            max_iter=params.get('max_iter', 300),
            random_state=42
        )
        model.fit(X_train)
        
        # Get cluster labels and centers
        labels = model.labels_
        centers = model.cluster_centers_
        
        # Calculate inertia (sum of squared distances)
        inertia = model.inertia_
        
        # Calculate silhouette score
        silhouette = silhouette_score(X_train, labels) if len(np.unique(labels)) > 1 else 0
        
        # Create cluster visualization (2D projection if more than 2 features)
        if X_train.shape[1] > 2:
            # Use PCA for dimensionality reduction to 2D
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_train)
            
            # Create scatter plot of clusters
            fig = px.scatter(
                x=X_pca[:, 0], 
                y=X_pca[:, 1], 
                color=labels,
                title="Cluster Visualization (PCA Projection)",
                color_continuous_scale=px.colors.sequential.Reds,
                labels={'color': 'Cluster'}
            )
            
            # Add cluster centers to the plot
            centers_pca = pca.transform(centers)
            fig.add_trace(
                go.Scatter(
                    x=centers_pca[:, 0],
                    y=centers_pca[:, 1],
                    mode='markers',
                    marker=dict(
                        symbol='x',
                        size=15,
                        color='black',
                        line=dict(width=2)
                    ),
                    name='Cluster Centers'
                )
            )
            
        else:  # Direct visualization for 2D data
            feature_names = params.get('feature_names', ['Feature 1', 'Feature 2'])
            fig = px.scatter(
                x=X_train[:, 0], 
                y=X_train[:, 1], 
                color=labels,
                title="Cluster Visualization",
                color_continuous_scale=px.colors.sequential.Reds,
                labels={
                    'x': feature_names[0],
                    'y': feature_names[1],
                    'color': 'Cluster'
                }
            )
            
            # Add cluster centers to the plot
            fig.add_trace(
                go.Scatter(
                    x=centers[:, 0],
                    y=centers[:, 1],
                    mode='markers',
                    marker=dict(
                        symbol='x',
                        size=15,
                        color='black',
                        line=dict(width=2)
                    ),
                    name='Cluster Centers'
                )
            )
        
        # Prepare evaluation metrics
        evaluation = {
            'inertia': inertia,
            'silhouette_score': silhouette,
            'cluster_centers': centers,
            'cluster_labels': labels,
            'feature_names': params.get('feature_names', [f'Feature {i}' for i in range(X_train.shape[1])]),
            'cluster_plot': fig
        }
    
    return model, evaluation

def evaluate_model(model, X_test, y_test, model_type, feature_names):
    """
    Evaluate the trained model on test data.
    
    Parameters:
    - model: Trained ML model
    - X_test: Test features
    - y_test: Test target
    - model_type: Type of ML model
    - feature_names: List of feature names
    
    Returns:
    - evaluation: Dict with updated evaluation metrics and visualizations
    """
    evaluation = {
        'feature_names': feature_names
    }
    
    if model_type == "Linear Regression":
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Add metrics to evaluation
        evaluation.update({
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'coefficients': model.coef_,
            'y_pred': y_pred
        })
    
    elif model_type == "Logistic Regression":
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # ROC curve
        try:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = auc(fpr, tpr)
            
            # Create ROC curve plot
            roc_fig = go.Figure()
            roc_fig.add_trace(
                go.Scatter(
                    x=fpr, 
                    y=tpr,
                    mode='lines',
                    name=f'ROC Curve (AUC = {auc_score:.3f})',
                    line=dict(color='red', width=2)
                )
            )
            roc_fig.add_trace(
                go.Scatter(
                    x=[0, 1], 
                    y=[0, 1],
                    mode='lines',
                    name='Random Classifier',
                    line=dict(color='gray', dash='dash')
                )
            )
            roc_fig.update_layout(
                title='ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                legend=dict(x=0.01, y=0.99),
                width=700,
                height=500
            )
            
            # Add to evaluation
            evaluation.update({
                'roc_curve_plot': roc_fig
            })
        except Exception as e:
            print(f"Error creating ROC curve: {str(e)}")
        
        # Confusion matrix
        try:
            cm = confusion_matrix(y_test, y_pred)
            
            # Create confusion matrix plot
            cm_fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Negative (0)', 'Positive (1)'],
                y=['Negative (0)', 'Positive (1)'],
                text_auto=True,
                color_continuous_scale='Reds'
            )
            cm_fig.update_layout(
                title='Confusion Matrix',
                width=600,
                height=500
            )
            
            # Add to evaluation
            evaluation.update({
                'confusion_matrix': cm,
                'confusion_matrix_plot': cm_fig
            })
        except Exception as e:
            print(f"Error creating confusion matrix: {str(e)}")
            # Ensure at least the raw matrix is available
            try:
                cm = confusion_matrix(y_test, y_pred)
                evaluation.update({'confusion_matrix': cm})
            except:
                pass
        
        # Add metrics to evaluation (without overwriting existing values)
        evaluation.update({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'coefficients': model.coef_[0],
            'y_pred': y_pred,
            'y_prob': y_prob
        })
        
        # Add AUC if available
        try:
            evaluation.update({'auc': auc_score})
        except:
            # AUC may not be calculated if ROC curve creation failed
            pass
    
    return evaluation

def predict_and_visualize(model, X_test, y_test, model_type, feature_names, target_column):
    """
    Make predictions on test data and visualize results.
    
    Parameters:
    - model: Trained ML model
    - X_test: Test features
    - y_test: Test target
    - model_type: Type of ML model
    - feature_names: List of feature names
    - target_column: Name of the target column
    
    Returns:
    - fig: Plotly figure with results visualization
    """
    if model_type == "Linear Regression":
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Create scatter plot of actual vs predicted values
        fig = go.Figure()
        
        # Add scatter plot for actual vs predicted
        fig.add_trace(
            go.Scatter(
                x=y_test,
                y=y_pred,
                mode='markers',
                name='Test Data',
                marker=dict(
                    color='red',
                    size=8
                )
            )
        )
        
        # Add perfect prediction line
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(
                    color='black',
                    dash='dash'
                )
            )
        )
        
        fig.update_layout(
            title='Actual vs Predicted Values',
            xaxis_title=f'Actual {target_column}',
            yaxis_title=f'Predicted {target_column}',
            width=800,
            height=600
        )
        
    elif model_type == "Logistic Regression":
        # Make probability predictions
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Create histogram of prediction probabilities
        fig = px.histogram(
            x=y_prob,
            color=y_test,
            nbins=50,
            opacity=0.7,
            color_discrete_sequence=['blue', 'red'],
            labels={'x': 'Prediction Probability', 'color': 'Actual Class'},
            title='Distribution of Prediction Probabilities by Class'
        )
        
        fig.update_layout(
            xaxis_title='Probability of Positive Class',
            yaxis_title='Count',
            width=800,
            height=600,
            bargap=0.1
        )
    
    return fig
