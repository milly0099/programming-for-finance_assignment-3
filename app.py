import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import base64
from datetime import datetime, timedelta
import io
from PIL import Image
import requests
from utils import (
    apply_money_heist_theme, 
    get_mask_image,
    get_welcome_gif, 
    get_completion_gif,
    display_notification,
    show_image_with_caption
)
from ml_models import (
    preprocess_data,
    engineer_features,
    split_data,
    train_model,
    evaluate_model,
    predict_and_visualize
)
from visuals import (
    plot_dataset_overview,
    plot_missing_values,
    plot_feature_importance,
    plot_train_test_split,
    plot_evaluation_metrics
)

# Page configuration
st.set_page_config(
    page_title="La Casa de Datos - Financial ML Heist",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Money Heist Theme
apply_money_heist_theme()

# Initialize session state for multi-step workflow
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'data' not in st.session_state:
    st.session_state.data = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'evaluation' not in st.session_state:
    st.session_state.evaluation = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None

# Sidebar setup
with st.sidebar:
    st.markdown("## 💰 **Operation Controls** 💰")
    mask_img = get_mask_image()
    st.image(mask_img, width=150)
    
    st.markdown("### 🕵️‍♀️ Data Acquisition")
    data_source = st.radio("Choose your data source:", ["Upload Kragle Dataset", "Fetch from Yahoo Finance"])
    
    if data_source == "Upload Kragle Dataset":
        uploaded_file = st.file_uploader("Upload your financial data CSV", type=['csv'])
        if uploaded_file is not None and st.session_state.data is None:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.session_state.step = 1
    
    else:  # Yahoo Finance
        ticker_symbol = st.text_input("Enter stock symbol (e.g., AAPL, MSFT, GOOGL)", "AAPL")
        start_date = st.date_input("Start date", datetime.now() - timedelta(days=365))
        end_date = st.date_input("End date", datetime.now())
        
        if st.button("Fetch Stock Data", key="fetch_button"):
            try:
                st.session_state.data = yf.download(ticker_symbol, start=start_date, end=end_date)
                # Reset index to make Date a column
                st.session_state.data = st.session_state.data.reset_index()
                st.session_state.step = 1
                display_notification("success", f"Successfully loaded {ticker_symbol} data!")
            except Exception as e:
                display_notification("error", f"Error fetching data: {str(e)}")
    
    st.markdown("### 🎭 Model Selection")
    model_options = ["Linear Regression", "Logistic Regression", "K-Means Clustering"]
    st.session_state.model_type = st.selectbox("Select your ML model:", model_options)
    
    st.markdown("### 📊 Heist Progress")
    progress_steps = ["Welcome", "Load Data", "Preprocessing", "Feature Engineering", 
                      "Train/Test Split", "Model Training", "Evaluation", "Results"]
    current_step = progress_steps[min(st.session_state.step, len(progress_steps)-1)]
    st.progress(min(st.session_state.step / (len(progress_steps)-1), 1.0))
    st.write(f"Current phase: **{current_step}**")

# Main content area
def display_welcome():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("""
        # 💰 **LA CASA DE DATOS** 💰
        ## *The Financial Machine Learning Heist*
        """)
        
        # Display welcome GIF
        welcome_gif = get_welcome_gif()
        st.image(welcome_gif, use_column_width=True)
        
        st.markdown("""
        ### Welcome to the most ambitious financial data heist of all time!
        
        In this operation, we'll infiltrate the financial markets, extract valuable data, 
        apply sophisticated machine learning algorithms, and extract insights that will make us rich (in knowledge)!
        
        #### 🎭 **THE PLAN:**
        1. **Secure the Data** - Upload your own dataset or extract it from Yahoo Finance
        2. **Prepare for the Heist** - Clean and preprocess the financial data
        3. **Build the Strategy** - Apply machine learning models to the data
        4. **Execute the Heist** - Visualize results and extract valuable insights
        
        **Are you ready to join the heist?** Start by selecting your data source on the left panel!
        """)
        
        st.markdown("---")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.image("https://pixabay.com/get/g922d4f239d427232ec1db1c27c974745fafb107e63bae34842f82da0148f14da9013592d21af96a9a7d2a5ed30123b7f8e3923c94f4a523297f61c9fb0f27fca_1280.jpg", 
                     caption="Join the team", width=300)
        with col_b:
            st.image("https://pixabay.com/get/g7080de81b8bcbfcd63eaabb1a946590acc03a427c91f8983d74fcced89ff999424439165fd51c423a5f563830b46cd0dfc1c7d2caf0882be52b3eb68b278dae0_1280.jpg", 
                     caption="Wear the mask", width=300)

def display_load_data():
    st.markdown("# 📊 **Phase 1: Data Acquisition**")
    st.markdown("### The first step of any successful heist is to secure the asset.")
    
    if st.session_state.data is not None:
        st.write(f"Dataset shape: {st.session_state.data.shape[0]} rows and {st.session_state.data.shape[1]} columns")
        
        # Display the first few rows of the data
        st.subheader("Data Preview:")
        st.dataframe(st.session_state.data.head())
        
        # Show dataset overview
        st.subheader("Dataset Overview:")
        fig = plot_dataset_overview(st.session_state.data)
        st.plotly_chart(fig, use_container_width=True, key="dataset_overview")
        
        # Display stock market trading imagery
        col1, col2, col3 = st.columns(3)
        with col1:
            show_image_with_caption(
                "https://pixabay.com/get/g1f630c8003bfa752b330d9ea61ac4f82bd6a4454e486974322dd3cb255c0d48675f075d62622b7b8126cea20505fd249680506c3e093a30609a7638975d832eb_1280.jpg",
                "Stock market data - the treasure we seek"
            )
        with col2:
            show_image_with_caption(
                "https://pixabay.com/get/g3555bdda2b61a40259f2554976c5634efc0bc708f69036c5c2a82f19269135af1080df3ebce889ab080acbc202d8d2eaad45f2461998d640e79ef8602ff01ed6_1280.jpg",
                "Trading patterns to exploit"
            )
        with col3:
            show_image_with_caption(
                "https://pixabay.com/get/g39435aeada1517701a144baf6051a6ac75a024eaa73faa161ffc6c10c3623e047e35f71f15d45097fa3c77179f232842d896ed1f69da75148850f0d58bf45107_1280.jpg",
                "Market insights to uncover"
            )
        
        if st.button("Proceed to Data Preprocessing", key="proceed_to_preprocessing"):
            st.session_state.step = 2
            display_notification("success", "Data loaded successfully! Moving to preprocessing phase.")
            st.rerun()
    else:
        st.info("⚠️ No data loaded yet. Please select your data source in the sidebar.")

def display_preprocessing():
    st.markdown("# 🧹 **Phase 2: Data Preprocessing**")
    st.markdown("### Before the heist, we need to clean up and prepare our tools.")
    
    if st.session_state.data is not None:
        # Show missing values before preprocessing
        st.subheader("Missing Values Before Preprocessing:")
        missing_vals_fig = plot_missing_values(st.session_state.data)
        st.plotly_chart(missing_vals_fig, use_container_width=True, key="missing_vals_before")
        
        # Define preprocessing options
        st.subheader("Preprocessing Options:")
        handle_missing = st.selectbox("How to handle missing values?", 
                                    ["Drop rows with missing values", "Fill with mean", "Fill with median", "Fill with zero"])
        
        remove_outliers = st.checkbox("Remove outliers?", value=True)
        
        # Target column selection based on model type
        if st.session_state.model_type in ["Linear Regression", "Logistic Regression"]:
            numeric_columns = st.session_state.data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if len(numeric_columns) > 0:
                st.session_state.target_column = st.selectbox("Select target column for prediction:", numeric_columns)
            else:
                st.error("No numeric columns found in the dataset. Can't proceed with regression models.")
                return
        
        # Preprocess the data
        if st.button("Preprocess Data", key="preprocess_button"):
            with st.spinner("Preprocessing data..."):
                st.session_state.preprocessed_data = preprocess_data(
                    st.session_state.data, 
                    handle_missing_method=handle_missing,
                    remove_outliers=remove_outliers,
                    target_column=st.session_state.target_column if 'target_column' in st.session_state else None,
                    model_type=st.session_state.model_type
                )
                
                # Show data after preprocessing
                st.subheader("Data After Preprocessing:")
                st.dataframe(st.session_state.preprocessed_data.head())
                
                # Show missing values after preprocessing
                st.subheader("Missing Values After Preprocessing:")
                missing_vals_after_fig = plot_missing_values(st.session_state.preprocessed_data)
                st.plotly_chart(missing_vals_after_fig, use_container_width=True, key="missing_vals_after")
                
                display_notification("success", "Data preprocessing complete! Your data is now clean and ready.")
                st.session_state.step = 3
                st.rerun()
    else:
        st.error("⚠️ No data available. Please go back to the data loading phase.")
        if st.button("Go Back to Data Loading", key="back_to_data_loading"):
            st.session_state.step = 1
            st.rerun()

def display_feature_engineering():
    st.markdown("# 🛠️ **Phase 3: Feature Engineering**")
    st.markdown("### Every successful heist needs a carefully crafted plan and the right tools.")
    
    if st.session_state.preprocessed_data is not None:
        # Show the preprocessed data
        st.subheader("Preprocessed Data:")
        st.dataframe(st.session_state.preprocessed_data.head())
        
        # Feature selection options based on model type
        if st.session_state.model_type in ["Linear Regression", "Logistic Regression"]:
            st.subheader("Feature Selection:")
            available_features = [col for col in st.session_state.preprocessed_data.columns 
                                if col != st.session_state.target_column]
            
            selected_features = st.multiselect(
                "Select features to use for training:",
                available_features,
                default=available_features[:min(5, len(available_features))]
            )
            
            if not selected_features:
                st.warning("⚠️ Please select at least one feature to proceed.")
                return
            
            # For Logistic Regression: ask if target needs to be binarized
            if st.session_state.model_type == "Logistic Regression":
                binarize_target = st.checkbox("Binarize target column? (required for Logistic Regression)", value=True)
                binarize_threshold = None
                if binarize_target:
                    binarize_threshold = st.slider("Select threshold for binarization:", 
                                                  float(st.session_state.preprocessed_data[st.session_state.target_column].min()), 
                                                  float(st.session_state.preprocessed_data[st.session_state.target_column].max()),
                                                  float(st.session_state.preprocessed_data[st.session_state.target_column].median()))
            else:
                binarize_target = False
                binarize_threshold = None
                
        elif st.session_state.model_type == "K-Means Clustering":
            st.subheader("Feature Selection for Clustering:")
            available_features = st.session_state.preprocessed_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            selected_features = st.multiselect(
                "Select features to use for clustering:",
                available_features,
                default=available_features[:min(2, len(available_features))]
            )
            
            if len(selected_features) < 2:
                st.warning("⚠️ Please select at least two features for K-Means clustering.")
                return
            
            n_clusters = st.slider("Number of clusters:", min_value=2, max_value=10, value=3)
            binarize_target = False
            binarize_threshold = None
        
        # Apply feature engineering
        if st.button("Engineer Features", key="engineer_features_button"):
            with st.spinner("Engineering features..."):
                st.session_state.features = engineer_features(
                    st.session_state.preprocessed_data,
                    selected_features=selected_features,
                    target_column=st.session_state.target_column if st.session_state.model_type != "K-Means Clustering" else None,
                    model_type=st.session_state.model_type,
                    binarize_target=binarize_target,
                    binarize_threshold=binarize_threshold,
                    n_clusters=n_clusters if st.session_state.model_type == "K-Means Clustering" else None
                )
                
                # Display feature importance if applicable
                if st.session_state.model_type in ["Linear Regression", "Logistic Regression"]:
                    st.subheader("Engineered Features Overview:")
                    st.write(f"Target: {st.session_state.target_column}")
                    # Debug the selected_features variable
                    st.write("DEBUG - selected_features type:", type(selected_features))
                    st.write("DEBUG - selected_features:", selected_features)
                    
                    # Check if it's a multiselect widget return value
                    if isinstance(selected_features, list):
                        # Handle each item directly, ensuring conversion to string
                        feature_names_list = []
                        for feature in selected_features:
                            st.write("DEBUG - feature type:", type(feature))
                            st.write("DEBUG - feature value:", feature)
                            feature_names_list.append(str(feature))
                        
                        # Join the properly converted strings
                        feature_names_str = ', '.join(feature_names_list)
                        st.write(f"Features: {feature_names_str}")
                    
                    # Show correlation heatmap
                    st.subheader("Feature Correlations:")
                    fig = plot_feature_importance(st.session_state.preprocessed_data, selected_features)
                    st.plotly_chart(fig, use_container_width=True, key="feature_correlation")
                
                display_notification("success", "Feature engineering complete! Your features are now ready for the model.")
                st.session_state.step = 4
                st.rerun()
    else:
        st.error("⚠️ No preprocessed data available. Please go back to the preprocessing phase.")
        if st.button("Go Back to Preprocessing", key="back_to_preprocessing"):
            st.session_state.step = 2
            st.rerun()

def display_train_test_split():
    st.markdown("# 🔄 **Phase 4: Train/Test Split**")
    st.markdown("### Every heist needs a practice run before the real thing.")
    
    if st.session_state.features is not None:
        # Split ratio selection
        st.subheader("Train/Test Split Configuration:")
        test_size = st.slider("Select test data percentage:", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random seed for reproducibility:", 0, 100, 42)
        
        # Split the data
        if st.button("Split Data", key="split_data_button"):
            with st.spinner("Splitting data into training and testing sets..."):
                split_result = split_data(
                    st.session_state.features,
                    test_size=test_size,
                    random_state=random_state,
                    model_type=st.session_state.model_type,
                    target_column=st.session_state.target_column if st.session_state.model_type != "K-Means Clustering" else None
                )
                
                if st.session_state.model_type != "K-Means Clustering":
                    st.session_state.X_train = split_result['X_train']
                    st.session_state.X_test = split_result['X_test']
                    st.session_state.y_train = split_result['y_train']
                    st.session_state.y_test = split_result['y_test']
                else:
                    st.session_state.X_train = split_result['data']  # For clustering, we use all data
                
                # Visualize the split
                st.subheader("Train/Test Split Visualization:")
                fig = plot_train_test_split(test_size)
                st.plotly_chart(fig, use_container_width=True, key="train_test_split")
                
                # Show training data sample
                if st.session_state.model_type != "K-Means Clustering":
                    st.subheader("Training Data Sample:")
                    st.write(f"Training set shape: {st.session_state.X_train.shape}")
                    training_sample = pd.DataFrame(st.session_state.X_train)
                    training_sample['target'] = st.session_state.y_train
                    st.dataframe(training_sample.head())
                    
                    st.subheader("Testing Data Sample:")
                    st.write(f"Testing set shape: {st.session_state.X_test.shape}")
                    testing_sample = pd.DataFrame(st.session_state.X_test)
                    testing_sample['target'] = st.session_state.y_test
                    st.dataframe(testing_sample.head())
                else:
                    st.subheader("Clustering Data:")
                    st.write(f"Data shape: {st.session_state.X_train.shape}")
                    st.dataframe(pd.DataFrame(st.session_state.X_train).head())
                
                display_notification("success", "Data successfully split into training and testing sets!")
                st.session_state.step = 5
                st.rerun()
    else:
        st.error("⚠️ No feature data available. Please go back to the feature engineering phase.")
        if st.button("Go Back to Feature Engineering", key="back_to_feature_engineering"):
            st.session_state.step = 3
            st.rerun()

def display_model_training():
    st.markdown("# 🧠 **Phase 5: Model Training**")
    st.markdown("### Time to execute the plan and train our heist team.")
    
    # Show images related to machine learning
    col1, col2 = st.columns(2)
    with col1:
        show_image_with_caption(
            "https://pixabay.com/get/g571efce2335b4ee8d3cb623a6fc5650df7ff7445e14c07ac8de59b9d02403146887ca5aeabc9f5e7a3c07bfd5253e1d4ccef90d8570048bbe22a1654ca1b2551_1280.jpg",
            "Machine Learning - Our secret weapon"
        )
    with col2:
        show_image_with_caption(
            "https://pixabay.com/get/g09c0a246645d04bba533b0eabb9c9c7771b0aba3928a456df597d89b32015a26dc7f5d1cd4bb34f37bc09e8c82086b7001a97023e45b14b3e492c827e7747a63_1280.jpg",
            "The powerful algorithms we'll deploy"
        )
    
    if st.session_state.model_type == "K-Means Clustering":
        if st.session_state.X_train is not None:
            # Training options
            st.subheader("K-Means Clustering Configuration:")
            n_clusters = st.slider("Number of clusters:", min_value=2, max_value=10, value=3)
            max_iter = st.slider("Maximum iterations:", min_value=100, max_value=1000, value=300, step=100)
            
            # Train the model
            if st.button("Train Clustering Model", key="train_clustering_button"):
                with st.spinner("Training K-Means clustering model..."):
                    st.session_state.model, st.session_state.evaluation = train_model(
                        X_train=st.session_state.X_train,
                        y_train=None,  # No target for clustering
                        model_type=st.session_state.model_type,
                        params={
                            'n_clusters': n_clusters,
                            'max_iter': max_iter
                        }
                    )
                    
                    display_notification("success", f"K-Means clustering model trained with {n_clusters} clusters!")
                    st.session_state.step = 6
                    st.rerun()
        else:
            st.error("⚠️ No data available for clustering. Please go back to the previous phase.")
    
    else:  # Linear or Logistic Regression
        if st.session_state.X_train is not None and st.session_state.y_train is not None:
            # Training options
            st.subheader(f"{st.session_state.model_type} Configuration:")
            
            if st.session_state.model_type == "Linear Regression":
                fit_intercept = st.checkbox("Fit intercept", value=True)
                # Note: 'normalize' parameter was removed in scikit-learn 1.2
                
                params = {
                    'fit_intercept': fit_intercept
                }
            
            elif st.session_state.model_type == "Logistic Regression":
                C = st.slider("Regularization strength (C):", 0.01, 10.0, 1.0, 0.1)
                max_iter = st.slider("Maximum iterations:", 100, 1000, 100, 100)
                
                params = {
                    'C': C,
                    'max_iter': max_iter
                }
            
            # Train the model
            if st.button(f"Train {st.session_state.model_type} Model", key="train_model_button"):
                with st.spinner(f"Training {st.session_state.model_type} model..."):
                    st.session_state.model, st.session_state.evaluation = train_model(
                        X_train=st.session_state.X_train,
                        y_train=st.session_state.y_train,
                        model_type=st.session_state.model_type,
                        params=params
                    )
                    
                    display_notification("success", f"{st.session_state.model_type} model successfully trained!")
                    st.session_state.step = 6
                    st.rerun()
        else:
            st.error("⚠️ Training data not available. Please go back to the train/test split phase.")
    
    if st.button("Go Back to Train/Test Split", key="back_to_split"):
        st.session_state.step = 4
        st.rerun()

def display_evaluation():
    st.markdown("# 📊 **Phase 6: Model Evaluation**")
    st.markdown("### Let's assess how successful our heist will be.")
    
    if st.session_state.model is not None and st.session_state.evaluation is not None:
        # Display evaluation metrics based on model type
        if st.session_state.model_type == "K-Means Clustering":
            st.subheader("K-Means Clustering Evaluation:")
            st.write(f"Inertia (sum of squared distances): {st.session_state.evaluation['inertia']:.4f}")
            st.write(f"Silhouette Score: {st.session_state.evaluation['silhouette_score']:.4f}")
            
            # Display cluster centers
            st.subheader("Cluster Centers:")
            centers_df = pd.DataFrame(
                st.session_state.evaluation['cluster_centers'],
                columns=st.session_state.evaluation['feature_names']
            )
            st.dataframe(centers_df)
            
            # Visualize clusters
            st.subheader("Cluster Visualization:")
            fig = st.session_state.evaluation['cluster_plot']
            st.plotly_chart(fig, use_container_width=True)
            
        else:  # Linear or Logistic Regression
            st.subheader(f"{st.session_state.model_type} Evaluation:")
            
            # Plot evaluation metrics
            metrics_fig = plot_evaluation_metrics(st.session_state.evaluation, st.session_state.model_type)
            st.plotly_chart(metrics_fig, use_container_width=True, key="evaluation_metrics")
            
            if st.session_state.model_type == "Linear Regression":
                # Display feature importance
                st.subheader("Feature Importance:")
                feature_importance = pd.DataFrame({
                    'Feature': st.session_state.evaluation['feature_names'],
                    'Coefficient': st.session_state.evaluation['coefficients']
                }).sort_values('Coefficient', ascending=False)
                
                fig = px.bar(
                    feature_importance, 
                    x='Feature', 
                    y='Coefficient',
                    title="Feature Importance (Coefficient Values)",
                    color='Coefficient',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif st.session_state.model_type == "Logistic Regression":
                # Confusion matrix
                st.subheader("Confusion Matrix:")
                if 'confusion_matrix_plot' in st.session_state.evaluation:
                    cm_fig = st.session_state.evaluation['confusion_matrix_plot']
                    st.plotly_chart(cm_fig, use_container_width=True)
                elif 'confusion_matrix' in st.session_state.evaluation:
                    # If plot not available but raw matrix is, create a plot
                    cm = st.session_state.evaluation['confusion_matrix']
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
                    st.plotly_chart(cm_fig, use_container_width=True)
                else:
                    st.info("Confusion matrix data not available")
                
                # ROC curve
                st.subheader("ROC Curve:")
                if 'roc_curve_plot' in st.session_state.evaluation:
                    roc_fig = st.session_state.evaluation['roc_curve_plot']
                    st.plotly_chart(roc_fig, use_container_width=True)
                else:
                    st.info("ROC curve data not available")
        
        display_notification("info", "Model evaluation complete. Review the metrics to understand model performance.")
        
        if st.button("Proceed to Final Results", key="proceed_to_results"):
            st.session_state.step = 7
            st.rerun()
            
    else:
        st.error("⚠️ No model or evaluation data available. Please go back to the model training phase.")
        if st.button("Go Back to Model Training", key="back_to_training"):
            st.session_state.step = 5
            st.rerun()

def display_results():
    st.markdown("# 💰 **Phase 7: The Heist Results**")
    st.markdown("### Time to collect our rewards and analyze our success!")
    
    if st.session_state.model is not None:
        # Final results based on model type
        if st.session_state.model_type == "K-Means Clustering":
            st.subheader("Clustering Results Summary:")
            
            # Show cluster distribution
            cluster_dist = pd.DataFrame({
                'Cluster': range(st.session_state.model.n_clusters),
                'Count': np.bincount(st.session_state.model.labels_)
            })
            
            fig = px.pie(
                cluster_dist,
                values='Count',
                names='Cluster',
                title="Distribution of Data Points in Clusters",
                color_discrete_sequence=px.colors.sequential.Reds,
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Allow downloading cluster assignments
            cluster_assignments = pd.DataFrame({
                'Cluster': st.session_state.model.labels_
            })
            
            st.subheader("Cluster Assignments:")
            st.dataframe(cluster_assignments.head(10))
            
            # Download option
            csv_buffer = io.StringIO()
            cluster_assignments.to_csv(csv_buffer, index=False)
            csv_str = csv_buffer.getvalue()
            
            st.download_button(
                label="Download Cluster Assignments",
                data=csv_str,
                file_name="cluster_assignments.csv",
                mime="text/csv"
            )
            
        else:  # Linear or Logistic Regression
            st.subheader("Make Predictions with Your Model:")
            st.markdown("Enter values for the features to get predictions from your trained model:")
            
            # Create input fields for each feature
            input_data = {}
            for feature in st.session_state.evaluation['feature_names']:
                min_val = float(st.session_state.X_test[:, list(st.session_state.evaluation['feature_names']).index(feature)].min())
                max_val = float(st.session_state.X_test[:, list(st.session_state.evaluation['feature_names']).index(feature)].max())
                mean_val = float(st.session_state.X_test[:, list(st.session_state.evaluation['feature_names']).index(feature)].mean())
                
                input_data[feature] = st.slider(
                    f"{feature}:", 
                    min_val,
                    max_val,
                    mean_val
                )
            
            # Make prediction
            if st.button("Predict", key="predict_button"):
                # Create input array
                input_array = np.array([input_data[feature] for feature in st.session_state.evaluation['feature_names']]).reshape(1, -1)
                
                # Make prediction
                prediction = st.session_state.model.predict(input_array)[0]
                
                if st.session_state.model_type == "Linear Regression":
                    st.markdown(f"### Predicted {st.session_state.target_column}: **{prediction:.4f}**")
                else:  # Logistic Regression
                    prob = st.session_state.model.predict_proba(input_array)[0][1]
                    st.markdown(f"### Predicted Class: **{prediction}** (Probability: {prob:.4f})")
            
            # Plot actual vs predicted values for test data
            st.subheader("Model Performance Visualization:")
            
            result_fig = predict_and_visualize(
                st.session_state.model,
                st.session_state.X_test,
                st.session_state.y_test,
                st.session_state.model_type,
                st.session_state.evaluation['feature_names'],
                st.session_state.target_column
            )
            
            st.plotly_chart(result_fig, use_container_width=True)
            
        # Show completion GIF
        completion_gif = get_completion_gif()
        st.image(completion_gif, use_column_width=True)
        
        st.markdown("## 🎭 **Heist Completed Successfully!** 🎭")
        st.markdown("""
        Congratulations! You've successfully completed the Financial ML Heist. 
        You've infiltrated the data, applied sophisticated machine learning techniques, 
        and extracted valuable insights.
        
        ### What's Next?
        - Download your results
        - Try different models or parameters
        - Apply your newfound skills to other financial datasets
        
        ### Remember: With great power comes great responsibility!
        """)
        
        # Offer to start over
        if st.button("Start a New Heist", key="start_over"):
            # Reset session state
            for key in st.session_state.keys():
                if key != 'theme':  # Keep theme settings
                    del st.session_state[key]
            st.session_state.step = 0
            st.rerun()
    else:
        st.error("⚠️ No model available. Please go back to the model training phase.")
        if st.button("Go Back to Evaluation", key="back_to_evaluation"):
            st.session_state.step = 6
            st.rerun()

# Display the appropriate page based on the current step
def main():
    if st.session_state.step == 0:
        display_welcome()
    elif st.session_state.step == 1:
        display_load_data()
    elif st.session_state.step == 2:
        display_preprocessing()
    elif st.session_state.step == 3:
        display_feature_engineering()
    elif st.session_state.step == 4:
        display_train_test_split()
    elif st.session_state.step == 5:
        display_model_training()
    elif st.session_state.step == 6:
        display_evaluation()
    elif st.session_state.step == 7:
        display_results()

if __name__ == "__main__":
    main()
