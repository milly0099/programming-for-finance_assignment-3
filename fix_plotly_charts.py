#!/usr/bin/env python3

with open('app.py', 'r') as file:
    content = file.read()

# Add unique keys to plotly_chart calls
replacements = [
    ('fig = plot_feature_importance(st.session_state.preprocessed_data, selected_features)\n                    st.plotly_chart(fig, use_container_width=True)', 
     'fig = plot_feature_importance(st.session_state.preprocessed_data, selected_features)\n                    st.plotly_chart(fig, use_container_width=True, key="feature_correlation")'),
    
    ('fig = plot_train_test_split(test_size)\n                st.plotly_chart(fig, use_container_width=True)', 
     'fig = plot_train_test_split(test_size)\n                st.plotly_chart(fig, use_container_width=True, key="train_test_split")'),
    
    ('fig = plot_dataset_overview(st.session_state.model_data)\n            st.plotly_chart(fig, use_container_width=True)', 
     'fig = plot_dataset_overview(st.session_state.model_data)\n            st.plotly_chart(fig, use_container_width=True, key="model_data_overview")'),
    
    ('metrics_fig = plot_evaluation_metrics(st.session_state.evaluation, st.session_state.model_type)\n            st.plotly_chart(metrics_fig, use_container_width=True)', 
     'metrics_fig = plot_evaluation_metrics(st.session_state.evaluation, st.session_state.model_type)\n            st.plotly_chart(metrics_fig, use_container_width=True, key="evaluation_metrics")'),
    
    ('fig = st.session_state.evaluation.get("fig")\n                st.plotly_chart(fig, use_container_width=True)', 
     'fig = st.session_state.evaluation.get("fig")\n                st.plotly_chart(fig, use_container_width=True, key="eval_scatter_plot")'),
    
    ('cm_fig = st.session_state.evaluation.get("cm_fig")\n                st.plotly_chart(cm_fig, use_container_width=True)', 
     'cm_fig = st.session_state.evaluation.get("cm_fig")\n                st.plotly_chart(cm_fig, use_container_width=True, key="confusion_matrix")'),
    
    ('roc_fig = st.session_state.evaluation.get("roc_fig")\n                st.plotly_chart(roc_fig, use_container_width=True)', 
     'roc_fig = st.session_state.evaluation.get("roc_fig")\n                st.plotly_chart(roc_fig, use_container_width=True, key="roc_curve")'),
    
    ('fig = predict_and_visualize(st.session_state.model, st.session_state.X_test, st.session_state.y_test, st.session_state.model_type, st.session_state.features_metadata["feature_names"], st.session_state.target_column)\n            st.plotly_chart(fig, use_container_width=True)', 
     'fig = predict_and_visualize(st.session_state.model, st.session_state.X_test, st.session_state.y_test, st.session_state.model_type, st.session_state.features_metadata["feature_names"], st.session_state.target_column)\n            st.plotly_chart(fig, use_container_width=True, key="prediction_visualization")'),
    
    ('result_fig = st.session_state.results_fig\n            st.plotly_chart(result_fig, use_container_width=True)', 
     'result_fig = st.session_state.results_fig\n            st.plotly_chart(result_fig, use_container_width=True, key="results_visualization")')
]

# Apply all replacements
for old, new in replacements:
    content = content.replace(old, new)

with open('app.py', 'w') as file:
    file.write(content)

print("Fixed plotly_chart calls by adding unique keys")
