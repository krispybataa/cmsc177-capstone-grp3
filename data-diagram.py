from graphviz import Digraph

# Create a directed graph with left-to-right flow
dot = Digraph('GallstonePipeline', format='png')

# Set left-to-right layout and larger square size
dot.attr(rankdir='LR')
dot.attr(size='10,10')  # width=10in, height=10in max
dot.attr(dpi='300')     # 300 dpi for high resolution


# Define node styles for each step category
styles = {
    'raw_data': {'shape': 'cylinder', 'style': 'filled', 'fillcolor': '#4CAF50', 'fontcolor': 'white'},        # Green
    'eda': {'shape': 'rectangle', 'style': 'rounded,filled', 'fillcolor': '#6A1B9A', 'fontcolor': 'white'},     # Purple
    'preprocessing': {'shape': 'rectangle', 'style': 'rounded,filled', 'fillcolor': '#5D4037', 'fontcolor': 'white'}, # Brown
    'modeling': {'shape': 'rectangle', 'style': 'rounded,filled', 'fillcolor': '#1976D2', 'fontcolor': 'white'}, # Blue
    'evaluation': {'shape': 'rectangle', 'style': 'rounded,filled', 'fillcolor': '#FBC02D', 'fontcolor': 'black'},# Yellow
    'xai': {'shape': 'rectangle', 'style': 'rounded,filled', 'fillcolor': '#D32F2F', 'fontcolor': 'white'}       # Red
}

def add_node(dot, node_id, label, style_key):
    dot.node(node_id, label=label, **styles[style_key])

# Add nodes with bullet points (use \n and • for bullets)
add_node(dot, 'A', 'Raw Gallstone Data', 'raw_data')

add_node(dot, 'B', 'Exploratory Data Analysis\n• Check for missing values\n• Visualize distributions (histograms, boxplots)\n• Correlation heatmaps\n• Analyze class imbalance', 'eda')

add_node(dot, 'C', 'Data Preprocessing\n• Train-test split\n• Handle missing values\n• Encode categorical variables\n• Scale numerical features\n• Address class imbalance (SMOTE)\n• Feature selection (ANOVA F-score)', 'preprocessing')

add_node(dot, 'D', 'Machine Learning Models\n• Random Forest\n• Gradient Boosting\n• Support Vector Machine\n• Logistic Regression\n• AdaBoost\n• Decision Tree\n• Bagging Classifier', 'modeling')

add_node(dot, 'E', 'Model Evaluation\n• Accuracy, Precision, Recall, F1-score, AUC\n• Confusion matrices ', 'evaluation')

add_node(dot, 'F', 'Explainable AI (XAI)\n• SHAP \n• LIME', 'xai')

# Connect nodes with arrows
dot.edges(['AB', 'BC', 'CD', 'DE', 'EF'])

# Save and render
output_path = dot.render(filename='gallstone_pipeline', cleanup=True)
print(f"Diagram saved as: {output_path}")
