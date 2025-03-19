import os
import time
import logging
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import List, Dict, Optional, Union
import traceback
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import re

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("data_agents")

# Load environment variables
load_dotenv()

# Disable OpenTelemetry globally to prevent connection errors
os.environ["OTEL_SDK_DISABLED"] = "true"

# Set up OpenAI since logs show it's already being used
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Configure rate limiting to prevent overwhelming the server
def rate_limit():
    """Simple rate limiting function that pauses execution briefly"""
    time.sleep(0.5)  # 500ms pause between operations

# Data analysis helpers
def analyze_column_data_type(df, column):
    """Analyze column to determine its data type and characteristics"""
    if pd.api.types.is_numeric_dtype(df[column]):
        unique_count = df[column].nunique()
        if unique_count <= 10:
            return "categorical_numeric"
        return "continuous_numeric"
    elif pd.api.types.is_string_dtype(df[column]):
        unique_count = df[column].nunique()
        if unique_count <= 20:
            return "categorical_text"
        return "text"
    elif pd.api.types.is_datetime64_dtype(df[column]):
        return "datetime"
    elif pd.api.types.is_bool_dtype(df[column]):
        return "boolean"
    else:
        return "unknown"

def infer_column_meaning(col_name, sample_values):
    """Infer semantic meaning of column based on name and values"""
    col_name = col_name.lower()
    
    # Common semantic patterns
    patterns = {
        'id': ['id', 'identifier', 'key'],
        'date': ['date', 'time', 'day', 'month', 'year', 'created', 'updated'],
        'price': ['price', 'cost', 'fee', 'amount', '$', 'dollar', 'eur', 'gbp'],
        'name': ['name', 'title', 'label'],
        'category': ['category', 'type', 'group', 'class', 'segment'],
        'location': ['location', 'address', 'city', 'country', 'state', 'zip', 'postal'],
        'email': ['email', 'mail', 'e-mail'],
        'phone': ['phone', 'telephone', 'mobile', 'cell'],
        'boolean': ['is_', 'has_', 'flag', 'active', 'enabled', 'status'],
        'score': ['score', 'rating', 'rank', 'grade'],
        'percentage': ['percent', 'rate', 'ratio'],
        'count': ['count', 'total', 'num', 'number'],
        'weight': ['weight', 'mass', 'kg', 'lb', 'gram'],
        'height': ['height', 'tall'],
        'width': ['width', 'broad'],
        'age': ['age', 'years'],
        'gender': ['gender', 'sex'],
        'description': ['description', 'desc', 'comment', 'note', 'detail'],
        'url': ['url', 'link', 'website', 'site'],
        'image': ['image', 'picture', 'photo', 'img'],
        'duration': ['duration', 'length', 'period', 'time'],
        'quantity': ['quantity', 'qty', 'amount', 'volume'],
        'size': ['size', 'dimension'],
        'color': ['color', 'colour'],
    }
    
    # Check for pattern matches in column name
    for meaning, keywords in patterns.items():
        if any(keyword in col_name for keyword in keywords):
            return meaning
            
    # If no patterns match, return general category based on column name
    return "numeric" if re.search(r'\d', col_name) else "text"

def get_display_name(column, df=None, metadata_dict=None):
    """Get a human-readable display name for a column with enhanced meaning detection"""
    # First priority: Use provided metadata if available
    if metadata_dict:
        for key, value in metadata_dict.items():
            if str(column) == str(value):
                return value
    
    # Second priority: Analyze column data to infer meaning
    if df is not None:
        try:
            col_type = analyze_column_data_type(df, column)
            sample_values = df[column].dropna().head(10).tolist()
            semantic_meaning = infer_column_meaning(str(column), sample_values)
            
            # Return enhanced column name
            if col_type == "categorical_numeric" and semantic_meaning != "id":
                return f"{str(column).replace('_', ' ').title()} (Category)"
            elif col_type == "continuous_numeric":
                if semantic_meaning in ["price", "score", "percentage", "count"]:
                    return f"{str(column).replace('_', ' ').title()} ({semantic_meaning.title()})"
                return f"{str(column).replace('_', ' ').title()} (Numeric)"
            elif col_type == "categorical_text":
                return f"{str(column).replace('_', ' ').title()} (Category)"
            elif col_type == "datetime":
                return f"{str(column).replace('_', ' ').title()} (Date/Time)"
            elif col_type == "boolean":
                return f"{str(column).replace('_', ' ').title()} (Yes/No)"
        except:
            pass
    
    # Third priority: Clean up the column name
    col_str = str(column)
    
    # Remove underscores and capitalize words
    if '_' in col_str:
        return ' '.join(word.capitalize() for word in col_str.split('_'))
    
    # Convert CamelCase to spaces
    if any(c.isupper() for c in col_str) and not col_str.isupper():
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', col_str)
        return re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1)
    
    # If it's like "Column_1", make it more readable
    if col_str.startswith(('Column_', 'column_')):
        parts = col_str.split('_')
        if len(parts) == 2 and parts[1].isdigit():
            return f"Feature {parts[1]}"
    
    # Default: just return the original with first letter capitalized
    return col_str.capitalize()

def suggest_visualization_type(df, column):
    """Suggest appropriate visualization type based on column data"""
    col_type = analyze_column_data_type(df, column)
    
    if col_type == "categorical_numeric" or col_type == "categorical_text":
        unique_count = df[column].nunique()
        if unique_count <= 5:
            return "pie"
        return "bar"
    elif col_type == "continuous_numeric":
        return "histogram"
    elif col_type == "datetime":
        return "line"
    elif col_type == "boolean":
        return "count"
    else:
        return "histogram" if pd.api.types.is_numeric_dtype(df[column]) else "count"

@tool("Read and parse uploaded data file")
def read_data_file(file_path: str) -> str:
    """Reads and parses data files, preserving original column names and analyzing data types."""
    try:
        logger.info(f"Starting to parse file: {file_path}")
        rate_limit()
        
        # Enhanced file path validation
        if not file_path or len(file_path) < 2:
            logger.error(f"Invalid file path: {file_path}")
            return f"Error: Invalid file path: {file_path}"
            
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return f"Error: File not found at path: {file_path}"
            
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Store column mapping and metadata for visualization 
        column_metadata = {}
        column_data_types = {}
        data_insights = {}
        
        if ext == '.csv':
            # Try to intelligently detect headers
            try:
                # First try with pandas auto-detection
                df = pd.read_csv(file_path)
                logger.info(f"CSV file parsed with automatic header detection")
                
                # Store original column names and detect data types
                for i, col_name in enumerate(df.columns):
                    column_metadata[f"col_{i}"] = str(col_name)
                    column_data_types[str(col_name)] = analyze_column_data_type(df, col_name)
                
            except Exception as csv_error:
                logger.warning(f"Error with automatic header detection: {str(csv_error)}")
                # Try again with explicit no header
                df = pd.read_csv(file_path, header=None)
                # Generate default column names
                column_names = [f'Column_{i+1}' for i in range(df.shape[1])]
                df.columns = column_names
                
                # Store column mapping
                for i, col_name in enumerate(column_names):
                    column_metadata[f"col_{i}"] = col_name
                    column_data_types[col_name] = analyze_column_data_type(df, col_name)
                
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            # Store original column names
            for i, col_name in enumerate(df.columns):
                column_metadata[f"col_{i}"] = str(col_name)
                column_data_types[str(col_name)] = analyze_column_data_type(df, col_name)
                
        elif ext == '.json':
            df = pd.read_json(file_path)
            # Store original column names
            for i, col_name in enumerate(df.columns):
                column_metadata[f"col_{i}"] = str(col_name)
                column_data_types[str(col_name)] = analyze_column_data_type(df, col_name)
        else:
            return f"Unsupported file format: {ext}"
        
        # Quick data insights
        data_insights["row_count"] = df.shape[0]
        data_insights["column_count"] = df.shape[1]
        data_insights["missing_values"] = df.isnull().sum().sum()
        data_insights["numeric_columns"] = len(df.select_dtypes(include=['number']).columns)
        data_insights["categorical_columns"] = len(df.select_dtypes(include=['object', 'category']).columns)
        
        # Save DataFrame as pickle for later use
        pickle_path = f"{file_path}.pkl"
        df.to_pickle(pickle_path)
        
        # Save column metadata and insights
        metadata_path = f"{file_path}.meta.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                "column_metadata": column_metadata,
                "column_data_types": column_data_types,
                "data_insights": data_insights
            }, f)
        
        # Generate enhanced summary
        summary = f"File successfully parsed. Found {df.shape[0]} rows and {df.shape[1]} columns.\n\n"
        
        # Column Names with data type insights
        summary += "Column Names:\n"
        for col in df.columns:
            col_type = column_data_types.get(str(col), "unknown")
            summary += f"- {col} ({col_type})\n"
        summary += "\n"
        
        # Data sample
        summary += "Data Sample (first 5 rows):\n"
        summary += df.head().to_string() + "\n\n"
        
        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary += "Basic Statistics:\n"
            summary += df[numeric_cols].describe().to_string()
        else:
            summary += "No numeric columns found for statistics."
            
        return summary
        
    except Exception as e:
        logger.error(f"Error parsing file: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error parsing file: {str(e)}"

@tool("Generate exploratory visualizations")
def generate_visualizations(file_path: str) -> List[Dict[str, str]]:
    """Generates intelligent visualizations with meaningful titles and descriptions based on data analysis."""
    try:
        logger.info(f"Starting visualization generation for: {file_path}")
        rate_limit()
        
        # Enhanced file path validation
        if not file_path or len(file_path) < 2:
            logger.error(f"Invalid file path: {file_path}")
            return [{"error": f"Error: Invalid file path: {file_path}"}]
            
        # Load the data
        pickle_path = f"{file_path}.pkl"
        if not os.path.exists(pickle_path):
            logger.error(f"Pickle file not found: {pickle_path}")
            return [{"error": "Processed data file not found. Please parse the file first."}]
            
        df = pd.read_pickle(pickle_path)
        
        # Load column metadata if available
        metadata_path = f"{file_path}.meta.json"
        column_metadata = {}
        column_data_types = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    column_metadata = metadata.get("column_metadata", {})
                    column_data_types = metadata.get("column_data_types", {})
            except:
                logger.warning("Could not load column metadata, using raw column names")
        
        # Extract dataset name from file path
        dataset_name = os.path.basename(file_path).split('.')[0]
        
        # Create directory for visualizations
        viz_dir = os.path.join('public', 'visualizations', dataset_name)
        os.makedirs(viz_dir, exist_ok=True)
        
        visualization_metadata = []
        
        # 1. Distribution of numeric columns with intelligent naming
        numeric_cols = df.select_dtypes(include=['number']).columns[:5]
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                try:
                    plt.figure(figsize=(10, 6))
                    
                    # Determine best visualization type
                    viz_type = suggest_visualization_type(df, col)
                    
                    # Get meaningful display name
                    display_name = get_display_name(col, df, column_metadata)
                    
                    # Generate appropriate visualization
                    if viz_type == "histogram":
                        sns.histplot(df[col].dropna(), kde=True)
                        title = f'Distribution of {display_name}'
                        desc = f"Shows the frequency distribution of {display_name} values"
                    elif viz_type == "bar":
                        value_counts = df[col].value_counts().head(10)
                        sns.barplot(x=value_counts.index, y=value_counts.values)
                        title = f'Most Common {display_name} Values'
                        desc = f"Shows the frequency of the top {len(value_counts)} {display_name} values"
                    elif viz_type == "pie":
                        plt.pie(df[col].value_counts(), labels=df[col].value_counts().index, autopct='%1.1f%%')
                        plt.axis('equal')
                        title = f'{display_name} Distribution'
                        desc = f"Shows the percentage breakdown of different {display_name} values"
                    else:
                        sns.histplot(df[col].dropna(), kde=True)
                        title = f'Distribution of {display_name}'
                        desc = f"Shows the frequency distribution of {display_name} values"
                    
                    plt.title(title)
                    plt.tight_layout()
                    
                    # Use a URL-safe filename
                    safe_col_name = str(col).replace(' ', '_').replace('/', '_').lower()
                    filename = f'dist_{safe_col_name}.png'
                    chart_path = os.path.join(viz_dir, filename)
                    plt.savefig(chart_path)
                    plt.close()
                    
                    visualization_metadata.append({
                        "path": f'/static/visualizations/{dataset_name}/{filename}',
                        "title": title,
                        "description": desc,
                        "type": viz_type,
                        "column": str(col)
                    })
                except Exception as e:
                    logger.error(f"Error generating distribution for {col}: {str(e)}")
        
        # 2. Categorical data with intelligent visualization
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns[:3]
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                try:
                    # Get meaningful display name
                    display_name = get_display_name(col, df, column_metadata)
                    
                    # Get value counts but limit to top categories
                    value_counts = df[col].value_counts().head(10)
                    
                    # Choose visualization based on number of categories
                    if len(value_counts) <= 5:
                        # Pie chart for few categories
                        plt.figure(figsize=(10, 6))
                        plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
                        plt.axis('equal')
                        title = f'{display_name} Distribution'
                        desc = f"Shows the percentage breakdown of different {display_name} categories"
                        viz_type = "pie"
                    else:
                        # Bar chart for more categories
                        plt.figure(figsize=(12, 6))
                        sns.barplot(x=value_counts.index, y=value_counts.values)
                        plt.xticks(rotation=45)
                        title = f'Count of {display_name} Values'
                        desc = f"Shows the frequency of the top {len(value_counts)} {display_name} categories"
                        viz_type = "bar"
                    
                    plt.title(title)
                    plt.tight_layout()
                    
                    safe_col_name = str(col).replace(' ', '_').replace('/', '_').lower()
                    filename = f'count_{safe_col_name}.png'
                    chart_path = os.path.join(viz_dir, filename)
                    plt.savefig(chart_path)
                    plt.close()
                    
                    visualization_metadata.append({
                        "path": f'/static/visualizations/{dataset_name}/{filename}',
                        "title": title,
                        "description": desc,
                        "type": viz_type,
                        "column": str(col)
                    })
                except Exception as e:
                    logger.error(f"Error generating categorical chart for {col}: {str(e)}")
        
        # 3. Correlation heatmap with enhanced readability
        if len(numeric_cols) >= 2:
            try:
                plt.figure(figsize=(12, 10))
                corr_matrix = df[numeric_cols].corr()
                
                # Create a mask for the upper triangle
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                
                # Generate a custom diverging colormap
                cmap = sns.diverging_palette(230, 20, as_cmap=True)
                
                # Draw the heatmap with better annotation format
                sns.heatmap(
                    corr_matrix, 
                    mask=mask,
                    annot=True, 
                    fmt=".2f",
                    cmap=cmap,
                    center=0,
                    square=True, 
                    linewidths=.5,
                    cbar_kws={"shrink": .8}
                )
                
                # Use human-readable column labels for better understanding
                display_labels = [get_display_name(col, df, column_metadata) for col in numeric_cols]
                plt.xticks(ticks=np.arange(len(numeric_cols)) + 0.5, labels=display_labels, rotation=45, ha="right")
                plt.yticks(ticks=np.arange(len(numeric_cols)) + 0.5, labels=display_labels, rotation=0)
                
                plt.title('Feature Correlation Heatmap')
                plt.tight_layout()
                
                filename = 'correlation_heatmap.png'
                chart_path = os.path.join(viz_dir, filename)
                plt.savefig(chart_path)
                plt.close()
                
                visualization_metadata.append({
                    "path": f'/static/visualizations/{dataset_name}/{filename}',
                    "title": 'Feature Correlation Heatmap',
                    "description": "Shows how numeric features relate to each other. Values close to 1 or -1 indicate strong correlation.",
                    "type": "correlation",
                    "columns": [str(col) for col in numeric_cols.tolist()]
                })
            except Exception as e:
                logger.error(f"Error generating correlation heatmap: {str(e)}")
        
        # 4. Generate PCA plot for high-dimensional data
        if len(numeric_cols) >= 3:
            try:
                # Select only numeric columns without missing values
                numeric_df = df[numeric_cols].dropna()
                
                if len(numeric_df) > 10:  # Ensure we have enough data points
                    # Standardize the data
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(numeric_df)
                    
                    # Apply PCA
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(scaled_data)
                    
                    # Create PCA plot
                    plt.figure(figsize=(10, 8))
                    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
                    
                    # Add axis labels with explained variance
                    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                    
                    plt.title('PCA Visualization of Numeric Features')
                    plt.grid(True, linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    
                    filename = 'pca_visualization.png'
                    chart_path = os.path.join(viz_dir, filename)
                    plt.savefig(chart_path)
                    plt.close()
                    
                    visualization_metadata.append({
                        "path": f'/static/visualizations/{dataset_name}/{filename}',
                        "title": 'Principal Component Analysis',
                        "description": "2D projection of high-dimensional numeric data showing patterns and clusters",
                        "type": "pca",
                        "columns": [str(col) for col in numeric_cols.tolist()]
                    })
            except Exception as e:
                logger.error(f"Error generating PCA visualization: {str(e)}")
                
        # 5. Add a data overview dashboard if we have enough visualizations
        if len(visualization_metadata) >= 3:
            try:
                # Create a dashboard with key stats and mini-charts
                plt.figure(figsize=(14, 8))
                
                # Set up a 2x2 grid
                plt.subplot(2, 2, 1)
                
                # Show basic dataset info
                plt.axis('off')
                info_text = f"Dataset Overview: {dataset_name}\n"
                info_text += f"Rows: {df.shape[0]}\n"
                info_text += f"Columns: {df.shape[1]}\n"
                info_text += f"Numeric Features: {len(df.select_dtypes(include=['number']).columns)}\n"
                info_text += f"Categorical Features: {len(df.select_dtypes(include=['object', 'category']).columns)}\n"
                info_text += f"Missing Values: {df.isnull().sum().sum()}"
                
                plt.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center')
                plt.title("Dataset Summary")
                
                # Add a small sample of the distribution chart in subplot 2
                if len(numeric_cols) > 0:
                    plt.subplot(2, 2, 2)
                    col = numeric_cols[0]
                    sns.histplot(df[col].dropna(), kde=True)
                    plt.title(f'{get_display_name(col, df, column_metadata)} Distribution')
                
                # Add a small sample of the categorical chart in subplot 3
                if len(categorical_cols) > 0:
                    plt.subplot(2, 2, 3)
                    col = categorical_cols[0]
                    value_counts = df[col].value_counts().head(5)
                    plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
                    plt.axis('equal')
                    plt.title(f'Top {get_display_name(col, df, column_metadata)} Categories')
                
                # Add correlation snippet
                if len(numeric_cols) >= 2:
                    plt.subplot(2, 2, 4)
                    corr_matrix = df[numeric_cols[:3]].corr()
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
                    plt.title("Feature Correlations")
                
                plt.tight_layout()
                filename = 'dashboard_overview.png'
                chart_path = os.path.join(viz_dir, filename)
                plt.savefig(chart_path)
                plt.close()
                
                visualization_metadata.append({
                    "path": f'/static/visualizations/{dataset_name}/{filename}',
                    "title": 'Dashboard Overview',
                    "description": "Summary visualization of key dataset characteristics",
                    "type": "dashboard",
                    "is_overview": True
                })
            except Exception as e:
                logger.error(f"Error generating dashboard overview: {str(e)}")
            
        return visualization_metadata
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        logger.error(traceback.format_exc())
        return [{"error": f"Error generating visualizations: {str(e)}"}]

@tool("Query data using natural language")
def query_data(file_path: str, query: str) -> Dict:
    """
    Processes a natural language query about the data and returns relevant information.
    Generates intelligent visualizations based on the query.
    
    Args:
        file_path: Path to the data file
        query: Natural language query about the data
        
    Returns:
        Dict: Contains the answer and any relevant visualizations
    """
    try:
        logger.info(f"Processing query: '{query}' for file: {file_path}")
        rate_limit()  # Add rate limiting
        
        # Enhanced file path validation
        if not file_path or len(file_path) < 2:
            logger.error(f"Invalid file path: {file_path}")
            return {"answer": f"Error: Invalid file path: {file_path}"}
            
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {"answer": f"Error: File not found at {file_path}"}
        
        # Load the data
        pickle_path = f"{file_path}.pkl"
        if not os.path.exists(pickle_path):
            logger.error(f"Pickle file not found: {pickle_path}")
            return {"answer": "Error: Processed data file not found. Please parse the file first."}
            
        df = pd.read_pickle(pickle_path)
        logger.info(f"Loaded data from pickle with shape: {df.shape}")
        
        # Load metadata
        metadata_path = f"{file_path}.meta.json"
        column_metadata = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    column_metadata = metadata.get("column_metadata", {})
            except:
                logger.warning("Could not load column metadata")
        
        # Create viz directory if needed
        dataset_name = os.path.basename(file_path).split('.')[0]
        viz_dir = os.path.join('public', 'visualizations', dataset_name)
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate response based on query type
        response = {"answer": "", "visualization": None}
        
        # Query preprocessing - convert to lowercase and remove punctuation
        query_text = query.lower()
        query_text = re.sub(r'[^\w\s]', ' ', query_text)
        
        # Handle different query types with intelligent visualization selection
        try:
            # 1. Basic data shape query
            if any(keyword in query_text for keyword in [" shape", " size", " rows", " columns", "how many rows", "how many columns"]):
                rows, cols = df.shape
                response["answer"] = f"The dataset has {rows} rows and {cols} columns."
                
                # Add a small data overview visualization
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, f"Dataset Structure:\nRows: {rows}\nColumns: {cols}", 
                        horizontalalignment='center', verticalalignment='center', fontsize=14)
                plt.axis('off')
                
                filename = f'data_structure.png'
                chart_path = os.path.join(viz_dir, filename)
                plt.savefig(chart_path)
                plt.close()
                
                response["visualization"] = f'/static/visualizations/{dataset_name}/{filename}'
                
            # 2. Column list query
            elif any(keyword in query_text for keyword in [" columns", " fields", " variables", " features", "what columns", "which columns"]):
                col_list = ", ".join([f"{col} ({get_display_name(col, df, column_metadata)})" 
                                     if get_display_name(col, df, column_metadata) != col else col 
                                     for col in df.columns])
                response["answer"] = f"The dataset contains the following columns: {col_list}"
                
                # Create a table visualization of columns and their types
                plt.figure(figsize=(12, max(6, len(df.columns) * 0.3)))
                plt.axis('off')
                
                col_types = {col: str(df[col].dtype) for col in df.columns}
                
                # Create table data
                table_data = []
                for i, col in enumerate(df.columns):
                    display_name = get_display_name(col, df, column_metadata)
                    col_type = col_types[col]
                    non_null = df[col].count()
                    null_percent = (1 - non_null / len(df)) * 100
                    unique_count = df[col].nunique()
                    
                    table_data.append([i+1, col, display_name, col_type, 
                                      f"{non_null}/{len(df)} ({null_percent:.1f}% missing)",
                                      unique_count])
                
                # Create the table
                table = plt.table(
                    cellText=table_data,
                    colLabels=['#', 'Column Name', 'Display Name', 'Data Type', 'Non-Null Count', 'Unique Values'],
                    loc='center',
                    cellLoc='center',
                    colWidths=[0.05, 0.25, 0.25, 0.15, 0.2, 0.1]
                )
                
                # Adjust table appearance
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.5)
                
                plt.title("Dataset Columns Overview", fontsize=14, pad=20)
                
                filename = f'column_overview.png'
                chart_path = os.path.join(viz_dir, filename)
                plt.savefig(chart_path, bbox_inches='tight')
                plt.close()
                
                response["visualization"] = f'/static/visualizations/{dataset_name}/{filename}'
                
            # 3. Summary statistics query
            elif any(keyword in query_text for keyword in [" summary", " statistics", " describe", "statistical summary"]):
                # Create a simplified description focusing on numerical columns
                numeric_cols = df.select_dtypes(include=['number']).columns
                
                if len(numeric_cols) > 0:
                    num_description = df[numeric_cols].describe().to_string()
                    response["answer"] = f"Here's a statistical summary of the numerical columns:\n\n{num_description}"
                    
                    # Create visualization summary of statistics
                    plt.figure(figsize=(14, 10))
                    
                    # For each numeric column, create a small subplot with distribution
                    for i, col in enumerate(numeric_cols[:min(6, len(numeric_cols))]):
                        plt.subplot(2, 3, i+1)
                        sns.histplot(df[col].dropna(), kde=True)
                        plt.title(get_display_name(col, df, column_metadata))
                        
                        # Add key statistics as text
                        stats = df[col].describe()
                        stat_text = f"Mean: {stats['mean']:.2f}\n"
                        stat_text += f"Std: {stats['std']:.2f}\n"
                        stat_text += f"Min: {stats['min']:.2f}\n"
                        stat_text += f"Max: {stats['max']:.2f}"
                        
                        plt.annotate(stat_text, xy=(0.7, 0.7), xycoords='axes fraction', 
                                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                    
                    plt.tight_layout()
                    filename = f'statistical_summary.png'
                    chart_path = os.path.join(viz_dir, filename)
                    plt.savefig(chart_path)
                    plt.close()
                    
                    response["visualization"] = f'/static/visualizations/{dataset_name}/{filename}'
                else:
                    response["answer"] = "The dataset does not contain any numerical columns for statistical analysis."
                
            # 4. Distribution query - more intelligent column detection
            elif "distribution" in query_text:
                # Find which column to analyze - more robust column matching
                matched_column = None
                highest_match_score = 0
                
                for col in df.columns:
                    # Try different representations of the column name
                    col_variations = [
                        str(col).lower(),
                        get_display_name(col, df, column_metadata).lower(),
                        str(col).lower().replace('_', ' '),
                        str(col).lower().replace(' ', '')
                    ]
                    
                    # Check if any variation appears in the query
                    for variation in col_variations:
                        if variation in query_text:
                            # Compute a match score based on specificity
                            match_score = len(variation) / (1 + query_text.count(variation))
                            if match_score > highest_match_score:
                                highest_match_score = match_score
                                matched_column = col
                
                if matched_column:
                    col = matched_column
                    display_name = get_display_name(col, df, column_metadata)
                    
                    if pd.api.types.is_numeric_dtype(df[col]):
                        try:
                            plt.figure(figsize=(12, 8))
                            
                            # Create a more informative distribution visualization
                            ax = plt.subplot(111)
                            sns.histplot(df[col].dropna(), kde=True, ax=ax)
                            
                            # Add vertical lines for key statistics
                            mean_val = df[col].mean()
                            median_val = df[col].median()
                            
                            plt.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, 
                                        label=f'Mean: {mean_val:.2f}')
                            plt.axvline(median_val, color='green', linestyle='--', linewidth=1.5, 
                                      label=f'Median: {median_val:.2f}')
                            
                            # Add percentile markers
                            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                            plt.axvline(q1, color='orange', linestyle='--', linewidth=1, 
                                      label=f'25th: {q1:.2f}')
                            plt.axvline(q3, color='purple', linestyle='--', linewidth=1, 
                                      label=f'75th: {q3:.2f}')
                            
                            plt.legend()
                            plt.title(f'Distribution of {display_name}')
                            
                            # Add statistics summary as text box
                            stats = df[col].describe()
                            stat_text = (f"Count: {stats['count']}\n"
                                        f"Mean: {stats['mean']:.2f}\n"
                                        f"Std Dev: {stats['std']:.2f}\n"
                                        f"Min: {stats['min']:.2f}\n"
                                        f"25%: {stats['25%']:.2f}\n"
                                        f"50%: {stats['50%']:.2f}\n"
                                        f"75%: {stats['75%']:.2f}\n"
                                        f"Max: {stats['max']:.2f}")
                            
                            plt.annotate(stat_text, xy=(0.02, 0.97), xycoords='axes fraction', 
                                        va='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                            
                            plt.tight_layout()
                            filename = f'query_dist_{col}.png'
                            chart_path = os.path.join(viz_dir, filename)
                            plt.savefig(chart_path)
                            plt.close()
                            
                            visualization_path = f'/static/visualizations/{dataset_name}/{filename}'
                            response["answer"] = (f"Here's the distribution of {display_name}. "
                                                f"The mean is {mean_val:.2f} and the median is {median_val:.2f}. "
                                                f"The standard deviation is {df[col].std():.2f}.")
                            response["visualization"] = visualization_path
                        except Exception as viz_error:
                            logger.error(f"Error creating visualization: {str(viz_error)}")
                    else:
                        try:
                            plt.figure(figsize=(12, 8))
                            
                            # For categorical, create enhanced visualization
                            value_counts = df[col].value_counts().head(10)
                            total_count = len(df)
                            
                            # Create main bar chart
                            ax = plt.subplot(111)
                            bars = ax.bar(value_counts.index, value_counts.values, color='skyblue')
                            
                            # Add percentage labels on top of bars
                            for bar in bars:
                                height = bar.get_height()
                                percentage = height / total_count * 100
                                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                        f'{percentage:.1f}%', ha='center', va='bottom')
                            
                            plt.title(f'Distribution of {display_name} (Top 10 Categories)')
                            plt.xticks(rotation=45, ha='right')
                            plt.ylabel('Count')
                            
                            # Add totals
                            plt.annotate(f"Total entries: {total_count}\nShowing top {len(value_counts)} of {df[col].nunique()} categories", 
                                        xy=(0.02, 0.95), xycoords='axes fraction', 
                                        va='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                            
                            plt.tight_layout()
                            filename = f'query_dist_{col}.png'
                            chart_path = os.path.join(viz_dir, filename)
                            plt.savefig(chart_path)
                            plt.close()
                            
                            visualization_path = f'/static/visualizations/{dataset_name}/{filename}'
                            response["answer"] = (f"Here's the distribution of values for {display_name}. "
                                                f"The most common value is '{value_counts.index[0]}' "
                                                f"which appears {value_counts.values[0]} times "
                                                f"({value_counts.values[0]/total_count:.1%} of all records).")
                            response["visualization"] = visualization_path
                        except Exception as viz_error:
                            logger.error(f"Error creating visualization: {str(viz_error)}")
                else:
                    response["answer"] = "I couldn't identify which column you want to see the distribution for. Please specify a column name in your query."
                    
            # 5. Correlation query
            elif any(keyword in query_text for keyword in ["correlation", "relationship", "related", "correlate"]):
                numeric_cols = df.select_dtypes(include=['number']).columns
                
                if len(numeric_cols) >= 2:
                    # For general correlation query, show heatmap
                    if not any(col.lower() in query_text for col in df.columns.astype(str)):
                        try:
                            plt.figure(figsize=(12, 10))
                            corr_matrix = df[numeric_cols].corr()
                            
                            # Create a mask for the upper triangle
                            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                            
                            # Generate a custom diverging colormap
                            cmap = sns.diverging_palette(230, 20, as_cmap=True)
                            
                            # Draw the heatmap with better annotation format
                            sns.heatmap(
                                corr_matrix, 
                                mask=mask,
                                annot=True, 
                                fmt=".2f",
                                cmap=cmap,
                                center=0,
                                square=True, 
                                linewidths=.5,
                                cbar_kws={"shrink": .8}
                            )
                            
                            # Use human-readable column labels for better understanding
                            display_labels = [get_display_name(col, df, column_metadata) for col in numeric_cols]
                            plt.xticks(ticks=np.arange(len(numeric_cols)) + 0.5, labels=display_labels, rotation=45, ha="right")
                            plt.yticks(ticks=np.arange(len(numeric_cols)) + 0.5, labels=display_labels, rotation=0)
                            
                            plt.title('Feature Correlation Heatmap')
                            plt.tight_layout()
                            
                            filename = 'query_correlation_heatmap.png'
                            chart_path = os.path.join(viz_dir, filename)
                            plt.savefig(chart_path)
                            plt.close()
                            
                            # Find strongest correlations for the answer
                            strongest_corr = []
                            for i in range(len(numeric_cols)):
                                for j in range(i+1, len(numeric_cols)):
                                    corr_value = corr_matrix.iloc[i, j]
                                    if abs(corr_value) > 0.5:  # Report correlations stronger than 0.5
                                        col1_name = get_display_name(numeric_cols[i], df, column_metadata)
                                        col2_name = get_display_name(numeric_cols[j], df, column_metadata)
                                        corr_type = "positive" if corr_value > 0 else "negative"
                                        strongest_corr.append((col1_name, col2_name, corr_value, corr_type))
                            
                            response["visualization"] = f'/static/visualizations/{dataset_name}/{filename}'
                            
                            if strongest_corr:
                                # Report top 3 strongest correlations
                                strongest_corr.sort(key=lambda x: abs(x[2]), reverse=True)
                                corr_text = "The strongest correlations are:\n"
                                for col1, col2, val, corr_type in strongest_corr[:3]:
                                    corr_text += f"- {col1} and {col2}: {val:.2f} ({corr_type} correlation)\n"
                                response["answer"] = corr_text
                            else:
                                response["answer"] = "Here's the correlation heatmap. No strong correlations (>0.5) were found between the numeric variables."
                                
                        except Exception as viz_error:
                            logger.error(f"Error creating correlation heatmap: {str(viz_error)}")
                    else:
                        # Try to find specific columns mentioned in query
                        col1 = None
                        col2 = None
                        
                        for col in df.columns:
                            col_str = str(col).lower()
                            display_name = get_display_name(col, df, column_metadata).lower()
                            
                            if col_str in query_text or display_name in query_text:
                                if col1 is None:
                                    col1 = col
                                elif col2 is None:
                                    col2 = col
                                    break
                        
                        if col1 is not None and col2 is not None:
                            # Check if both columns are numeric
                            if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                                try:
                                    plt.figure(figsize=(10, 8))
                                    
                                    # Create a scatter plot with regression line
                                    sns.regplot(x=df[col1], y=df[col2], scatter_kws={'alpha':0.5})
                                    
                                    col1_display = get_display_name(col1, df, column_metadata)
                                    col2_display = get_display_name(col2, df, column_metadata)
                                    
                                    plt.xlabel(col1_display)
                                    plt.ylabel(col2_display)
                                    plt.title(f'Relationship between {col1_display} and {col2_display}')
                                    
                                    # Calculate and add correlation coefficient
                                    corr_val = df[col1].corr(df[col2])
                                    corr_type = "positive" if corr_val > 0 else "negative"
                                    strength = "strong" if abs(corr_val) > 0.7 else "moderate" if abs(corr_val) > 0.3 else "weak"
                                    
                                    plt.annotate(f"Correlation: {corr_val:.2f}\n({strength} {corr_type})", 
                                                xy=(0.05, 0.95), xycoords='axes fraction', 
                                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                                    
                                    plt.tight_layout()
                                    filename = f'correlation_{col1}_{col2}.png'
                                    chart_path = os.path.join(viz_dir, filename)
                                    plt.savefig(chart_path)
                                    plt.close()
                                    
                                    response["visualization"] = f'/static/visualizations/{dataset_name}/{filename}'
                                    response["answer"] = (f"The correlation between {col1_display} and {col2_display} is {corr_val:.2f}, "
                                                        f"indicating a {strength} {corr_type} relationship. "
                                                        f"This means that as {col1_display} {'increases' if corr_val > 0 else 'decreases'}, "
                                                        f"{col2_display} tends to {'increase' if corr_val > 0 else 'decrease'} as well.")
                                    
                                except Exception as viz_error:
                                    logger.error(f"Error creating correlation visualization: {str(viz_error)}")
                            else:
                                response["answer"] = f"I can only show correlations between numeric columns, but one or both of your selected columns isn't numeric."
                        else:
                            response["answer"] = "I couldn't identify which columns you want to correlate. Please specify two column names in your query."
                else:
                    response["answer"] = "The dataset doesn't have enough numeric columns to compute correlations. At least two numeric columns are required."
        
        except Exception as query_error:
            logger.error(f"Error processing specific query type: {str(query_error)}")
            response["answer"] = f"I encountered an error processing your query: {str(query_error)}"
        
        # If no handlers triggered, give generic response
        if not response["answer"]:
            response["answer"] = ("I'm not sure how to answer this specific query. "
                                  "Try asking about data shape, columns, distributions, "
                                  "correlations, or statistical summaries.")
        
        logger.info(f"Query processing complete, response size: {len(response['answer'])}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        return {"answer": f"Error processing query: {str(e)}"}

# Define CrewAI agents with OpenAI as the LLM
data_reader = Agent(
    role="Data Reader",
    goal="Accurately parse and summarize uploaded data files",
    backstory=("You are an expert data engineer who can parse any file format and "
              "extract meaningful information. Your specialty is providing clear summaries "
              "of datasets to help analysts understand what they're working with."),
    tools=[read_data_file],
    llm_config={
        "config_list": [{"model": "gpt-4o-mini"}],
        "temperature": 0.1,
        "max_tokens": 4000
    },
    allow_delegation=False
)

data_visualizer = Agent(
    role="Data Visualizer",
    goal="Create insightful visualizations from data",
    backstory=("You are a data visualization specialist with a gift for representing "
              "complex data in clear, informative charts. You know exactly which type of "
              "visualization works best for different data types and questions."),
    tools=[generate_visualizations],
    llm_config={
        "config_list": [{"model": "gpt-4o-mini"}],
        "temperature": 0.1,
        "max_tokens": 4000
    },
    allow_delegation=False
)

data_analyst = Agent(
    role="Data Analyst",
    goal="Answer questions about data using natural language",
    backstory=("You are a skilled data analyst who excels at interpreting data and answering "
              "questions in plain language. You have a knack for understanding the intent behind "
              "questions and providing relevant insights from the data."),
    tools=[query_data],
    llm_config={
        "config_list": [{"model": "gpt-4o-mini"}],
        "temperature": 0.1,
        "max_tokens": 4000
    },
    allow_delegation=False,
    verbose=True
)

# Define tasks for each agent
def create_data_reader_task(file_path):
    return Task(
        description=f"Read and parse the uploaded file at {file_path}. Provide a comprehensive summary of the data.",
        expected_output=("A detailed summary of the data file including row count, column names, and basic statistics. "
                         "Do not use the same tool repeatedly. Once you have the data summary, return it as your final answer."),
        agent=data_reader
    )

def create_visualization_task(file_path):
    return Task(
        description=f"Generate a comprehensive set of visualizations for the data at {file_path}.",
        expected_output="A list of paths to generated visualization images that provide insights into the data.",
        agent=data_visualizer
    )

def create_query_task(file_path, query):
    return Task(
        description=f"Answer the following query about the data at {file_path}: '{query}'",
        expected_output="A detailed answer to the query, possibly with a relevant visualization.",
        agent=data_analyst
    )

# Function to create and run a data processing crew with telemetry disabled
def process_data_file(file_path: str) -> Union[str, Dict]:
    """
    Process a data file using CrewAI agents
    
    Args:
        file_path: Path to the data file
        
    Returns:
        Union[str, Dict]: Summary of the data and any generated visualizations
    """
    try:
        logger.info(f"Starting data processing for file: {file_path}")
        
        # Enhanced file path validation
        if not file_path or len(file_path) < 2:
            logger.error(f"Invalid file path: {file_path}")
            return f"Error: Invalid file path: {file_path}"
            
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return f"Error: File not found at {file_path}"
        
        # Create necessary directories
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        tasks = [
            create_data_reader_task(file_path),
            create_visualization_task(file_path)
        ]
        
        # Updated Crew initialization with telemetry disabled
        crew = Crew(
            agents=[data_reader, data_visualizer],
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
            telemetry=False,
            max_iterations=5
        )
        
        # Execute with timeout handling
        import threading
        import _thread
        
        result = None
        error = None
        
        def run_crew():
            nonlocal result, error
            try:
                result = crew.kickoff()
            except Exception as e:
                error = str(e)
                logger.error(f"Error in crew execution: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Create and start thread
        crew_thread = threading.Thread(target=run_crew)
        crew_thread.start()
        
        # Wait for completion with timeout
        crew_thread.join(timeout=300)  # 5 minute timeout
        
        # Check if thread is still running
        if crew_thread.is_alive():
            logger.error("Crew execution timed out after 5 minutes")
            # Force thread termination
            _thread._async_raise(crew_thread.ident, SystemExit)
            return "Error: Processing timed out after 5 minutes"
        
        if error:
            return f"Error during processing: {error}"
        
        # Structure the result for frontend consumption
        if isinstance(result, dict) and 'tasks_output' in result:
            for task in result['tasks_output']:
                if task.get('agent') == 'Data Reader' and task.get('raw'):
                    # Parse the raw output to extract structured data
                    summary = task.get('raw')
                    
                    # Return a structured result
                    return {
                        "status": "success",
                        "summary": summary,
                        "visualizations": result['tasks_output'][1]['raw'] if len(result['tasks_output']) > 1 else [],
                        "raw_result": result  # Keep the raw result for debugging
                    }
        
        logger.info("Data processing completed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Error in process_data_file: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error processing file: {str(e)}"

# Function to query data with updated message history
def query_data_file(file_path: str, query: str) -> Dict:
    """
    Query data using natural language
    
    Args:
        file_path: Path to the data file
        query: Natural language query
        
    Returns:
        Dict: Query result with answer and optional visualization
    """
    try:
        logger.info(f"Processing query: '{query}' for file: {file_path}")
        
        # Enhanced validation
        if not query.strip():
            return {"answer": "Error: Query cannot be empty"}
        
        # Enhanced file path validation
        if not file_path or len(file_path) < 2:
            logger.error(f"Invalid file path: {file_path}")
            return {"answer": f"Error: Invalid file path: {file_path}"}
            
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {"answer": f"Error: File not found at {file_path}"}
        
        task = create_query_task(file_path, query)
        
        # Crew with telemetry disabled
        crew = Crew(
            agents=[data_analyst],
            tasks=[task],
            verbose=True,
            telemetry=False,
            max_iterations=3
        )
        
        # Execute and handle result
        result = crew.kickoff()
            
        logger.info("Query processing completed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Error in query_data_file: {str(e)}")
        logger.error(traceback.format_exc())
        return {"answer": f"Error processing query: {str(e)}"}
