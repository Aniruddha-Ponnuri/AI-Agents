import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import List, Dict, Optional
from openai import OpenAI


# Load environment variables
load_dotenv()
openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])


@tool("Read and parse uploaded data file")
def read_data_file(file_path: str) -> str:
    """
    Reads and parses various file formats including CSV, Excel, and JSON.
    Returns a summary of the data and saves a serialized version for further processing.
    
    Args:
        file_path: Path to the uploaded file
        
    Returns:
        str: Summary of the parsed data
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return f"Error: File not found at path: {file_path}"
            
        # Determine file type by extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Read file based on extension
        if ext == '.csv':
            # Try to determine if there's a header row
            df = pd.read_csv(file_path, header=None)
            
            # Check if first row might be header
            first_row = df.iloc[0]
            if all(isinstance(x, str) for x in first_row):
                df = pd.read_csv(file_path)
            else:
                # If no header, provide column names
                column_names = [f'Column_{i+1}' for i in range(df.shape[1])]
                df = pd.read_csv(file_path, header=None, names=column_names)
                
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif ext == '.json':
            df = pd.read_json(file_path)
        else:
            return f"Unsupported file format: {ext}"
        
        # Save DataFrame as pickle for later use
        pickle_path = f"{file_path}.pkl"
        df.to_pickle(pickle_path)
        
        # Generate summary
        summary = f"File successfully parsed. Found {df.shape[0]} rows and {df.shape[1]} columns.\n\n"
        summary += "Column Names:\n" + "\n".join([f"- {col}" for col in df.columns]) + "\n\n"
        
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
        return f"Error parsing file: {str(e)}"

@tool("Generate exploratory visualizations")
def generate_visualizations(file_path: str) -> List[Dict[str, str]]:
    """
    Generates a comprehensive set of visualizations based on the data file.
    Returns metadata about each visualization.
    """
    try:
        # Load the data
        pickle_path = f"{file_path}.pkl"
        if not os.path.exists(pickle_path):
            return [{"error": "Processed data file not found. Please parse the file first."}]
            
        df = pd.read_pickle(pickle_path)
        
        # Extract dataset name from file path
        dataset_name = os.path.basename(file_path).split('.')[0]
        
        # Create directory for visualizations
        viz_dir = os.path.join('public', 'visualizations', dataset_name)
        os.makedirs(viz_dir, exist_ok=True)
        
        visualization_metadata = []
        
        # 1. Distribution of numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns[:5]  # Limit to first 5 for simplicity
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                plt.figure(figsize=(10, 6))
                sns.histplot(df[col].dropna(), kde=True)
                plt.title(f'Distribution of {col}')
                plt.tight_layout()
                
                filename = f'distribution_{col.replace(" ", "_").lower()}.png'
                chart_path = os.path.join(viz_dir, filename)
                plt.savefig(chart_path)
                plt.close()
                
                # Store metadata
                visualization_metadata.append({
                    "path": f'/visualizations/{dataset_name}/{filename}',
                    "title": f'Distribution of {col}',
                    "type": "distribution",
                    "column": col
                })
        
        # 2. Categorical data counts
        categorical_cols = df.select_dtypes(include=['object']).columns[:3]
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                plt.figure(figsize=(10, 6))
                value_counts = df[col].value_counts().head(10)  # Top 10 categories
                sns.barplot(x=value_counts.index, y=value_counts.values)
                plt.title(f'Count of Top 10 Values in {col}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                filename = f'count_{col.replace(" ", "_").lower()}.png'
                chart_path = os.path.join(viz_dir, filename)
                plt.savefig(chart_path)
                plt.close()
                
                # Store metadata
                visualization_metadata.append({
                    "path": f'/visualizations/{dataset_name}/{filename}',
                    "title": f'Count of Values in {col}',
                    "type": "categorical",
                    "column": col
                })
        
        # 3. If we have at least two numeric columns, create a correlation heatmap
        if len(numeric_cols) >= 2:
            plt.figure(figsize=(10, 8))
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Heatmap')
            plt.tight_layout()
            
            filename = 'correlation_heatmap.png'
            chart_path = os.path.join(viz_dir, filename)
            plt.savefig(chart_path)
            plt.close()
            
            # Store metadata
            visualization_metadata.append({
                "path": f'/visualizations/{dataset_name}/{filename}',
                "title": 'Correlation Heatmap',
                "type": "correlation",
                "columns": numeric_cols.tolist()
            })
            
        return visualization_metadata
        
    except Exception as e:
        return [{"error": f"Error generating visualizations: {str(e)}"}]


@tool("Query data using natural language")
def query_data(file_path: str, query: str) -> Dict:
    """
    Processes a natural language query about the data and returns relevant information.
    
    Args:
        file_path: Path to the data file
        query: Natural language query about the data
        
    Returns:
        Dict: Contains the answer and any relevant visualizations
    """
    try:
        # Load the data
        pickle_path = f"{file_path}.pkl"
        if not os.path.exists(pickle_path):
            return {"answer": "Error: Processed data file not found. Please parse the file first."}
            
        df = pd.read_pickle(pickle_path)
        
        # Process common query types
        query = query.lower()
        
        # Create viz directory if needed
        viz_dir = os.path.join('public', 'visualizations', os.path.basename(file_path).split('.')[0])
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate response based on query type
        response = {"answer": "", "visualization": None}
        
        # Basic data shape query
        if any(keyword in query for keyword in ["shape", "size", "rows", "columns"]):
            rows, cols = df.shape
            response["answer"] = f"The dataset has {rows} rows and {cols} columns."
            
        # Column list query
        elif any(keyword in query for keyword in ["columns", "fields", "variables"]):
            col_list = ", ".join(df.columns.tolist())
            response["answer"] = f"The dataset contains the following columns: {col_list}"
            
        # Summary statistics
        elif any(keyword in query for keyword in ["summary", "statistics", "describe"]):
            # Create a simplified description focusing on numerical columns
            num_description = df.describe().to_string()
            response["answer"] = f"Here's a statistical summary of the numerical columns:\n\n{num_description}"
            
        # Distribution query
        elif "distribution" in query:
            # Find which column to analyze
            for col in df.columns:
                if col.lower() in query:
                    if df[col].dtype in ['int64', 'float64']:
                        plt.figure(figsize=(10, 6))
                        sns.histplot(df[col].dropna(), kde=True)
                        plt.title(f'Distribution of {col}')
                        plt.tight_layout()
                        
                        chart_path = os.path.join(viz_dir, f'query_dist_{col}.png')
                        plt.savefig(chart_path)
                        plt.close()
                        
                        web_path = '/'.join(chart_path.split(os.sep)[1:])
                        response["answer"] = f"Here's the distribution of {col}."
                        response["visualization"] = web_path
                        break
                    else:
                        plt.figure(figsize=(10, 6))
                        value_counts = df[col].value_counts().head(10)
                        sns.barplot(x=value_counts.index, y=value_counts.values)
                        plt.title(f'Distribution of {col} (Top 10 Values)')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        
                        chart_path = os.path.join(viz_dir, f'query_dist_{col}.png')
                        plt.savefig(chart_path)
                        plt.close()
                        
                        web_path = '/'.join(chart_path.split(os.sep)[1:])
                        response["answer"] = f"Here's the distribution of values for {col} (showing top 10)."
                        response["visualization"] = web_path
                        break
                        
        # Correlation query
        elif "correlation" in query or "relationship" in query:
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            if len(numeric_cols) >= 2:
                # Try to find specific columns mentioned
                col1, col2 = None, None
                for col in numeric_cols:
                    if col.lower() in query:
                        if col1 is None:
                            col1 = col
                        elif col2 is None:
                            col2 = col
                            break
                
                # If no specific columns found or only one found, use the first two numeric columns
                if col1 is None or col2 is None:
                    if len(numeric_cols) >= 2:
                        col1, col2 = numeric_cols[0], numeric_cols[1]
                    
                if col1 and col2:
                    # Create scatterplot
                    plt.figure(figsize=(10, 6))
                    plt.scatter(df[col1], df[col2])
                    plt.title(f'Relationship between {col1} and {col2}')
                    plt.xlabel(col1)
                    plt.ylabel(col2)
                    plt.tight_layout()
                    
                    chart_path = os.path.join(viz_dir, f'query_corr_{col1}_{col2}.png')
                    plt.savefig(chart_path)
                    plt.close()
                    
                    # Calculate correlation
                    corr = df[col1].corr(df[col2])
                    corr_strength = "strong positive" if corr > 0.7 else "moderate positive" if corr > 0.3 else "weak positive" if corr > 0 else "strong negative" if corr < -0.7 else "moderate negative" if corr < -0.3 else "weak negative" if corr < 0 else "no"
                    
                    web_path = '/'.join(chart_path.split(os.sep)[1:])
                    response["answer"] = f"The correlation between {col1} and {col2} is {corr:.2f}, indicating a {corr_strength} correlation."
                    response["visualization"] = web_path
                    
        # Top/highest/lowest values
        elif any(keyword in query for keyword in ["top", "highest", "lowest", "maximum", "minimum"]):
            for col in df.columns:
                if col.lower() in query:
                    if "top" in query or "highest" in query or "maximum" in query:
                        top_values = df[col].nlargest(5)
                        response["answer"] = f"The top 5 values for {col} are:\n{top_values.to_string()}"
                    else:
                        bottom_values = df[col].nsmallest(5)
                        response["answer"] = f"The lowest 5 values for {col} are:\n{bottom_values.to_string()}"
                    break
                    
        # Average/mean query
        elif any(keyword in query for keyword in ["average", "mean"]):
            for col in df.columns:
                if col.lower() in query and df[col].dtype in ['int64', 'float64']:
                    avg_value = df[col].mean()
                    response["answer"] = f"The average {col} is {avg_value:.2f}"
                    break
        
        # Generic fallback
        if not response["answer"]:
            response["answer"] = ("I'm not sure how to answer this specific query. "
                                 "Try asking about data shape, columns, distributions, "
                                 "correlations, or statistical summaries.")
        
        return response
        
    except Exception as e:
        return {"answer": f"Error processing query: {str(e)}"}

# Define CrewAI agents with updated syntax
data_reader = Agent(
    role="Data Reader",
    goal="Accurately parse and summarize uploaded data files",
    backstory=("You are an expert data engineer who can parse any file format and "
              "extract meaningful information. Your specialty is providing clear summaries "
              "of datasets to help analysts understand what they're working with."),
    tools=[read_data_file],
    llm_config={"temperature": 0.2},  # Lower temperature for more consistent output
    allow_delegation=False  # Prevent excessive delegation
)

data_visualizer = Agent(
    role="Data Visualizer",
    goal="Create insightful visualizations from data",
    backstory=("You are a data visualization specialist with a gift for representing "
              "complex data in clear, informative charts. You know exactly which type of "
              "visualization works best for different data types and questions."),
    tools=[generate_visualizations],
    llm_config={"temperature": 0.3},
    allow_delegation=False
)



data_analyst = Agent(
    role="Data Analyst",
    goal="Answer questions about data using natural language",
    backstory=("You are a skilled data analyst who excels at interpreting data and answering "
              "questions in plain language. You have a knack for understanding the intent behind "
              "questions and providing relevant insights from the data."),
    tools=[query_data],
    verbose=True
)

# Define tasks for each agent
# Define tasks with clearer expected outputs
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

# Function to create and run a data processing crew
def process_data_file(file_path):
    tasks = [
        create_data_reader_task(file_path),
        create_visualization_task(file_path)
    ]
    
    crew = Crew(
        agents=[data_reader, data_visualizer],
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )
    
    result = crew.kickoff()
    return result

# Function to query data
def query_data_file(file_path, query):
    task = create_query_task(file_path, query)
    
    crew = Crew(
        agents=[data_analyst],
        tasks=[task],
        verbose=True
    )
    
    result = crew.kickoff()
    return result
