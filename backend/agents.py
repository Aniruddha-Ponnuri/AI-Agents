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

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("data_agents")

# Load environment variables
load_dotenv()

# Set up Groq
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
from groq import Groq

# Set up Langchain for conversation memory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Initialize Groq client
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Import LangChain Groq components
from langchain_groq import ChatGroq

# Initialize LangChain's Groq model
llm = ChatGroq(
    api_key=os.environ["GROQ_API_KEY"],
    model_name="llama-3.1-8b-instant"
)

# Create a dictionary to store chat histories for different sessions
message_histories = {}

# Create a function to get or create message history for a session
def get_message_history(session_id: str):
    if session_id not in message_histories:
        message_histories[session_id] = ChatMessageHistory()
    return message_histories[session_id]

# Create a runnable chain with message history
def create_chain_with_history():
    chain = (
        RunnablePassthrough.assign(
            response=lambda x: llm.invoke(x["messages"])
        )
        | (lambda x: x["response"].content)
    )
    
    return RunnableWithMessageHistory(
        chain,
        get_message_history,
        input_messages_key="messages",
        history_messages_key="history"
    )

# Initialize the chain
conversation_chain = create_chain_with_history()

# Add rate limiting to prevent overwhelming the server
def rate_limit():
    """Simple rate limiting function that pauses execution briefly"""
    time.sleep(0.5)  # 500ms pause between operations

@tool("Read and parse uploaded data file")
def read_data_file(file_path: str) -> str:
    """Reads and parses data files, preserving original column names."""
    try:
        logger.info(f"Starting to parse file: {file_path}")
        rate_limit()
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return f"Error: File not found at path: {file_path}"
            
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Store column mapping for visualization 
        column_metadata = {}
        
        if ext == '.csv':
            # Try to intelligently detect headers
            try:
                # First try with pandas auto-detection
                df = pd.read_csv(file_path)
                logger.info(f"CSV file parsed with automatic header detection")
                
                # Store original column names
                for i, col_name in enumerate(df.columns):
                    column_metadata[f"col_{i}"] = str(col_name)
                
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
                
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            # Store original column names
            for i, col_name in enumerate(df.columns):
                column_metadata[f"col_{i}"] = str(col_name)
                
        elif ext == '.json':
            df = pd.read_json(file_path)
            # Store original column names
            for i, col_name in enumerate(df.columns):
                column_metadata[f"col_{i}"] = str(col_name)
        else:
            return f"Unsupported file format: {ext}"
        
        # Save DataFrame as pickle for later use
        pickle_path = f"{file_path}.pkl"
        df.to_pickle(pickle_path)
        
        # Save column metadata
        metadata_path = f"{file_path}.meta.json"
        with open(metadata_path, 'w') as f:
            json.dump(column_metadata, f)
        
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
        logger.error(f"Error parsing file: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error parsing file: {str(e)}"


@tool("Generate exploratory visualizations")
def generate_visualizations(file_path: str) -> List[Dict[str, str]]:
    """Generates visualizations using actual column names from the dataset."""
    try:
        logger.info(f"Starting visualization generation for: {file_path}")
        rate_limit()
        
        # Load the data
        pickle_path = f"{file_path}.pkl"
        if not os.path.exists(pickle_path):
            logger.error(f"Pickle file not found: {pickle_path}")
            return [{"error": "Processed data file not found. Please parse the file first."}]
            
        df = pd.read_pickle(pickle_path)
        
        # Load column metadata if available
        metadata_path = f"{file_path}.meta.json"
        column_readable_names = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    column_readable_names = json.load(f)
            except:
                logger.warning("Could not load column metadata, using raw column names")
        
        # Extract dataset name from file path
        dataset_name = os.path.basename(file_path).split('.')[0]
        
        # Create directory for visualizations
        viz_dir = os.path.join('public', 'visualizations', dataset_name)
        os.makedirs(viz_dir, exist_ok=True)
        
        visualization_metadata = []
        
        # 1. Distribution of numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns[:5]
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                try:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(df[col].dropna(), kde=True)
                    
                    # Use a clean display name for the column in the title
                    display_name = get_display_name(col, column_readable_names)
                    plt.title(f'Distribution of {display_name}')
                    plt.tight_layout()
                    
                    # Use a URL-safe filename
                    safe_col_name = str(col).replace(' ', '_').replace('/', '_').lower()
                    filename = f'dist_{safe_col_name}.png'
                    chart_path = os.path.join(viz_dir, filename)
                    plt.savefig(chart_path)
                    plt.close()
                    
                    visualization_metadata.append({
                        "path": f'/static/visualizations/{dataset_name}/{filename}',
                        "title": f'Distribution of {display_name}',
                        "type": "distribution",
                        "column": str(col)
                    })
                except Exception as e:
                    logger.error(f"Error generating distribution for {col}: {str(e)}")
        
        # 2. Categorical data counts (similar modifications for categorical columns)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns[:3]
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                try:
                    plt.figure(figsize=(10, 6))
                    value_counts = df[col].value_counts().head(10)
                    sns.barplot(x=value_counts.index, y=value_counts.values)
                    
                    display_name = get_display_name(col, column_readable_names)
                    plt.title(f'Count of Values in {display_name}')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    safe_col_name = str(col).replace(' ', '_').replace('/', '_').lower()
                    filename = f'count_{safe_col_name}.png'
                    chart_path = os.path.join(viz_dir, filename)
                    plt.savefig(chart_path)
                    plt.close()
                    
                    visualization_metadata.append({
                        "path": f'/static/visualizations/{dataset_name}/{filename}',
                        "title": f'Count of Values in {display_name}',
                        "type": "categorical",
                        "column": str(col)
                    })
                except Exception as e:
                    logger.error(f"Error generating categorical chart for {col}: {str(e)}")
        
        # 3. Correlation heatmap (similar modifications)
        if len(numeric_cols) >= 2:
            try:
                plt.figure(figsize=(10, 8))
                corr_matrix = df[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
                
                # Use human-readable column labels for better understanding
                display_labels = [get_display_name(col, column_readable_names) for col in numeric_cols]
                plt.xticks(ticks=range(len(numeric_cols)), labels=display_labels, rotation=45)
                plt.yticks(ticks=range(len(numeric_cols)), labels=display_labels, rotation=0)
                
                plt.title('Correlation Heatmap')
                plt.tight_layout()
                
                filename = 'correlation_heatmap.png'
                chart_path = os.path.join(viz_dir, filename)
                plt.savefig(chart_path)
                plt.close()
                
                visualization_metadata.append({
                    "path": f'/static/visualizations/{dataset_name}/{filename}',
                    "title": 'Correlation Heatmap',
                    "type": "correlation",
                    "columns": [str(col) for col in numeric_cols.tolist()]
                })
            except Exception as e:
                logger.error(f"Error generating correlation heatmap: {str(e)}")
            
        return visualization_metadata
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        logger.error(traceback.format_exc())
        return [{"error": f"Error generating visualizations: {str(e)}"}]

# Helper function to get a clean display name for a column
def get_display_name(column, metadata_dict=None):
    """Get a human-readable display name for a column"""
    if metadata_dict:
        # Look for the column in metadata
        for key, value in metadata_dict.items():
            if str(column) == str(value):
                return value
    
    # If not found in metadata or no metadata, clean up the column name
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
    
    # Default: just return the original
    return col_str


@tool("Query data using natural language")
def query_data(file_path: str, query: str) -> Dict:
    """
    Processes a natural language query about the data and returns relevant information.
    Uses conversation history for context.
    
    Args:
        file_path: Path to the data file
        query: Natural language query about the data
        
    Returns:
        Dict: Contains the answer and any relevant visualizations
    """
    try:
        logger.info(f"Processing query: '{query}' for file: {file_path}")
        rate_limit()  # Add rate limiting
        
        # Load the data
        pickle_path = f"{file_path}.pkl"
        if not os.path.exists(pickle_path):
            logger.error(f"Pickle file not found: {pickle_path}")
            return {"answer": "Error: Processed data file not found. Please parse the file first."}
            
        df = pd.read_pickle(pickle_path)
        logger.info(f"Loaded data from pickle with shape: {df.shape}")
        
        # Create a unique session ID based on the file path
        session_id = f"query_{os.path.basename(file_path)}"
        
        # Get context from previous interactions
        history = get_message_history(session_id)
        context = ""
        if len(history.messages) > 0:
            context = "Previous queries and answers:\n"
            for msg in history.messages[-4:]:  # Get last 2 exchanges (4 messages)
                if isinstance(msg, HumanMessage):
                    context += f"Q: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    context += f"A: {msg.content}\n"
        
        # Process query with conversation chain
        messages = [
            HumanMessage(content=f"{context}\nNew query about data in {os.path.basename(file_path)}: {query}")
        ]
        
        # Process common query types
        query = query.lower()
        
        # Create viz directory if needed
        dataset_name = os.path.basename(file_path).split('.')[0]
        viz_dir = os.path.join('public', 'visualizations', dataset_name)
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate response based on query type
        response = {"answer": "", "visualization": None}
        
        # Handle different query types
        try:
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
                    if col.lower() in query or str(col).lower() in query:
                        if df[col].dtype in ['int64', 'float64']:
                            try:
                                plt.figure(figsize=(10, 6))
                                sns.histplot(df[col].dropna(), kde=True)
                                plt.title(f'Distribution of {col}')
                                plt.tight_layout()
                                
                                filename = f'query_dist_{col}.png'
                                chart_path = os.path.join(viz_dir, filename)
                                plt.savefig(chart_path)
                                plt.close()
                                
                                visualization_path = f'/static/visualizations/{dataset_name}/{filename}'
                                response["answer"] = f"Here's the distribution of {col}."
                                response["visualization"] = visualization_path
                                logger.info(f"Created distribution visualization for query: {filename}")
                                break
                            except Exception as viz_error:
                                logger.error(f"Error creating visualization: {str(viz_error)}")
                        else:
                            try:
                                plt.figure(figsize=(10, 6))
                                value_counts = df[col].value_counts().head(10)
                                sns.barplot(x=value_counts.index, y=value_counts.values)
                                plt.title(f'Distribution of {col} (Top 10 Values)')
                                plt.xticks(rotation=45)
                                plt.tight_layout()
                                
                                filename = f'query_dist_{col}.png'
                                chart_path = os.path.join(viz_dir, filename)
                                plt.savefig(chart_path)
                                plt.close()
                                
                                visualization_path = f'/static/visualizations/{dataset_name}/{filename}'
                                response["answer"] = f"Here's the distribution of values for {col} (showing top 10)."
                                response["visualization"] = visualization_path
                                logger.info(f"Created categorical distribution for query: {filename}")
                                break
                            except Exception as viz_error:
                                logger.error(f"Error creating visualization: {str(viz_error)}")
        except Exception as query_error:
            logger.error(f"Error processing specific query type: {str(query_error)}")
        
        # If no specific handler was triggered, use the LLM
        if not response["answer"]:
            try:
                # Use the conversation chain to get a response
                llm_response = conversation_chain.invoke(
                    {
                        "messages": messages,
                        "config": {"configurable": {"session_id": session_id}}
                    }
                )
                response["answer"] = llm_response
                logger.info("Used LLM for response generation")
            except Exception as llm_error:
                logger.error(f"Error using LLM for response: {str(llm_error)}")
                response["answer"] = ("I'm not sure how to answer this specific query. "
                                    "Try asking about data shape, columns, distributions, "
                                    "correlations, or statistical summaries.")
        
        # Add the query and response to message history
        history.add_user_message(query)
        history.add_ai_message(response["answer"])
        
        logger.info(f"Query processing complete, response size: {len(response['answer'])}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        return {"answer": f"Error processing query: {str(e)}"}

# Define CrewAI agents with Groq as the LLM
data_reader = Agent(
    role="Data Reader",
    goal="Accurately parse and summarize uploaded data files",
    backstory=("You are an expert data engineer who can parse any file format and "
              "extract meaningful information. Your specialty is providing clear summaries "
              "of datasets to help analysts understand what they're working with."),
    tools=[read_data_file],
    llm_config={
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
        "temperature": 0.1,
        "max_tokens": 4000  # Limit token usage
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
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
        "temperature": 0.1,
        "max_tokens": 4000  # Limit token usage
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
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
        "temperature": 0.01,
        "max_tokens": 4000  # Limit token usage
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
        
        # Basic validation
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return f"Error: File not found at {file_path}"
        
        # Create necessary directories
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        tasks = [
            create_data_reader_task(file_path),
            create_visualization_task(file_path)
        ]
        
        # Updated Crew initialization with correct configuration structure
        crew = Crew(
            agents=[data_reader, data_visualizer],
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
            telemetry=False,  # Configuration moved here
            max_iterations=5  # Configuration moved here
        )
        
        # Execute with timeout handling
        import threading
        import _thread
        
        result = None
        error = None
        
        def run_crew():
            nonlocal result, error
            try:
                # No config parameter here
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
        
        logger.info("Data processing completed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Error in process_data_file: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error processing file: {str(e)}"

# Function to query data with updated message history
def query_data_file(file_path: str, query: str) -> Dict:
    try:
        logger.info(f"Processing query: '{query}' for file: {file_path}")
        
        if not query.strip():
            return {"answer": "Error: Query cannot be empty"}
        
        if not os.path.exists(file_path):
            return {"answer": f"Error: File not found at {file_path}"}
        
        # Create a unique session ID for this file
        session_id = f"query_{os.path.basename(file_path)}"
        
        # Add the query to the history
        history = get_message_history(session_id)
        history.add_user_message(f"Query about {os.path.basename(file_path)}: {query}")
        
        task = create_query_task(file_path, query)
        
        # Move configuration parameters to the Crew initialization
        crew = Crew(
            agents=[data_analyst],
            tasks=[task],
            verbose=True,
            telemetry=False,   
            max_iterations=3   
        )
        
        # No config parameter here
        result = crew.kickoff()
        
        # Add the result to history if it's properly formatted
        if hasattr(result, 'answer'):
            history.add_ai_message(result.answer)
            
        logger.info("Query processing completed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Error in query_data_file: {str(e)}")
        logger.error(traceback.format_exc())
        return {"answer": f"Error processing query: {str(e)}"}
