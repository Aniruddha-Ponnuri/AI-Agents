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
    """
    Reads and parses various file formats including CSV, Excel, and JSON.
    Returns a summary of the data and saves a serialized version for further processing.
    
    Args:
        file_path: Path to the uploaded file
        
    Returns:
        str: Summary of the parsed data
    """
    try:
        logger.info(f"Starting to parse file: {file_path}")
        rate_limit()  # Add rate limiting
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found at path: {file_path}")
            return f"Error: File not found at path: {file_path}"
            
        # Determine file type by extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        logger.info(f"Detected file format: {ext}")
        
        # Read file based on extension
        if ext == '.csv':
            # Try to determine if there's a header row
            try:
                df = pd.read_csv(file_path, header=None)
                
                # Check if first row might be header
                first_row = df.iloc[0]
                if all(isinstance(x, str) for x in first_row):
                    df = pd.read_csv(file_path)
                else:
                    # If no header, provide column names
                    column_names = [f'Column_{i+1}' for i in range(df.shape[1])]
                    df = pd.read_csv(file_path, header=None, names=column_names)
            except Exception as e:
                logger.error(f"Error parsing CSV file: {str(e)}")
                # Try with no header as fallback
                column_names = [f'Column_{i+1}' for i in range(5)]  # Assume at least 5 columns
                df = pd.read_csv(file_path, header=None, names=column_names)
                
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif ext == '.json':
            df = pd.read_json(file_path)
        else:
            logger.warning(f"Unsupported file format: {ext}")
            return f"Unsupported file format: {ext}"
        
        # Save DataFrame as pickle for later use
        pickle_path = f"{file_path}.pkl"
        df.to_pickle(pickle_path)
        logger.info(f"DataFrame saved to: {pickle_path}")
        
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
            
        logger.info(f"Successfully parsed file with {df.shape[0]} rows and {df.shape[1]} columns")
        return summary
        
    except Exception as e:
        logger.error(f"Error parsing file: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error parsing file: {str(e)}"

@tool("Generate exploratory visualizations")
def generate_visualizations(file_path: str) -> List[Dict[str, str]]:
    """
    Generates a comprehensive set of visualizations based on the data file.
    Returns metadata about each visualization.
    """
    try:
        logger.info(f"Starting visualization generation for: {file_path}")
        rate_limit()  # Add rate limiting
        
        # Load the data
        pickle_path = f"{file_path}.pkl"
        if not os.path.exists(pickle_path):
            logger.error(f"Pickle file not found: {pickle_path}")
            return [{"error": "Processed data file not found. Please parse the file first."}]
            
        df = pd.read_pickle(pickle_path)
        logger.info(f"Loaded data from pickle with shape: {df.shape}")
        
        # Extract dataset name from file path
        dataset_name = os.path.basename(file_path).split('.')[0]
        
        # Create directory for visualizations
        viz_dir = os.path.join('public', 'visualizations', dataset_name)
        os.makedirs(viz_dir, exist_ok=True)
        logger.info(f"Using visualization directory: {viz_dir}")
        
        visualization_metadata = []
        
        # 1. Distribution of numeric columns
        logger.info("Generating distribution visualizations for numeric columns")
        numeric_cols = df.select_dtypes(include=['number']).columns[:5]  # Limit to first 5 for simplicity
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                try:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(df[col].dropna(), kde=True)
                    plt.title(f'Distribution of {col}')
                    plt.tight_layout()
                    
                    filename = f'dist_{col}.png'
                    chart_path = os.path.join(viz_dir, filename)
                    plt.savefig(chart_path)
                    plt.close()
                    
                    # Store metadata
                    visualization_metadata.append({
                        "path": f'/static/visualizations/{dataset_name}/{filename}',
                        "title": f'Distribution of {col}',
                        "type": "distribution",
                        "column": str(col)
                    })
                    logger.info(f"Created distribution visualization for {col}")
                    rate_limit()  # Add rate limiting between visualizations
                except Exception as e:
                    logger.error(f"Error generating distribution for {col}: {str(e)}")
        
        # 2. Categorical data counts
        logger.info("Generating categorical visualizations")
        categorical_cols = df.select_dtypes(include=['object']).columns[:3]
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                try:
                    plt.figure(figsize=(10, 6))
                    value_counts = df[col].value_counts().head(10)  # Top 10 categories
                    sns.barplot(x=value_counts.index, y=value_counts.values)
                    plt.title(f'Count of Top 10 Values in {col}')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    filename = f'count_{col}.png'
                    chart_path = os.path.join(viz_dir, filename)
                    plt.savefig(chart_path)
                    plt.close()
                    
                    # Store metadata
                    visualization_metadata.append({
                        "path": f'/static/visualizations/{dataset_name}/{filename}',
                        "title": f'Count of Values in {col}',
                        "type": "categorical",
                        "column": str(col)
                    })
                    logger.info(f"Created categorical visualization for {col}")
                    rate_limit()  # Add rate limiting between visualizations
                except Exception as e:
                    logger.error(f"Error generating categorical chart for {col}: {str(e)}")
        
        # 3. If we have at least two numeric columns, create a correlation heatmap
        if len(numeric_cols) >= 2:
            try:
                logger.info("Generating correlation heatmap")
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
                    "path": f'/static/visualizations/{dataset_name}/{filename}',
                    "title": 'Correlation Heatmap',
                    "type": "correlation",
                    "columns": [str(col) for col in numeric_cols.tolist()]
                })
                logger.info("Created correlation heatmap")
            except Exception as e:
                logger.error(f"Error generating correlation heatmap: {str(e)}")
            
        logger.info(f"Generated {len(visualization_metadata)} visualizations")
        return visualization_metadata
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        logger.error(traceback.format_exc())
        return [{"error": f"Error generating visualizations: {str(e)}"}]

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
        rew = Crew(
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
                result = Crew.kickoff()
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
            telemetry=False,   # Configuration moved here
            max_iterations=3   # Configuration moved here
        )
        
        # No config parameter here
        result = Crew.kickoff()
        
        # Add the result to history if it's properly formatted
        if hasattr(result, 'answer'):
            history.add_ai_message(result.answer)
            
        logger.info("Query processing completed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Error in query_data_file: {str(e)}")
        logger.error(traceback.format_exc())
        return {"answer": f"Error processing query: {str(e)}"}
