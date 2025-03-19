from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import shutil
from typing import Optional
import logging
import traceback
from agents import process_data_file, query_data_file

app = FastAPI()

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories first - streamlined approach
for dir_path in ["public", "public/uploads", "public/visualizations"]:
    if not os.path.exists(dir_path):
        print(f"Creating directory: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="public"), name="static")

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("data_agents")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint to upload a data file
    """
    try:
        # Create file path
        file_path = os.path.join("public/uploads", file.filename)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the file with CrewAI
        result = process_data_file(file_path)
        
        summary = ""
        if isinstance(result, dict):
            if 'summary' in result:
                summary = result['summary']
            elif 'tasks_output' in result:
                for task in result['tasks_output']:
                    if task.get('agent') == 'Data Reader':
                        summary = task.get('raw', '')
                        break
        
        return {
            "status": "success", 
            "filename": file.filename,
            "file_path": file_path,
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/query")
async def query_endpoint(file_path: str = Form(...), query: str = Form(...)):
    print(f"Received file_path: {file_path}")
    """
    Endpoint to query data using natural language
    """
    try:
        # Use your existing query_data_file function
        result = query_data_file(file_path, query)
        
        # Handle the result based on its structure
        response = {
            "status": "success",
            "query": query,
            "sender_type": "bot"  # Add explicit sender type
        }
        
        # Process the result based on its structure
        if isinstance(result, dict):
            # Check if it has direct answer property
            if "answer" in result:
                response["result"] = result
            # Check if it's a CrewAI tasks output format
            elif "tasks_output" in result:
                data_analyst_output = None
                for task in result["tasks_output"]:
                    if task.get("agent") == "Data Analyst":
                        data_analyst_output = task
                        break
                
                if data_analyst_output:
                    # Extract the answer from Data Analyst output
                    response["result"] = {
                        "answer": data_analyst_output.get("raw", ""),
                        "tasks_output": result["tasks_output"],
                        "raw": result
                    }
                else:
                    response["result"] = result
            else:
                response["result"] = result
        else:
            # For string or other primitive responses
            response["result"] = {"answer": str(result)}
        
        return response
    except Exception as e:
        logger.error(f"Error in query_endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))




@app.get("/api/visualizations/{dataset_name}")
async def list_visualizations(dataset_name: str):
    """
    Get a list of all visualization files for a specific dataset
    """
    try:
        viz_dir = os.path.join("public/visualizations", dataset_name)
        
        if not os.path.exists(viz_dir):
            return {"visualizations": []}
        
        # Get all visualization files in the directory
        viz_files = []
        for file in os.listdir(viz_dir):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                # Format the path for frontend use
                file_path = f"/static/visualizations/{dataset_name}/{file}"
                
                # Extract visualization info from filename
                if file.startswith('dist_'):
                    col_name = file.replace('dist_', '').replace('.png', '')
                    title = f"Distribution of {format_column_name(col_name)}"
                    viz_type = "distribution"
                    column = col_name
                elif file.startswith('count_'):
                    col_name = file.replace('count_', '').replace('.png', '')
                    title = f"Count of Values in {format_column_name(col_name)}"
                    viz_type = "categorical"
                    column = col_name
                elif 'correlation_heatmap' in file:
                    title = "Correlation Heatmap"
                    viz_type = "correlation"
                    column = None
                else:
                    title = file.replace('.png', '').replace('_', ' ').title()
                    viz_type = "other"
                    column = None
                
                viz_files.append({
                    "path": file_path,
                    "title": title,
                    "type": viz_type,
                    "column": column,
                    "filename": file
                })
        
        return {"visualizations": viz_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def format_column_name(column_name: str) -> str:
    """Format a column name for display"""
    words = column_name.replace('_', ' ').split()
    return ' '.join(word.capitalize() for word in words)


def get_visualization_title(filename: str) -> str:
    """Generate a readable title from a visualization filename"""
    if filename.startswith('dist_'):
        column_name = filename.replace('dist_', '').replace('.png', '')
        return f"Distribution of {column_name}"
    elif filename.startswith('count_'):
        column_name = filename.replace('count_', '').replace('.png', '')
        return f"Count of Values in {column_name}"
    elif 'correlation_heatmap' in filename:
        return "Correlation Heatmap"
    else:
        # Default: Convert underscores to spaces and capitalize
        return filename.replace('.png', '').replace('_', ' ').title()
