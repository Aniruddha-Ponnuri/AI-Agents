from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import shutil
from typing import Optional

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
        
        # Handle result properly for frontend display
        if isinstance(result, dict):
            # Format result for frontend
            formatted_result = {
                "status": "success", 
                "filename": file.filename,
                "file_path": file_path,
                "summary": str(result)  # Convert complex objects to string
            }
        else:
            formatted_result = {
                "status": "success", 
                "filename": file.filename,
                "file_path": file_path,
                "summary": result
            }
        
        return formatted_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
async def query_endpoint(file_path: str = Form(...), query: str = Form(...)):
    """
    Endpoint to query data using natural language
    """
    try:
        # Validate inputs
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file path or file does not exist"
            )
        
        if not query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        # Use your existing query_data_file function
        result = query_data_file(file_path, query)
        
        return {
            "status": "success",
            "query": query,
            "result": result
        }
    except Exception as e:
        # Return proper error responses
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/visualizations/{dataset_name}")
async def list_visualizations(dataset_name: str):
    """
    Get a list of all visualization files for a specific dataset
    """
    try:
        # Extract base name without extension if the dataset_name includes an extension
        base_name = dataset_name.split('.')[0]
        viz_dir = os.path.join("public/visualizations", base_name)
        
        if not os.path.exists(viz_dir):
            return {"visualizations": []}
        
        # Get all visualization files in the directory
        viz_files = []
        for file in os.listdir(viz_dir):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                # Format the path for frontend use
                file_path = f"/static/visualizations/{base_name}/{file}"
                title = get_visualization_title(file)
                viz_files.append({
                    "path": file_path,
                    "title": title,
                    "filename": file
                })
        
        return {"visualizations": viz_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
