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
    allow_origins=["*"],  # Set to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories BEFORE mounting static files
os.makedirs("public", exist_ok=True)
os.makedirs("public/uploads", exist_ok=True)
os.makedirs("public/visualizations", exist_ok=True)

if not os.path.exists("public"):
    print("Warning: 'public' directory doesn't exist, creating it now...")
    os.makedirs("public", exist_ok=True)


# Now mount static files directory
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
async def query_data(file_path: str = Form(...), query: str = Form(...)):
    """
    Endpoint to query data using natural language
    """
    try:
        # Validate file path
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Query the data with CrewAI
        result = query_data_file(file_path, query)
        
        return {
            "status": "success",
            "query": query,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/visualizations/{file_name}")
async def get_visualizations(file_name: str):
    """
    Get list of visualizations for a specific file
    """
    try:
        # Extract base name without extension
        base_name = file_name.split('.')[0]
        viz_dir = os.path.join("public/visualizations", base_name)
        
        if not os.path.exists(viz_dir):
            return {"visualizations": []}
        
        viz_files = [f"/static/visualizations/{base_name}/{f}" for f in os.listdir(viz_dir) 
                    if f.endswith(('.png', '.jpg'))]
        
        return {"visualizations": viz_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

