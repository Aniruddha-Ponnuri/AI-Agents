import axios from 'axios';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  try {
    // Create FormData to forward to the backend
    const formData = new FormData();
    
    // Extract data from request body
    const { file_path, query } = req.body;
    
    if (!file_path || !query) {
      return res.status(400).json({ 
        error: 'Missing required parameters: file_path and query are required' 
      });
    }
    
    formData.append('file_path', file_path);
    formData.append('query', query);
    
    // Make request to FastAPI backend
    const response = await axios.post('http://localhost:8000/api/query', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    
    return res.status(200).json(response.data);
  } catch (error) {
    console.error('Error querying data:', error);
    
    // Forward error details from the backend
    if (error.response) {
      return res.status(error.response.status).json({
        error: error.response.data.detail || 'Query processing failed'
      });
    }
    
    return res.status(500).json({ error: 'Failed to query data' });
  }
}
