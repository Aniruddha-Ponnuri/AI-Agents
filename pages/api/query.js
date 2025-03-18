import axios from 'axios';
import formidable from 'formidable';

// Disable the default bodyParser to handle form data
export const config = {
  api: {
    bodyParser: false,
  },
};

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  try {
    // Use formidable to parse the form data
    const form = formidable({});
    
    const [fields] = await new Promise((resolve, reject) => {
      form.parse(req, (err, fields, files) => {
        if (err) reject(err);
        resolve([fields, files]);
      });
    });
    
    // Extract the file_path and query from fields
    const file_path = fields.file_path?.[0] || fields.file_path;
    const query = fields.query?.[0] || fields.query;
    
    if (!file_path || !query) {
      return res.status(400).json({ 
        error: 'Missing required parameters: file_path and query are required' 
      });
    }
    
    // Create a new FormData for the backend request
    const formData = new FormData();
    formData.append('file_path', file_path);
    formData.append('query', query);
    
    // Forward the request to the FastAPI backend
    const response = await axios.post('http://localhost:8000/api/query', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    
    return res.status(200).json(response.data);
  } catch (error) {
    console.error('Error querying data:', error);
    
    if (error.response) {
      return res.status(error.response.status).json({
        error: error.response.data.detail || 'Query processing failed'
      });
    }
    
    return res.status(500).json({ error: 'Failed to query data' });
  }
}
