import axios from 'axios';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  try {
    const { file_path, query } = req.body;
    
    if (!file_path || !query) {
      return res.status(400).json({ error: 'File path and query are required' });
    }
    
    // Forward to FastAPI backend
    const formData = new FormData();
    formData.append('file_path', file_path);
    formData.append('query', query);
    
    const response = await axios.post('http://localhost:8000/api/query', formData);
    
    return res.status(200).json(response.data);
  } catch (error) {
    console.error('Error querying data:', error);
    return res.status(500).json({ error: 'Error querying data' });
  }
}
