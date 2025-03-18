import formidable from 'formidable';
import fs from 'fs';
import axios from 'axios';

export const config = {
  api: {
    bodyParser: false,
  },
};

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  // Use formidable v3 syntax (modern approach)
  const form = formidable({});
  
  try {
    // Parse form using promises
    const [fields, files] = await new Promise((resolve, reject) => {
      form.parse(req, (err, fields, files) => {
        if (err) reject(err);
        resolve([fields, files]);
      });
    });
    
    // Get the file (object structure depends on formidable version)
    const file = files.file[0]; // For formidable v3
    // OR const file = files.file; // For formidable v2
    
    console.log('File details:', file); // Debug file object structure
    
    if (!file || !file.filepath) {
      return res.status(400).json({ error: 'No file or invalid file uploaded' });
    }
    
    // Read file and create form data for backend
    const fileData = fs.readFileSync(file.filepath);
    const formData = new FormData();
    formData.append('file', new Blob([fileData]), file.originalFilename || 'uploaded_file');
    
    // Forward to backend
    const response = await axios.post('http://localhost:8000/api/upload', formData);
    
    return res.status(200).json(response.data);
  } catch (error) {
    console.error('Error processing upload:', error);
    return res.status(500).json({ error: error.message || 'Error uploading file' });
  }
}
