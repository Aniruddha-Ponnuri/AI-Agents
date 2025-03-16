import multer from 'multer';
import axios from 'axios';
import { createRouter } from 'next-connect';

const upload = multer({ 
  storage: multer.memoryStorage() 
});

export const config = {
  api: {
    bodyParser: false,
  },
};

const apiRoute = createRouter();

apiRoute.use(upload.single('file'));

apiRoute.post(async (req, res) => {
  try {
    const file = req.file;
    const formData = new FormData();
    formData.append('file', new Blob([file.buffer]), file.originalname);
    
    const response = await axios.post('http://localhost:8000/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    res.status(200).json(response.data);
  } catch (error) {
    console.error('Error uploading file:', error);
    res.status(500).json({ error: 'Error uploading file' });
  }
});

export default apiRoute.handler();
