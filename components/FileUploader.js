import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';

export default function FileUploader({ onFileProcessed }) {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setIsUploading(true);
    setUploadError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await axios.post('/api/upload', formData);
      
      if (response.data && response.data.status === 'success') {
        onFileProcessed(response.data);
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      setUploadError('Error uploading file. Please try again.');
    } finally {
      setIsUploading(false);
    }
  }, [onFileProcessed]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/json': ['.json'],
      'application/pdf': ['.pdf'],
    },
    multiple: false
  });

  return (
    <div>
      <div
        {...getRootProps()}
        className={`file-dropzone ${isDragActive ? 'border-blue-500 bg-blue-50' : ''}`}
      >
        <input {...getInputProps()} />
        {isDragActive ? (
          <p>Drop your file here...</p>
        ) : (
          <div>
            <p className="mb-2">Drag and drop a file here, or click to select a file</p>
            <p className="text-sm text-gray-500">Supported formats: CSV, Excel, JSON, PDF</p>
          </div>
        )}
      </div>
      
      {isUploading && (
        <div className="mt-4 text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-t-blue-500 border-gray-200"></div>
          <p className="mt-2 text-gray-600">Uploading and processing your file...</p>
        </div>
      )}
      
      {uploadError && (
        <div className="mt-4 p-3 bg-red-100 text-red-700 rounded-lg">
          {uploadError}
        </div>
      )}
    </div>
  );
}
