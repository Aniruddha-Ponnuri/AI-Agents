import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';

export default function FileUploader({ onFileProcessed }) {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);

  const onDrop = useCallback(async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    setIsUploading(true);
    setUploadError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post('/api/upload', formData);
      
      if (onFileProcessed) {
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
    },
    multiple: false,
  });

  return (
    <div className="w-full max-w-2xl mx-auto mb-8">
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors
          ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-400'}`}
      >
        <input {...getInputProps()} />
        {isUploading ? (
          <div className="flex flex-col items-center">
            <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-500 mb-4"></div>
            <p className="text-gray-500">Uploading and processing your file...</p>
          </div>
        ) : (
          <div>
            <svg
              className="mx-auto h-12 w-12 text-gray-400"
              stroke="currentColor"
              fill="none"
              viewBox="0 0 48 48"
              aria-hidden="true"
            >
              <path
                d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                strokeWidth={2}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            <p className="mt-2 text-gray-600">
              Drag and drop a file here, or click to select a file
            </p>
            <p className="mt-1 text-sm text-gray-500">
              Supported formats: CSV, Excel, JSON
            </p>
          </div>
        )}
      </div>
      
      {uploadError && (
        <div className="mt-4 p-3 bg-red-100 border border-red-200 text-red-700 rounded">
          {uploadError}
        </div>
      )}
    </div>
  );
}
