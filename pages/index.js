import { useState } from "react";
import Head from "next/head";
import FileUploader from "../components/FileUploader";
import DataVisualizationGallery from '../components/DataVisualizationGallery';
import QueryInterface from "../components/QueryInterface";
import { FaFileCsv, FaFileAlt } from 'react-icons/fa';

export default function Home() {
  const [processedFileData, setProcessedFileData] = useState(null);

  const handleFileProcessed = (data) => {
    setProcessedFileData(data);
  };

  return (
    <div>
      <Head>
        <title>Data Analysis Platform</title>
        <meta name="description" content="Upload and analyze data files with AI" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="app-container">
        {/* Left Panel */}
        <div className="left-panel">
          <h1 className="text-2xl font-bold mb-6">Data Analysis Platform</h1>
          
          {/* File Upload Section */}
          <div className="upload-section">
            <h2 className="text-xl font-semibold mb-4">Upload File</h2>
            
            {!processedFileData ? (
              <FileUploader onFileProcessed={handleFileProcessed} />
            ) : (
              <div className="uploaded-file">
                <FaFileCsv className="file-icon" />
                <div>
                  <div className="font-medium">{processedFileData.filename}</div>
                  <div className="text-sm text-gray-500">File successfully uploaded and analyzed</div>
                </div>
              </div>
            )}
          </div>
          
          {/* Data Summary Section */}
          {processedFileData && processedFileData.summary && (
            <div className="data-summary">
              <h2 className="text-xl font-semibold mb-4">Data Summary</h2>
              <div className="bg-gray-50 rounded-lg p-4">
                {typeof processedFileData.summary === "object" ? (
                  <pre className="whitespace-pre-wrap overflow-auto text-sm">
                    {JSON.stringify(processedFileData.summary, null, 2)}
                  </pre>
                ) : (
                  <div className="whitespace-pre-line text-sm">
                    {processedFileData.summary}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
        
        {/* Right Panel */}
        <div className="right-panel">
          {/* Visualization Section */}
          <div className="visualizations">
            <h2 className="text-xl font-semibold mb-4">Data Visualizations</h2>
            {processedFileData ? (
              <DataVisualizationGallery fileName={processedFileData.filename} />
            ) : (
              <div className="text-center text-gray-500 py-16">
                Upload a file to view visualizations
              </div>
            )}
          </div>
          
          {/* Chat Section */}
          <div className="chat-section">
            <h2 className="text-xl font-semibold p-4 border-b">Ask Questions</h2>
            <div className="chat-messages p-4">
              {processedFileData ? (
                <QueryInterface fileData={processedFileData} />
              ) : (
                <div className="text-center text-gray-500 py-6">
                  Upload a file to ask questions about your data
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
