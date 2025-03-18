import { useState } from "react";
import Head from "next/head";
import FileUploader from "../components/FileUploader";
import QueryInterface from "../components/QueryInterface";
import DataVisualizationGallery from '../components/DataVisualizationGallery';

export default function Home() {
  const [processedFileData, setProcessedFileData] = useState(null);

  const handleFileProcessed = (data) => {
    setProcessedFileData(data);
  };

  return (
    <div>
      <Head>
        <title>Data Analysis with CrewAI</title>
        <meta
          name="description"
          content="Upload data files and analyze them with AI"
        />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="container mx-auto px-4 py-8">
        <h1 className="text-3xl md:text-4xl font-bold text-center mb-8">
          CrewAI Data Analysis Platform
        </h1>

        <div className="max-w-4xl mx-auto">
          <h2 className="text-2xl font-bold mb-4">Upload Your Data</h2>
          <FileUploader onFileProcessed={handleFileProcessed} />

          {processedFileData && (
            <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
              <h3 className="text-lg font-medium text-green-800">
                File Successfully Processed!
              </h3>
              <p className="mt-1 text-green-700">
                Your file{" "}
                <span className="font-semibold">
                  {processedFileData.filename}
                </span>{" "}
                has been uploaded and analyzed.
              </p>
            </div>
          )}

          {processedFileData && processedFileData.summary && (
            <div className="mt-6">
              <h3 className="text-xl font-bold mb-2">Data Summary</h3>
              <div className="bg-gray-50 border rounded-lg p-4">
                {typeof processedFileData.summary === "object" ? (
                  <pre className="whitespace-pre-wrap overflow-auto">
                    {JSON.stringify(processedFileData.summary, null, 2)}
                  </pre>
                ) : (
                  <div className="whitespace-pre-line">
                    {processedFileData.summary}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Replace DataVisualizer with DataVisualizationGallery */}
          {processedFileData && (
            <DataVisualizationGallery fileName={processedFileData.filename} />
          )}

          <QueryInterface fileData={processedFileData} />
        </div>
      </main>

      <footer className="mt-12 py-6 border-t">
        <div className="container mx-auto px-4 text-center text-gray-500">
          <p>Powered by CrewAI and Next.js</p>
        </div>
      </footer>
    </div>
  );
}
