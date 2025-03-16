import { useState, useEffect } from 'react';
import axios from 'axios';

export default function DataVisualizer({ fileData }) {
  const [visualizations, setVisualizations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchVisualizations = async () => {
      if (!fileData || !fileData.filename) return;
      
      setLoading(true);
      setError(null);
      
      try {
        const response = await axios.get(`/api/visualizations/${fileData.filename}`);
        setVisualizations(response.data.visualizations || []);
      } catch (err) {
        console.error('Error fetching visualizations:', err);
        setError('Error loading visualizations. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchVisualizations();
  }, [fileData]);

  if (!fileData) {
    return null;
  }

  return (
    <div className="mt-8">
      <h2 className="text-2xl font-bold mb-4">Data Visualizations</h2>
      
      {loading && (
        <div className="flex justify-center py-8">
          <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-500"></div>
        </div>
      )}
      
      {error && (
        <div className="p-3 bg-red-100 border border-red-200 text-red-700 rounded">
          {error}
        </div>
      )}
      
      {!loading && visualizations.length === 0 && (
        <p className="text-gray-500 italic">No visualizations available yet. They will appear here once generated.</p>
      )}
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {visualizations.map((vizPath, index) => (
          <div key={index} className="border rounded-lg overflow-hidden shadow-sm">
            <img 
              src={vizPath} 
              alt={`Visualization ${index + 1}`} 
              className="w-full h-auto" 
            />
          </div>
        ))}
      </div>
    </div>
  );
}
