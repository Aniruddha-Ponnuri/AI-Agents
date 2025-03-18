import { useState, useEffect } from 'react';
import axios from 'axios';

export default function DataVisualizationGallery({ fileName }) {
  const [visualizations, setVisualizations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    if (!fileName) return;
    
    const fetchVisualizations = async () => {
      try {
        setLoading(true);
        // Get the dataset name without extension
        const datasetName = fileName.split('.')[0];
        const response = await axios.get(`/api/visualizations/${datasetName}`);
        setVisualizations(response.data.visualizations || []);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching visualizations:', err);
        setError('Failed to load visualizations');
        setLoading(false);
      }
    };
    
    fetchVisualizations();
  }, [fileName]);
  
  if (loading) {
    return <div className="flex justify-center py-8">Loading visualizations...</div>;
  }
  
  if (error) {
    return <div className="text-red-500">{error}</div>;
  }
  
  if (visualizations.length === 0) {
    return <div>No visualizations available for this data.</div>;
  }
  
  return (
    <div className="mt-8">
      <h2 className="text-2xl font-bold mb-4">Data Visualizations</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {visualizations.map((viz, index) => (
          <div key={index} className="border rounded-lg overflow-hidden shadow-md">
            <div className="p-3 bg-gray-50 border-b">
              <h3 className="font-medium">{viz.title}</h3>
            </div>
            <img 
              src={`http://localhost:8000${viz.path}`} 
              alt={viz.title || `Visualization ${index + 1}`}
              className="w-full h-auto" 
            />
          </div>
        ))}
      </div>
    </div>
  );
}
