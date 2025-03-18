import { useState, useEffect } from 'react';
import axios from 'axios';
import Image from 'next/image';

export default function DataVisualizationGallery({ fileName }) {
  const [visualizations, setVisualizations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    if (!fileName) return;
    
    const fetchVisualizations = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`/api/visualizations/${fileName}`);
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
        {visualizations.map((vizUrl, index) => (
          <div key={index} className="border rounded-lg overflow-hidden shadow-md">
            <div className="p-3 bg-gray-50 border-b">
              <h3 className="font-medium">
                {getVisualizationTitle(vizUrl)}
              </h3>
            </div>
            <img 
              src={`http://localhost:8000${vizUrl}`} 
              alt={`Visualization ${index + 1}`}
              className="w-full h-auto" 
            />
          </div>
        ))}
      </div>
    </div>
  );
}

// Helper function to generate readable titles from file paths
function getVisualizationTitle(path) {
  const filename = path.split('/').pop();
  
  if (filename.includes('dist_Column_1')) {
    return 'Distribution of Sepal Length';
  } else if (filename.includes('dist_Column_2')) {
    return 'Distribution of Sepal Width';
  } else if (filename.includes('dist_Column_3')) {
    return 'Distribution of Petal Length';
  } else if (filename.includes('dist_Column_4')) {
    return 'Distribution of Petal Width';
  } else if (filename.includes('count_Column_5')) {
    return 'Count of Iris Species';
  } else if (filename.includes('correlation_heatmap')) {
    return 'Correlation Heatmap';
  }
  
  return filename;
}
