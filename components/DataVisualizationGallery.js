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
    <div className="visualizations-container">
      <h2 className="text-xl font-semibold mb-4">Data Visualizations</h2>
      <div className="visualization-grid">
        {visualizations.map((viz, index) => (
          <div key={index} className="visualization-card">
            <div className="visualization-header">
              {viz.title || getVisualizationTitle(viz.path)}
            </div>
            <div className="visualization-body">
              <img 
                src={`http://localhost:8000${viz.path}`} 
                alt={viz.title || `Visualization ${index + 1}`}
                className="w-full h-auto" 
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Fallback function for visualization title generation if metadata is missing
function getVisualizationTitle(path) {
  const filename = path.split('/').pop();
  
  // Extract visualization type and column name from filename
  if (filename.startsWith('dist_')) {
    const columnName = filename.replace('dist_', '').replace('.png', '');
    return `Distribution of ${formatColumnName(columnName)}`;
  } else if (filename.startsWith('count_')) {
    const columnName = filename.replace('count_', '').replace('.png', '');
    return `Count of Values in ${formatColumnName(columnName)}`;
  } else if (filename.includes('correlation_heatmap')) {
    return "Correlation Heatmap";
  }
  
  // Default fallback: clean up the filename
  return filename.replace('.png', '').replace(/_/g, ' ')
    .split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
}

// Helper function to format column names nicely
function formatColumnName(columnName) {
  return columnName.replace(/_/g, ' ')
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}
