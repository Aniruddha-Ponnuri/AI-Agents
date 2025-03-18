import { useState } from 'react';
import axios from 'axios';

export default function QueryInterface({ fileData }) {
  const [query, setQuery] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim() || !fileData) return;
    
    setLoading(true);
    setError(null);
    
    try {
      // Instead of FormData, use a regular object for Next.js API routes
      const requestData = {
        file_path: fileData.file_path,
        query: query.trim()
      };
      
      const response = await axios.post('/api/query', requestData);
      
      setResult(response.data.result);
    } catch (err) {
      console.error('Error querying data:', err);
      
      // Improved error handling with specific messages
      if (err.response && err.response.data && err.response.data.error) {
        setError(err.response.data.error);
      } else {
        setError('Error processing your query. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  if (!fileData) {
    return null;
  }

  return (
    <div className="mt-10 border-t pt-8">
      <h2 className="text-2xl font-bold mb-4">Ask Questions About Your Data</h2>
      
      <form onSubmit={handleSubmit} className="mb-6">
        <div className="flex gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="E.g., What's the distribution of values in Column 1?"
            className="flex-grow p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-blue-300"
          >
            {loading ? 'Processing...' : 'Ask'}
          </button>
        </div>
      </form>
      
      {error && (
        <div className="p-3 mb-4 bg-red-100 border border-red-200 text-red-700 rounded">
          {error}
        </div>
      )}
      
      {result && (
        <div className="bg-gray-50 border rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-3">Answer:</h3>
          <p className="whitespace-pre-line">{result.answer}</p>
          
          {result.visualization && (
            <div className="mt-4">
              <h4 className="text-md font-medium mb-2">Visualization:</h4>
              <img 
                src={`http://localhost:8000${result.visualization}`} 
                alt="Query visualization" 
                className="max-w-full h-auto border rounded" 
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
