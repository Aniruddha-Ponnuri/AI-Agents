import { useState } from 'react';
import axios from 'axios';
import { FaPaperPlane } from 'react-icons/fa';

export default function QueryInterface({ fileData }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim() || !fileData) return;
    
    setLoading(true);
    setError(null);
    
    try {
      // Add user query to the results
      setResults(prev => [...prev, { type: 'user', content: query }]);
      
      // Create FormData for the API - THIS IS THE KEY FIX
      const formData = new FormData();
      formData.append('file_path', fileData.file_path);
      formData.append('query', query.trim());
      
      // Send with proper headers
      const response = await axios.post('/api/query', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      console.log('Query response:', response.data);
      
      // Extract the answer
      const answer = extractAnswer(response.data.result);
      
      // Add AI response to the results
      setResults(prev => [...prev, { 
        type: 'ai', 
        content: answer,
        visualization: response.data.result?.visualization || null
      }]);
      
      // Clear the input
      setQuery('');
    } catch (err) {
      console.error('Error querying data:', err);
      setError('Error processing your query. Please try again.');
      
      // Add error message to chat
      setResults(prev => [...prev, { 
        type: 'ai', 
        content: `Error: ${err.response?.data?.error || 'Could not process your query. Please try again.'}`,
        isError: true
      }]);
    } finally {
      setLoading(false);
    }
  };

  // Function to extract answer from different response formats
  const extractAnswer = (result) => {
    if (!result) return "Sorry, I couldn't process your query.";
    
    if (typeof result === 'string') return result;
    if (result.answer) return result.answer;
    
    if (result.tasks_output) {
      const task = result.tasks_output.find(t => 
        t.agent === "Data Analyst" || t.description?.includes("query")
      );
      if (task?.raw) {
        const match = task.raw.match(/## Final Answer:\s*([\s\S]*)/);
        if (match) return match[1].trim();
        return task.raw;
      }
    }
    
    return JSON.stringify(result);
  };

  if (!fileData) return null;

  return (
    <div>
      {/* Chat Messages */}
      <div className="chat-messages space-y-4 mb-4 max-h-[300px] overflow-y-auto">
        {results.map((result, index) => (
          <div key={index} className={`flex ${result.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] p-3 rounded-lg ${
              result.type === 'user' 
                ? 'bg-blue-100 text-blue-900' 
                : result.isError
                  ? 'bg-red-100 text-red-900'
                  : 'bg-gray-100 text-gray-900'
            }`}>
              <div className="whitespace-pre-line">{result.content}</div>
              
              {result.visualization && (
                <div className="mt-3">
                  <img 
                    src={`http://localhost:8000${result.visualization}`} 
                    alt="Data visualization" 
                    className="max-w-full rounded border border-gray-200" 
                  />
                </div>
              )}
            </div>
          </div>
        ))}
        
        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 p-3 rounded-lg">
              <div className="flex space-x-2">
                <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce"></div>
                <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce delay-75"></div>
                <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce delay-150"></div>
              </div>
            </div>
          </div>
        )}
      </div>
      
      {/* Chat Input */}
      <form onSubmit={handleSubmit} className="mt-2">
        <div className="chat-input-container">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question..."
            className="chat-input"
            disabled={loading}
          />
          <button 
            type="submit" 
            className="chat-button"
            disabled={loading || !query.trim()}
          >
            <FaPaperPlane />
          </button>
        </div>
        
        {error && (
          <div className="mt-2 text-sm text-red-600">
            {error}
          </div>
        )}
      </form>
    </div>
  );
}
