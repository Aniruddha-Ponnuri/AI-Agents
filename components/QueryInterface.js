import { useState } from "react";
import axios from "axios";
import { FaPaperPlane } from "react-icons/fa";

export default function QueryInterface({ fileData }) {
  const [query, setQuery] = useState("");
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
      setResults((prev) => [...prev, { type: "user", content: query }]);

      // Create FormData for the API
      const formData = new FormData();
      formData.append("file_path", fileData.file_path);
      formData.append("query", query.trim());

      // Send with proper headers
      const response = await axios.post("/api/query", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      console.log("Query response:", response.data);

      // Extract the answer
      const answer = extractAnswer(response.data.result);

      // Add AI response to the results
      setResults((prev) => [
        ...prev,
        {
          type: "bot",
          content: answer,
          visualization: response.data.result?.visualization || null,
        },
      ]);

      // Clear the input
      setQuery("");
    } catch (err) {
      console.error("Error querying data:", err);
      setError("Error processing your query. Please try again.");

      // Add error message to chat
      setResults((prev) => [
        ...prev,
        {
          type: "bot",
          content: `Error: ${
            err.response?.data?.error ||
            "Could not process your query. Please try again."
          }`,
          isError: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Function to extract answer from different response formats
  const extractAnswer = (result) => {
    if (!result) return "Sorry, I couldn't process your query.";

    if (typeof result === "string") return result;
    if (result.answer) return result.answer;

    if (result.tasks_output) {
      const task = result.tasks_output.find(
        (t) =>
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
    <div className="chat-container">
      {/* Chat Messages */}
      <div className="chat-messages flex flex-col space-y-6 mb-4 overflow-y-auto p-4">
        {results.map((result, index) => (
          <div key={index} className="w-full">
            {/* Message bubble with sender alignment */}
            <div className={`flex ${result.type === "user" ? "justify-end" : "justify-start"}`}>
              <div className={`${result.type === "user" ? "user-bubble" : "bot-bubble"}`}>
                <div className="whitespace-pre-line">{result.content}</div>
                
                {result.visualization && (
                  <div className="mt-3">
                    <img src={`http://localhost:8000${result.visualization}`} 
                        alt="Data visualization" 
                        className="max-w-full rounded border border-gray-200" />
                  </div>
                )}
              </div>
            </div>
            
            {/* Sender label */}
            <div className={`text-xs text-gray-500 mt-1 ${result.type === "user" ? "text-right mr-2" : "text-left ml-2"}`}>
              {result.type === "user" ? "You" : "Bot"}
            </div>
          </div>
        ))}

        {/* Loading indicator */}
        {loading && (
          <div className="flex justify-start w-full mt-4">
            <div>
              <div className="bot-bubble">
                <div className="flex space-x-2">
                  <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce"></div>
                  <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce delay-75"></div>
                  <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce delay-150"></div>
                </div>
              </div>
              <div className="text-xs text-gray-500 mt-1 text-left ml-2">Bot</div>
            </div>
          </div>
        )}
      </div>

      {/* Chat input area */}
      <div className="chat-input-wrapper">
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
        </form>
      </div>
    </div>
  );
}
