import axios from "axios";

export default async function handler(req, res) {
  const { datasetName } = req.query;

  if (!datasetName) {
    return res.status(400).json({ error: "Dataset name is required" });
  }

  try {
    // Forward the request to the backend API
    const response = await axios.get(
      `http://localhost:8000/api/visualizations/${datasetName}`
    );
    return res.status(200).json(response.data);
  } catch (error) {
    console.error("Error fetching visualizations:", error);
    return res.status(500).json({ error: "Failed to fetch visualizations" });
  }
}
