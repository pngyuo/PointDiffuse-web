import React from 'react';

const ResultsPanel = ({ results }) => {
  if (!results) return null;

  // 获取后端基础URL（从环境变量或默认值）
  const apiBaseUrl = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

  return (
    <div className="results-panel">
      <h3>Segmentation Results</h3>
      
      <div className="stats">
        <p>Processing Time: {results.processing_time} seconds</p>
        <p>Points Processed: {results.points_count.toLocaleString()}</p>
      </div>
      
      <div className="distribution">
        <h4>Class Distribution</h4>
        <ul>
          {Object.entries(results.class_distribution).map(([cls, count]) => (
            <li key={cls}>
              Class {cls}: {count.toLocaleString()} points
            </li>
          ))}
        </ul>
      </div>
      
      <a 
        href={`${apiBaseUrl}${results.visualization}`} 
        download="segmentation_result.png"
        className="download-button"
      >
        Download Full Visualization
      </a>
    </div>
  );
};

export default ResultsPanel;