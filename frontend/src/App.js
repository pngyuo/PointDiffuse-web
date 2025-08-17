import React, { useState, useEffect } from 'react';
import FileUpload from './components/FileUpload';
import PointCloudViewer from './components/PointCloudViewer';
import ResultsPanel from './components/ResultsPanel';
import './App.css';

function App() {
  const [results, setResults] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState(null);

  // 检查后端连接
  useEffect(() => {
    const checkBackendConnection = async () => {
      try {
        const apiBaseUrl = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';
        const response = await fetch(`${apiBaseUrl}/`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });
        
        if (response.ok) {
          setIsConnected(true);
          setConnectionError(null);
        } else {
          throw new Error(`Backend returned ${response.status}`);
        }
      } catch (error) {
        console.error('Backend connection error:', error);
        setIsConnected(false);
        setConnectionError(error.message);
      }
    };

    checkBackendConnection();
    // 每30秒检查一次连接
    const interval = setInterval(checkBackendConnection, 30000);
    
    return () => clearInterval(interval);
  }, []);

  const handleFileProcessed = (data) => {
    setResults(data);
  };

  const handleReset = () => {
    setResults(null);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>PointDiffuse Web Demo</h1>
        <p>Upload a point cloud file (.npy or .npz) for semantic segmentation</p>
        
        {/* 连接状态指示器 */}
        <div className="connection-status">
          {isConnected ? (
            <div className="status-connected">✓ Backend Connected</div>
          ) : (
            <div className="status-disconnected">
              ⚠ Backend Disconnected
              {connectionError && <div className="error-details">Error: {connectionError}</div>}
            </div>
          )}
        </div>
      </header>

      <main className="App-main">
        {!results ? (
          <div className="upload-section">
            <FileUpload onFileProcessed={handleFileProcessed} />
            <div className="instructions">
              <h3>Instructions:</h3>
              <ul>
                <li>Upload a point cloud file in .npy or .npz format</li>
                <li>File should contain point coordinates (x, y, z) and optionally colors (r, g, b)</li>
                <li>Maximum file size: 100MB</li>
                <li>Processing time depends on the number of points</li>
              </ul>
            </div>
          </div>
        ) : (
          <div className="results-section">
            <div className="results-container">
              <div className="viewer-section">
                <h2>Segmentation Visualization</h2>
                <PointCloudViewer 
                  imageUrl={`${process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000'}${results.visualization}`} 
                />
              </div>
              
              <div className="panel-section">
                <ResultsPanel results={results} />
                <button onClick={handleReset} className="reset-button">
                  Process Another File
                </button>
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="App-footer">
        <p>Powered by PointDiffuse - Dual-Conditional Diffusion Model for Point Cloud Semantic Segmentation</p>
      </footer>
    </div>
  );
}

export default App;