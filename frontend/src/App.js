import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import PointCloudViewer from './components/PointCloudViewer';
import ResultsPanel from './components/ResultsPanel';
import './App.css';

function App() {
  const [results, setResults] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  
  // 添加环境变量处理
  const apiBaseUrl = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

  const handleFileProcessed = (data) => {
    setResults(data);
    // 使用完整URL设置预览图
    setPreviewUrl(`${apiBaseUrl}${data.visualization}`);
  };

  return (
    <div className="App">
      <header>
        <h1>PointDiffuse Semantic Segmentation</h1>
        <p>Upload a point cloud file (.npy) to see semantic segmentation results</p>
      </header>
      
      <main>
        <FileUpload onFileProcessed={handleFileProcessed} />
        
        <div className="results-container">
          <div className="visualization">
            {previewUrl ? (
              <PointCloudViewer imageUrl={previewUrl} />
            ) : (
              <div className="placeholder">Visualization will appear here</div>
            )}
          </div>
          
          {/* 传递apiBaseUrl给ResultsPanel组件 */}
          <ResultsPanel results={results} apiBaseUrl={apiBaseUrl} />
        </div>
      </main>
      
      <footer>
        <p>Powered by PointDiffuse Diffusion Model</p>
      </footer>
    </div>
  );
}

export default App;