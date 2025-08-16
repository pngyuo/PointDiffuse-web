import React, { useState } from 'react';

const FileUpload = ({ onFileProcessed }) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setIsProcessing(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await fetch('http://localhost:5000/process', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(await response.text());
      }
      
      const data = await response.json();
      onFileProcessed(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="file-upload">
      <label className="upload-button">
        {isProcessing ? 'Processing...' : 'Upload Point Cloud'}
        <input 
          type="file" 
          onChange={handleFileChange} 
          accept=".npy,.npz" 
          disabled={isProcessing}
          hidden
        />
      </label>
      {error && <div className="error">{error}</div>}
    </div>
  );
};

export default FileUpload;