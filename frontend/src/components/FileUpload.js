import React, { useState } from 'react';

const FileUpload = ({ onFileProcessed }) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    // 检查文件类型
    const allowedTypes = ['npy', 'npz'];
    const fileExtension = file.name.split('.').pop().toLowerCase();
    
    if (!allowedTypes.includes(fileExtension)) {
      setError('Please upload a .npy or .npz file');
      return;
    }

    // 检查文件大小（100MB限制）
    if (file.size > 100 * 1024 * 1024) {
      setError('File size must be less than 100MB');
      return;
    }
    
    setIsProcessing(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      // 获取API基础URL
      const apiBaseUrl = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';
      
      const response = await fetch(`${apiBaseUrl}/process`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.status === 'success') {
        onFileProcessed(data);
      } else {
        throw new Error(data.error || 'Processing failed');
      }
      
    } catch (err) {
      console.error('Upload error:', err);
      setError(`Upload failed: ${err.message}`);
    } finally {
      setIsProcessing(false);
      // 清除文件选择
      e.target.value = '';
    }
  };

  return (
    <div className="file-upload">
      <label className="upload-button">
        {isProcessing ? 'Processing...' : 'Upload Point Cloud (.npy, .npz)'}
        <input 
          type="file" 
          onChange={handleFileChange} 
          accept=".npy,.npz" 
          disabled={isProcessing}
          hidden
        />
      </label>
      {error && <div className="error">{error}</div>}
      {isProcessing && (
        <div className="processing-info">
          <div className="spinner"></div>
          <p>Processing your point cloud... This may take a few seconds.</p>
        </div>
      )}
    </div>
  );
};

export default FileUpload;