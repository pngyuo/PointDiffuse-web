import React, { useEffect, useRef } from 'react';

const PointCloudViewer = ({ imageUrl }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!imageUrl) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
    };
    
    img.src = imageUrl;
  }, [imageUrl]);

  return (
    <div className="viewer-container">
      <canvas ref={canvasRef} className="point-cloud-canvas" />
    </div>
  );
};

export default PointCloudViewer;