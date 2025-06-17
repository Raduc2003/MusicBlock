// src/components/common/Loader.jsx
import React from 'react';
import './Loader.scss'; // We'll create this SCSS file

const Loader = ({ message = "Analyzing your music, please wait..." }) => {
  return (
    <div className="loader-overlay">
      <div className="loader-content">
        <div className="spinner"></div>
        <p className="loader-message">{message}</p>
      </div>
    </div>
  );
};

export default Loader;