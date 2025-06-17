// src/components/layout/MainLayout.jsx
import React from 'react';
import Header from './Header';
import Footer from './Footer';

const MainLayout = ({ children }) => {
  return (
    <>
      <div className="noise-overlay"></div>
      <div className="app-content-wrapper">
        <Header /> {/* Add Header */}
        <main className="main-content">
          {children}
        </main>
        <Footer /> {/* Add Footer */}
      </div>
    </>
  );
};

export default MainLayout;