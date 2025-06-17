import React from 'react';
import './Footer.scss'; // We'll create this

const Footer = () => {
  return (
    <footer className="site-footer">
      <p>© {new Date().getFullYear()} MusicMoodboard Inc. All rights reserved.</p>
    </footer>
  );
};

export default Footer;