import React from 'react';
import './Header.scss'; // We'll create this

const Header = () => {
  return (
    <header className="site-header">
      <div className="header-content">
        <div className="logo text-gradient-neon-pink-purple">MusicMood</div>
        {/* Navigation or other elements can go here */}
      </div>
    </header>
  );
};

export default Header;