// src/App.jsx
import React from 'react';
import { Routes, Route } from 'react-router-dom';
import MainLayout from './components/layout/MainLayout';
import InputPage from './pages/InputPage';
import MoodboardPage from './pages/MoodboardPage'; // We'll create this next

function App() {
  return (
    <MainLayout>
      <Routes>
        <Route path="/" element={<InputPage />} />
        <Route path="/moodboard" element={<MoodboardPage />} />

      </Routes>
    </MainLayout>
  );
}

export default App;