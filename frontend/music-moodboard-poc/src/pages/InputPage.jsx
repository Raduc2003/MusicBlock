// src/pages/InputPage.jsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Loader from '../components/common/Loader'; // Import the Loader
import './InputPage.scss';

const InputPage = () => {
  const navigate = useNavigate();
  const [textQuery, setTextQuery] = useState('');
  const [audioFile, setAudioFile] = useState(null);
  const [audioFileName, setAudioFileName] = useState('');
  const [isLoading, setIsLoading] = useState(false); // This state is key
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    // ... (same as before)
    const file = event.target.files[0];
    if (file) {
      setAudioFile(file);
      setAudioFileName(file.name);
      setError(null); // Clear error if a new file is selected
    } else {
      setAudioFile(null);
      setAudioFileName('');
    }
  };

  const handleSubmit = async (event) => {
    // ... (same as before, including setIsLoading(true) and setIsLoading(false))
    event.preventDefault();
    setError(null);

    if (!audioFile) {
      setError('Please select an audio file.');
      return;
    }
    if (!textQuery.trim()) {
      setError('Please enter a text query or a general mood.');
      return;
    }

    setIsLoading(true); // <--- Loader will appear here

    const formData = new FormData();
    formData.append('audio_file', audioFile);
    formData.append('user_text_query', textQuery);

    try {
      const response = await fetch('http://localhost:8080/create_inspiration_moodboard', {
        method: 'POST',
        body: formData,
      });

      // setIsLoading(false) will be called in both success and error cases
      // to ensure the loader is hidden.

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ message: `HTTP error! Status: ${response.status}. Could not parse error response.` }));
        const errorMessage = errorData?.detail || errorData?.error || errorData?.message || `API request failed with status: ${response.status}`;
        setIsLoading(false); // Hide loader on error
        throw new Error(errorMessage);
      }

      const result = await response.json();
      setIsLoading(false); // Hide loader on success
      console.log('API Response:', result);
      navigate('/moodboard', { state: { moodboardData: result } });

    } catch (err) {
      setIsLoading(false); // Hide loader on exception
      console.error('API Error:', err);
      setError(err.message || 'An unexpected error occurred during analysis.');
    }
  };

  return (
    <> {/* Use a fragment to allow Loader to be at the top level */}
      {isLoading && <Loader message="Crafting your moodboard... this may take a minute or two." />} {/* Render Loader when isLoading is true */}
      
      <div className={`input-page-container ${isLoading ? 'is-loading' : ''}`}> {/* Optionally blur/disable page when loading */}
        <header className="input-page-header">
          <h1>
            <span className="text-gradient-neon-pink-purple">Upload & Describe</span>
            <span className="header-subtitle">Your Music</span>
          </h1>
          <p className="header-tagline">Let's find the perfect moodboard for your sound.</p>
        </header>

        <main className="input-form-main">
          <form className="input-form" onSubmit={handleSubmit}>
            {/* ... (rest of your form JSX: file input, text area) ... */}
            <div className="form-group">
            <label htmlFor="audio-upload-input" className="form-label">
              1. Upload Your Audio Track
            </label>
            <div className="file-drop-area">
              <svg className="file-drop-icon" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true">
                <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              <div className="file-upload-text">
                <label htmlFor="audio-upload-input" className="file-upload-button">
                  <span>{audioFileName ? 'Change file' : 'Upload a file'}</span>
                </label>
                <input
                  id="audio-upload-input"
                  name="audio-upload"
                  type="file"
                  className="sr-only"
                  accept="audio/mp3, audio/wav, audio/mpeg, audio/ogg, audio/aac"
                  onChange={handleFileChange}
                  key={audioFile ? audioFile.name : 'empty-file-input'} // Helps reset input if file is cleared
                  disabled={isLoading} // Disable input when loading
                />
                <p className="file-drop-hint">or drag and drop</p>
              </div>
              <p className="file-types-info">MP3, WAV, OGG, AAC up to 50MB</p>
              {audioFileName && <p className="selected-file-name">{audioFileName}</p>}
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="moodDescription" className="form-label">
              2. Describe the Vibe
            </label>
            <textarea
              id="moodDescription"
              name="moodDescription"
              rows="6"
              className="mood-textarea"
              placeholder="e.g., 'Energetic synthwave for a night drive', 'Chill lo-fi beat for studying'..."
              value={textQuery}
              onChange={(e) => setTextQuery(e.target.value)}
              disabled={isLoading} // Disable input when loading
            />
          </div>

          {error && !isLoading && <p className="error-message">{error}</p>} {/* Only show error if not loading */}
            
            <div className="form-actions">
              <button type="submit" className="submit-button" disabled={isLoading}>
                {isLoading ? 'Analyzing...' : 'Analyze & Create Moodboard'}
              </button>
            </div>
          </form>
        </main>
      </div>
    </>
  );
};

export default InputPage;