import React, { useState, useCallback , useMemo} from 'react'; // Added useCallback
import { useLocation, Link } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import { FaPlayCircle, FaTimesCircle } from 'react-icons/fa'; // Added FaTimesCircle for consistency

import SongGraph from '../components/moodboard/SongGraph'; // <<< IMPORT
import ErrorBoundary from '../components/common/ErrorBoundary';
import '../components/moodboard/SongGraph.scss';
import './MoodboardPage.scss';
// Helper function to format key signature
const formatKeySignature = (key) => {
  if (!key) return '';
  return key.replace('_', ' ');
};

// Helper function to parse RAG source strings
const parseRagSource = (sourceString) => {
  if (sourceString.startsWith('KB: ')) {
    const pathWithPage = sourceString.substring(4);
    const pageMatch = pathWithPage.match(/\(Pg:(\d+)\)$/);
    let pageNumber = null;
    let path = pathWithPage;

    if (pageMatch) {
      pageNumber = pageMatch[1];
      path = pathWithPage.substring(0, pageMatch.index).trim();
    }
    const pathParts = path.split('/');
    const fileName = pathParts.pop() || path;
    const bookName = fileName.replace(/\.(pdf|txt|md)$/i, '').replace(/_/g, ' ');
    return {
      type: 'KB',
      raw: sourceString,
      bookName: bookName.trim() || "Unknown Document",
      page: pageNumber,
      category: pathParts.length > 0 ? pathParts[0] : "General",
    };
  } else if (sourceString.startsWith('SE(') && sourceString.endsWith(')')) {
    const content = sourceString.substring(3, sourceString.length - 1);
    const parts = content.split('): ');
    if (parts.length === 2) {
      return { type: 'SE', raw: sourceString, url: parts[0], title: parts[1] };
    }
    return { type: 'SE', raw: sourceString, url: content, title: "External Link" };
  }
  return { type: 'Unknown', raw: sourceString };
};


const MoodboardPage = () => {
  const location = useLocation();
  const moodboardData = location.state?.moodboardData;
  const [expandedTrackId, setExpandedTrackId] = useState(null);
  const [embeddedPlayerTrack, setEmbeddedPlayerTrack] = useState(null);

  if (!moodboardData) {
    return (
      <div className="moodboard-page-container error-container">
        <h2>No Moodboard Data</h2>
        <p>Moodboard data was not found. Please try generating one again.</p>
        <Link to="/" className="button-link">Go to Input Page</Link>
      </div>
    );
  }

  // Destructure with fallback for safety, though moodboardData check should prevent this
  const { 
    final_moodboard = '', 
    processed_similar_tracks = [], 
    rag_sources = [] 
  } = moodboardData;
    // console.log("Mooasdasdasdasd:", moodboardData.processed_similar_tracks); // Uncomment for debugging
  
  // Memoize sortedTracks so its reference only changes if processed_similar_tracks changes
  const sortedTracks = useMemo(() => {
    console.log("MOODBOARDPAGE: Re-calculating sortedTracks"); // For debugging
    return Array.isArray(processed_similar_tracks)
      ? [...processed_similar_tracks].sort((a, b) => (b.sim_score || 0) - (a.sim_score || 0))
      : [];
  }, [processed_similar_tracks]); // Dependency: only re-sort if the raw tracks change

  // Memoize categorizedSources as well if rag_sources could cause re-renders
  const categorizedSources = useMemo(() => {
    console.log("MOODBOARDPAGE: Re-calculating categorizedSources");
    const sources = { kb: [], se: [], other: [] };
    if (Array.isArray(rag_sources)) {
      rag_sources.forEach(sourceStr => {
        const parsed = parseRagSource(sourceStr);
        if (parsed.type === 'KB') sources.kb.push(parsed);
        else if (parsed.type === 'SE') sources.se.push(parsed);
        else sources.other.push(parsed);
      });
    }
    return sources;
  }, [rag_sources]);

  if (sortedTracks.length > 0) {
    console.log("Track QIDs for keys:", sortedTracks.map(t => t.qid));
    const qidSet = new Set(sortedTracks.map(t => t.qid).filter(qid => qid !== null && typeof qid !== 'undefined')); 
    const actualTrackCount = sortedTracks.length;
    

    let qidIssues = false;
    if (qidSet.size !== actualTrackCount) {
        console.error("!!! POTENTIAL QID ISSUES: Number of unique QIDs does not match track count, or null/undefined QIDs exist.");
        qidIssues = true;
    }

    const definedQids = sortedTracks.map(t => t.qid).filter(qid => qid !== null && typeof qid !== 'undefined');
    if (new Set(definedQids).size !== definedQids.length) {
        console.error("!!! DUPLICATE QIDs DETECTED among defined QIDs !!!");
        qidIssues = true;
    }

    if (qidIssues) {
        const qidCounts = sortedTracks.reduce((acc, t) => {
            const key = t.qid === null ? "null" : (typeof t.qid === 'undefined' ? "undefined" : t.qid);
            acc[key] = (acc[key] || 0) + 1;
            return acc;
        }, {});
        console.log("QID Counts (includes null/undefined as strings):", qidCounts);
    }
  }
  const toggleTrackExpansion = useCallback((trackIndex) => {
    setExpandedTrackId(prevId => (prevId === trackIndex ? null : trackIndex));
  }, []);

  const handlePlayTrack = useCallback((track, event) => {
    if (event) event.stopPropagation(); 
    // console.log(`Play track: ${track.title}`);
    setEmbeddedPlayerTrack(track);
  }, []);

  const closeEmbeddedPlayer = useCallback(() => {
    setEmbeddedPlayerTrack(null);
  }, []); 

  return (
    <div className="moodboard-page-container">
      <header className="moodboard-header">
        <h1>Your Music Moodboard</h1>
      </header>

      <section className="moodboard-section final-moodboard-section card-style">
        <h2 className="section-title">Inspiration Breakdown</h2>
        <div className="markdown-content">
          <ReactMarkdown>{final_moodboard}</ReactMarkdown>
        </div>
      </section>

      <section className="moodboard-section similar-tracks-section">
        <h2 className="section-title">Similar Tracks Analysis</h2>
        <div className="tracks-layout">
          <div className="graph-container card-style">
            <h3 className="subsection-title">Song Relation Graph</h3>
            <div className="graph-content">
              <ErrorBoundary>
                <SongGraph tracks={sortedTracks} />
              </ErrorBoundary>
            </div>
          </div>

          <div className="tracks-list-container custom-scrollbar-thin">
            <h3 className="subsection-title">Track List (Sorted by Similarity)</h3>
           
           {sortedTracks.length > 0 ? (
              sortedTracks.map((track, index) => {
                // Use index as the unique identifier for expansion instead of qid
                // This ensures each track has a unique expansion state regardless of qid issues
                const trackIndex = index;
                const keyForTrack = track.qid !== null && typeof track.qid !== 'undefined' ? track.qid : `track-${index}`; 
                
                if (track.qid === null || typeof track.qid === 'undefined') {
                  console.warn("Track found with null or undefined qid (potential key/comparison issue):", track.title || "Unknown Title");
                }
                const isExpanded = expandedTrackId === trackIndex;

                return (
                  <div key={keyForTrack} className={`track-card-wrapper ${isExpanded ? 'expanded' : ''}`}>
                    <div className="track-card card-style">
                      <img
                        src={track.album_art_url !== "https://via.placeholder.com/250?text=No+Art" ? track.album_art_url : `https://picsum.photos/seed/fallback${track.qid || Math.random()}/64/64`}
                        alt={track.title || 'Album Art'}
                        className="track-album-art"
                        onError={(e) => { e.target.onerror = null; e.target.src = `https://picsum.photos/seed/error${track.qid || Math.random()}/64/64`; }}
                      />
                      <div className="track-info" onClick={() => toggleTrackExpansion(trackIndex)}>
                        <h4 className="track-title" title={track.title}>{track.title || 'Unknown Title'}</h4>
                        <p className="track-artist" title={track.artist}>{track.artist || 'Unknown Artist'}</p>
                        <div className="track-meta-pills">
                          {track.sim_score && <span className="track-pill score-pill">Score: {track.sim_score.toFixed(2)}</span>}
                          {track.bpm && <span className="track-pill bpm-pill">{Math.round(track.bpm)} BPM</span>}
                          {track.key_signature && <span className="track-pill key-pill">{formatKeySignature(track.key_signature)}</span>}
                        </div>
                      </div>
                      {track.listenbrainz_url && (
                        <button
                          onClick={(e) => handlePlayTrack(track, e)}
                          className="play-button"
                          title="Play Track"
                          aria-label={`Play ${track.title || 'track'}`}
                        >
                          <FaPlayCircle size={28} />
                        </button>
                      )}
                    </div>
                    {isExpanded && (
                      <div className="track-details-expanded card-style">
                        {(track.tags && track.tags.length > 0) &&
                          <div className="detail-section">
                            <h5 className="detail-title">Tags:</h5>
                            <div className="tags-container">
                              {track.tags.map((tag, i) => <span key={`tag-${keyForTrack}-${i}`} className="tag-pill">{tag}</span>)}
                            </div>
                          </div>
                        }
                        {(track.genres && track.genres.length > 0) &&
                          <div className="detail-section">
                            <h5 className="detail-title">Genres:</h5>
                            <div className="tags-container">
                              {track.genres.map((genre, i) => <span key={`genre-${keyForTrack}-${i}`} className="tag-pill genre-pill">{genre}</span>)}
                            </div>
                          </div>
                        }
                        {!(track.tags && track.tags.length > 0) && !(track.genres && track.genres.length > 0) &&
                          <p className="no-details-text">No additional tags or genres available.</p>
                        }
                      </div>
                    )}
                  </div>
                );
              })
            ) : (
              <p className="no-tracks-message">No similar tracks found.</p>
            )}
          </div>
        </div>
      </section>

      {embeddedPlayerTrack && (
        <div className="embedded-player-overlay" onClick={closeEmbeddedPlayer}>
          <div className="embedded-player-container card-style" onClick={(e) => e.stopPropagation()}>
            <button onClick={closeEmbeddedPlayer} className="close-player-button" title="Close Player" aria-label="Close player">
              <FaTimesCircle size={26} /> {/* Using FaTimesCircle from react-icons */}
            </button>
            <h4 className="player-track-title">{embeddedPlayerTrack.title} - {embeddedPlayerTrack.artist}</h4>
            <iframe
              src={embeddedPlayerTrack.listenbrainz_url}
              width="100%"
              height="450"
              frameBorder="0"
              allowFullScreen
              allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
              loading="lazy"
              title={`ListenBrainz Player - ${embeddedPlayerTrack.title}`}
            ></iframe>
            <p className="iframe-note">If the player does not load, ListenBrainz may not allow embedding. You can <a href={embeddedPlayerTrack.listenbrainz_url} target="_blank" rel="noopener noreferrer">open it directly</a>.</p>
          </div>
        </div>
      )}

      {(categorizedSources.kb.length > 0 || categorizedSources.se.length > 0 || categorizedSources.other.length > 0) && (
        <section className="moodboard-section rag-sources-section card-style">
          <h2 className="section-title">Data Sources</h2>
          {categorizedSources.kb.length > 0 && (
            <div className="source-category">
              <h3 className="subsection-title source-category-title">Internal Knowledge</h3>
              <ul className="sources-list">
                {categorizedSources.kb.map((source, index) => (
                  <li key={`kb-${index}`} className="source-item kb-item">
                    <span className="book-name">{source.bookName}</span>
                    {source.category && <span className="category-tag">({source.category})</span>}
                    {source.page && <span className="page-number">Pg: {source.page}</span>}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {categorizedSources.se.length > 0 && (
            <div className="source-category">
              <h3 className="subsection-title source-category-title">External Knowledge (Stack Exchange)</h3>
              <ul className="sources-list">
                {categorizedSources.se.map((source, index) => (
                  <li key={`se-${index}`} className="source-item se-item">
                    <a href={source.url} target="_blank" rel="noopener noreferrer" className="external-link">
                      {source.title || source.url}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {categorizedSources.other.length > 0 && (
             <div className="source-category">
              <h3 className="subsection-title source-category-title">Other Sources</h3>
              <ul className="sources-list">
                {categorizedSources.other.map((source, index) => (
                  <li key={`other-${index}`} className="source-item other-item">
                    {source.raw}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}
      <div className="actions-footer">
        <Link to="/" className="button-link">Create Another Moodboard</Link>
      </div>
    </div>
  );
};

export default MoodboardPage;