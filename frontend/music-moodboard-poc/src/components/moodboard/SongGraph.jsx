// src/components/moodboard/SongGraph.jsx
import React, { useRef, useState, useEffect, useMemo, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import * as d3 from 'd3-force';
import './SongGraph.scss';

// Qdrant configuration
const QDRANT_API_URL = 'http://localhost:6333';
const QDRANT_COLLECTION_NAME = 'zscore_94';
const QDRANT_API_KEY = null;
const INPUT_ID = 'input_audio';
const placeholder = id => `https://picsum.photos/seed/${id}/80/80`;

/**
 * Retrieve multiple points by ID, including vectors
 */
async function fetchVectorsById(ids) {
  if (!Array.isArray(ids) || ids.length === 0) return [];
  const endpoint = `${QDRANT_API_URL}/collections/${QDRANT_COLLECTION_NAME}/points`;
  const payload = { ids: ids.map(id => isNaN(id) ? id : Number(id)), with_vector: true };
  const headers = { 'Content-Type': 'application/json' };
  if (QDRANT_API_KEY) headers['api-key'] = QDRANT_API_KEY;

  try {
    const resp = await fetch(endpoint, { method: 'POST', headers, body: JSON.stringify(payload) });
    if (!resp.ok) {
      console.error('Qdrant points fetch error', resp.status, await resp.text());
      return [];
    }
    const data = await resp.json();
    return Array.isArray(data.result) ? data.result : [];
  } catch (e) {
    console.error('Error fetching Qdrant points by ID', e);
    return [];
  }
}

/**
 * Cosine similarity between two vectors
 */
function cosineSimilarity(v1, v2) {
  let dot = 0, mag1 = 0, mag2 = 0;
  for (let i = 0; i < v1.length; i++) {
    dot += v1[i] * (v2[i] || 0);
    mag1 += v1[i] * v1[i];
    mag2 += (v2[i] || 0) * (v2[i] || 0);
  }
  return mag1 && mag2 ? dot / (Math.sqrt(mag1) * Math.sqrt(mag2)) : 0;
}

export default function SongGraph({ tracks = [] }) {
  const containerRef = useRef(null);
  const fgRef = useRef(null);
  const [size, setSize] = useState({ width: 600, height: 600 });
  const [vectors, setVectors] = useState({});
  const [allPairs, setAllPairs] = useState([]);
  const [links, setLinks] = useState([]);
  const [selected, setSelected] = useState(null);

  // Handle container resize
  useEffect(() => {
    const update = () => {
      if (containerRef.current) {
        setSize({ width: containerRef.current.offsetWidth, height: containerRef.current.offsetHeight });
      }
    };
    update();
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, []);

  // Build node list with fixed center for user audio
  const nodes = useMemo(() => {
    const cx = size.width / 2;
    const cy = size.height / 2;
    return [
      { id: INPUT_ID, fx: cx, fy: cy, title: 'Your Audio' },
      ...tracks.map(t => ({
        id: String(t.qid),
        title: t.title,
        img: t.album_art_url || placeholder(t.qid),
        value: t.sim_score || 0
      }))
    ];
  }, [tracks, size]);

  // Create primary links (input -> each track)
  const primaryLinks = useMemo(
    () => nodes.filter(n => n.id !== INPUT_ID).map(n => ({
      source: INPUT_ID,
      target: n.id,
      value: n.value,
      distance: 1 - n.value
    })),
    [nodes]
  );

  // Fetch vectors and compute pairwise similarities
  useEffect(() => {
    const ids = tracks.map(t => String(t.qid));
    fetchVectorsById(ids).then(points => {
      const vecMap = {};
      points.forEach(p => { if (p.id && p.vector) vecMap[String(p.id)] = p.vector; });
      setVectors(vecMap);
    });
  }, [tracks]);

  useEffect(() => {
    const ids = Object.keys(vectors);
    if (ids.length < 2) {
      setAllPairs([]);
      setLinks(primaryLinks);
      setSelected(null);
      return;
    }
    const pairs = [];
    for (let i = 0; i < ids.length; i++) {
      for (let j = i + 1; j < ids.length; j++) {
        const a = ids[i], b = ids[j];
        const sim = cosineSimilarity(vectors[a], vectors[b]);
        pairs.push({ source: a, target: b, value: sim, distance: 1 - sim });
      }
    }
    setAllPairs(pairs);
    setLinks(primaryLinks);
    setSelected(null);
  }, [vectors, primaryLinks]);

  // Node click toggles secondary links
  const onNodeClick = useCallback(node => {
    if (node.id === INPUT_ID) return;
    if (selected === node.id) {
      setLinks(primaryLinks);
      setSelected(null);
    } else {
      setSelected(node.id);
      const secondary = allPairs.filter(l => l.source === node.id || l.target === node.id);
      setLinks([...primaryLinks, ...secondary]);
    }
    fgRef.current?.d3ReheatSimulation();
  }, [primaryLinks, allPairs, selected]);

  // Preload cover art images
  const imgCache = useRef({});
  useEffect(() => {
    nodes.forEach(n => {
      if (n.img && imgCache.current[n.img] === undefined) {
        const img = new Image(); img.src = n.img;
        img.onload = () => { imgCache.current[n.img] = img; fgRef.current?.d3ReheatSimulation(); };
        img.onerror = () => { imgCache.current[n.img] = null; };
        imgCache.current[n.img] = undefined;
      }
    });
  }, [nodes]);

  // Configure force simulation for stability
  useEffect(() => {
    if (!fgRef.current) return;
    const graph = fgRef.current;
    graph.d3Force('center', d3.forceCenter(size.width / 2, size.height / 2));
    graph.d3Force('charge', d3.forceManyBody().strength(-200));
    graph.d3Force('link', d3.forceLink().id(d => d.id)
      .distance(l => l.distance * Math.min(size.width, size.height) * 0.5)
      .strength(l => Math.max(0.1, l.value))
    );
    graph.d3Force('collision', d3.forceCollide().radius(20));
    graph.d3ReheatSimulation();
  }, [size, links]);

  // Draw nodes as circular images
  const paintNode = useCallback((node, ctx) => {
    const R = node.id === INPUT_ID ? 24 : 18;
    ctx.save(); ctx.beginPath(); ctx.arc(node.x, node.y, R, 0, 2 * Math.PI); ctx.clip();
    if (node.id === INPUT_ID) { ctx.fillStyle = '#fff'; ctx.fill(); }
    else {
      const img = imgCache.current[node.img];
      if (img && img.complete) ctx.drawImage(img, node.x - R, node.y - R, R * 2, R * 2);
      else { ctx.fillStyle = '#888'; ctx.fill(); }
    }
    ctx.restore();
  }, []);

  // Draw links colored by similarity
  const paintLink = useCallback((link, ctx) => {
    const sim = link.value ?? 0;
    const r = Math.round(200 * (1 - sim));
    const g = Math.round(100 + sim * 155);
    const alpha = 0.4 + sim * 0.6;
    ctx.strokeStyle = `rgba(${r},${g},50,${alpha})`;
    ctx.lineWidth = 1 + sim * 4;
    ctx.beginPath(); ctx.moveTo(link.source.x, link.source.y); ctx.lineTo(link.target.x, link.target.y); ctx.stroke();
  }, []);

  return (
    <div ref={containerRef} className="song-graph-container">
      <ForceGraph2D
        ref={fgRef}
        width={size.width}
        height={size.height}
        graphData={{ nodes, links }}
        d3AlphaDecay={0.02}
        d3AlphaMin={0.001}
        d3VelocityDecay={0.3}
        nodeCanvasObject={paintNode}
        linkCanvasObject={paintLink}
        onNodeClick={onNodeClick}
        enableNodeDrag
        cooldownTicks={150}
        backgroundColor="transparent"
      />
    </div>
  );
}
