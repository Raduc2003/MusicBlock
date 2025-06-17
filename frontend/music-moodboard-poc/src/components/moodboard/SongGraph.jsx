import React, { useEffect, useRef, useCallback, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

const QDRANT_API_URL = 'http://localhost:6333'; // Base URL for your Qdrant instance
const QDRANT_COLLECTION_NAME = 'zscore_94'; // YOUR COLLECTION NAME
const QDRANT_API_KEY = null; // YOUR QDRANT API KEY if required

const fetchQdrantDistanceMatrix = async (trackQids) => {
  if (!trackQids || trackQids.length < 2) {
    return [];
  }
  const endpoint = `${QDRANT_API_URL}/collections/${QDRANT_COLLECTION_NAME}/points/distances-batch`; // EXAMPLE
  const requestBody = { point_ids: trackQids };
  const headers = { 'Content-Type': 'application/json' };
  if (QDRANT_API_KEY) {
    headers['api-key'] = QDRANT_API_KEY;
  }

  try {
    console.log("GRAPH: Simulating Qdrant fetch for QIDs:", trackQids);
    await new Promise(resolve => setTimeout(resolve, 500)); 
    const qdrantResult = { status: "ok", time: 0.1, result: [] };

    for (let i = 0; i < trackQids.length; i++) {
      for (let j = i + 1; j < trackQids.length; j++) {
        if (Math.random() > 0.65) { // Adjust link density
          qdrantResult.result.push({
            id1: trackQids[i],
            id2: trackQids[j],
            distance: Math.random() * 0.7 + 0.2 
          });
        }
      }
    }
    console.log("GRAPH: Mock Qdrant-like distance result:", qdrantResult.result.slice(0, 3));
    
    if (qdrantResult.status === "ok" && Array.isArray(qdrantResult.result)) {
      const links = qdrantResult.result.map(item => {
        const similarity = 1 - (item.distance || 0.5); // Ensure distance is defined
        return {
          source: item.id1, 
          target: item.id2,
          value: Math.max(0, similarity), // Ensure similarity isn't negative
          distance: item.distance
        };
      }).filter(link => link.value > 0.2); // Adjust similarity threshold

      console.log("GRAPH: Transformed links:", links.slice(0, 3));
      return links;
    } else {
      console.error("GRAPH: Unexpected Qdrant response format or error status.");
      return [];
    }
  } catch (error) {
    console.error('GRAPH: Error fetching or processing Qdrant distances:', error);
    return [];
  }
};


const SongGraph = ({ tracks = [] }) => {
  const fgRef = useRef();
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [hoveredNode, setHoveredNode] = useState(null);
  const [loadedImages, setLoadedImages] = useState({});
  const [containerSize, setContainerSize] = useState({ width: 300, height: 350 }); // Default size
  const containerRef = useRef(null); // Ref for the graph container div


  // Effect to update graph size based on its container
  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        setContainerSize({
          width: containerRef.current.offsetWidth,
          height: containerRef.current.offsetHeight
        });
      }
    };
    updateSize(); // Initial size
    window.addEventListener('resize', updateSize);
    return () => window.removeEventListener('resize', updateSize);
  }, []);


  useEffect(() => {
    if (!Array.isArray(tracks) || tracks.length === 0) {
      setGraphData({ nodes: [], links: [] });
      return;
    }

    const nodes = tracks.map(track => ({
      id: track.qid, // CRITICAL: This MUST be unique for each track
      title: track.title,
      artist: track.artist,
      imgUrl: track.album_art_url !== "https://via.placeholder.com/250?text=No+Art" 
              ? track.album_art_url 
              : `https://picsum.photos/seed/graphNode${track.qid || String(Math.random()).slice(2)}/60/60`, // Ensure fallback has unique seed
      ...track
    }));

    const imagePromises = nodes.map(node => {
      return new Promise((resolve) => {
        if (loadedImages[node.id]?.complete) { // Check if already loaded and complete
          resolve(); return;
        }
        const img = new Image();
        img.src = node.imgUrl;
        img.onload = () => { setLoadedImages(prev => ({ ...prev, [node.id]: img })); resolve(); };
        img.onerror = () => {
          // console.warn(`Graph: Failed image load for node ${node.id}: ${node.imgUrl}`);
          const fallbackImg = new Image();
          fallbackImg.src = `https://picsum.photos/seed/errorGraphNode${node.id || String(Math.random()).slice(2)}/60/60`;
          fallbackImg.onload = () => setLoadedImages(prev => ({ ...prev, [node.id]: fallbackImg }));
          fallbackImg.onerror = () => resolve(); // Still resolve if fallback fails
          resolve();
        };
      });
    });

    Promise.all(imagePromises).then(() => {
      const trackQids = nodes.map(n => n.id).filter(id => id != null); 
      if (trackQids.length > 1) {
        fetchQdrantDistanceMatrix(trackQids).then(linksFromQdrant => {
          setGraphData({ nodes, links: linksFromQdrant });
        });
      } else {
        setGraphData({ nodes, links: [] });
      }
    });

  }, [tracks]); // `loadedImages` removed from deps intentionally to avoid re-fetching links on image loads

  const baseNodeSize = 40; // Reduced base size
  const hoverScale = 1.3; // Increased hover scale for better feedback
  const minNodeSize = 20; // Minimum node size when zoomed out
  const maxNodeSize = 80; // Maximum node size when zoomed in

  const handleNodeHover = useCallback((node) => {
    setHoveredNode(node);
  }, []);
  
  const nodeCanvasObject = useCallback((node, ctx, globalScale) => {
    if (typeof node.x !== 'number' || typeof node.y !== 'number' || !isFinite(node.x) || !isFinite(node.y)) {
      return; 
    }

    const isHovered = hoveredNode?.id === node.id;
    
    // Check if this is the center node (highest similarity score)
    const isCenterNode = node.sim_score && node.sim_score >= 0.9; // Nodes with score >= 0.9 are "center" nodes
    
    // Scale node size based on zoom level, with min/max constraints
    let scaledNodeSize = baseNodeSize / Math.max(0.5, Math.min(3, globalScale));
    scaledNodeSize = Math.max(minNodeSize, Math.min(maxNodeSize, scaledNodeSize));
    
    // Center nodes are slightly larger
    if (isCenterNode) {
      scaledNodeSize *= 1.15;
    }
    
    const currentSize = isHovered ? scaledNodeSize * hoverScale : scaledNodeSize;
    const halfSize = currentSize / 2;
    const borderRadius = Math.max(4, 12 / globalScale); // Scale border radius with zoom

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(node.x - halfSize + borderRadius, node.y - halfSize);
    ctx.lineTo(node.x + halfSize - borderRadius, node.y - halfSize);
    ctx.quadraticCurveTo(node.x + halfSize, node.y - halfSize, node.x + halfSize, node.y - halfSize + borderRadius);
    ctx.lineTo(node.x + halfSize, node.y + halfSize - borderRadius);
    ctx.quadraticCurveTo(node.x + halfSize, node.y + halfSize, node.x + halfSize - borderRadius, node.y + halfSize);
    ctx.lineTo(node.x - halfSize + borderRadius, node.y + halfSize);
    ctx.quadraticCurveTo(node.x - halfSize, node.y + halfSize, node.x - halfSize, node.y + halfSize - borderRadius);
    ctx.lineTo(node.x - halfSize, node.y - halfSize + borderRadius);
    ctx.quadraticCurveTo(node.x - halfSize, node.y - halfSize, node.x - halfSize + borderRadius, node.y - halfSize);
    ctx.closePath();

    const gradient = ctx.createLinearGradient(node.x - halfSize, node.y - halfSize, node.x + halfSize, node.y + halfSize);
    
    // Different gradients for center nodes
    if (isCenterNode) {
      gradient.addColorStop(0, '#FBBF24'); // Golden yellow for center
      gradient.addColorStop(1, '#F59E0B'); // Amber for center
    } else {
      gradient.addColorStop(0, '#A78BFA'); // Purple
      gradient.addColorStop(1, '#EC4899'); // Pink
    }
    
    ctx.fillStyle = gradient;
    
    const img = loadedImages[node.id];
    if (img?.complete && img.naturalHeight !== 0) {
      try {
        ctx.clip(); 
        ctx.drawImage(img, node.x - halfSize, node.y - halfSize, currentSize, currentSize);
      } catch (e) {
        console.error("Error drawing image for node:", node.id, e);
        ctx.fill(); 
      }
    } else {
      ctx.fill(); 
    }
    
    if (isHovered) {
        ctx.shadowColor = '#F472B6'; 
        ctx.shadowBlur = 15;
    } else if (isCenterNode) {
        ctx.shadowColor = '#F59E0B'; // Golden glow for center nodes
        ctx.shadowBlur = 8;
    }
    
    // Different stroke styles for center nodes and hovered states
    if (isHovered) {
      ctx.strokeStyle = '#F472B6';
      ctx.lineWidth = 3 / globalScale;
    } else if (isCenterNode) {
      ctx.strokeStyle = '#F59E0B'; // Golden stroke for center nodes
      ctx.lineWidth = 2.5 / globalScale;
    } else {
      ctx.strokeStyle = 'rgba(120,120,150,0.5)';
      ctx.lineWidth = 1.5 / globalScale;
    }
    
    ctx.stroke();
    
    // Reset shadow for subsequent drawings
    ctx.shadowColor = 'transparent'; 
    ctx.shadowBlur = 0;

    ctx.restore();
  }, [hoveredNode, loadedImages]); // Updated dependencies

  const linkCanvasObject = useCallback((link, ctx, globalScale) => {
    const { source, target } = link;
    if (typeof source.x !== 'number' || typeof source.y !== 'number' || !isFinite(source.x) || !isFinite(source.y) ||
        typeof target.x !== 'number' || typeof target.y !== 'number' || !isFinite(target.x) || !isFinite(target.y)) {
      return;
    }

    // Calculate link strength based on similarity (value) or distance
    const similarity = link.value || (1 - (link.distance || 0.5));
    const lineWidth = Math.max(0.5, similarity * 4) / globalScale; // Scale line width with zoom and similarity
    
    // Color intensity based on similarity - stronger links are more vibrant
    const alpha = Math.max(0.3, similarity);
    
    const gradient = ctx.createLinearGradient(source.x, source.y, target.x, target.y);
    gradient.addColorStop(0, `rgba(99, 102, 241, ${alpha})`); // Indigo with alpha
    gradient.addColorStop(1, `rgba(244, 114, 182, ${alpha})`); // Pink with alpha

    ctx.beginPath();
    ctx.moveTo(source.x, source.y);
    ctx.lineTo(target.x, target.y);
    ctx.strokeStyle = gradient;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
  }, []);

  const handleNodeClick = useCallback((node) => {
    // console.log("Clicked node:", node);
    if (node && typeof node.x === 'number' && typeof node.y === 'number' && isFinite(node.x) && isFinite(node.y)) {
        fgRef.current?.centerAt(node.x, node.y, 800); // Smoother transition
        fgRef.current?.zoom(2.5, 800);
    }
  }, []); // fgRef is stable

  // If no tracks, or graphData nodes are not yet populated (e.g. during initial fetch)
  if (!graphData.nodes || graphData.nodes.length === 0) {
    return (
        <div ref={containerRef} style={{ width: '100%', height: '350px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <p className="no-tracks-for-graph">Loading graph or no track data...</p>
        </div>
    );
  }
  
  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%', minHeight: '350px' }}> {/* Ensure parent div has dimensions */}
        <ForceGraph2D
        ref={fgRef}
        graphData={graphData}
        width={containerSize.width} // Use dynamic width
        height={containerSize.height} // Use dynamic height
        backgroundColor="transparent" 

        nodeLabel={node => `${node.title} - ${node.artist}`}
        nodeRelSize={baseNodeSize / 15} // Use baseNodeSize reference
        nodeCanvasObjectMode={() => 'after'}
        nodeCanvasObject={nodeCanvasObject}
        
        linkCanvasObjectMode={() => 'after'}
        linkCanvasObject={linkCanvasObject}
        
        // Improved force simulation settings
        linkStrength={link => {
          const similarity = link.value || (1 - (link.distance || 0.5));
          return Math.max(0.1, similarity * 0.8); // Stronger links for higher similarity
        }}
        linkDistance={link => {
          const distance = link.distance || (1 - (link.value || 0.5));
          return Math.max(80, Math.min(300, 120 + distance * 150)); // Dynamic distance based on similarity
        }}
        
        // Center the most similar node and spread others based on distance
        d3AlphaDecay={0.02} // Slower decay for more stable layout
        d3VelocityDecay={0.3} // Higher friction for less bouncy movement
        
        cooldownTicks={200} // More ticks for better settling
        cooldownTime={15000} // More time for complex layouts
        onEngineStop={() => fgRef.current?.zoomToFit(400, 100)}
        
        onNodeHover={handleNodeHover}
        onNodeClick={handleNodeClick}

        enableZoomPanInteraction={true}
        enablePointerInteraction={true}
        />
    </div>
  );
};

export default SongGraph;