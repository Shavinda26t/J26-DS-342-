import React, { useState, useEffect } from 'react';
import { getComponent3Figures, getComponent3FigureUrl } from '../../api/component3Api';

const ResearchFigures = () => {
  const [figures, setFigures] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedFigure, setSelectedFigure] = useState(null);

  useEffect(() => {
    let isMounted = true;
    const fetchFigures = async () => {
      try {
        const data = await getComponent3Figures();
        if (isMounted) setFigures(data);
      } catch (err) {
        console.error('Failed to fetch figure metadata:', err);
      } finally {
        if (isMounted) setLoading(false);
      }
    };
    fetchFigures();
    return () => { isMounted = false; };
  }, []);

  if (loading) return null;

  return (
    <div>
      <div className="section-header">
        <div>
          <h2 className="section-title">Research Visualizations</h2>
          <p style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
            Validated figures generated from global SHAP, PCMCI stability analysis, and matched ATT risk estimations.
          </p>
        </div>
      </div>

      <div className="figures-grid">
        {figures.map((fig) => {
          const imgUrl = getComponent3FigureUrl(fig.key);
          return (
            <div key={fig.key} className="card figure-card">
              <h4 style={{ color: '#ffffff', fontSize: '1.05rem' }}>{fig.title}</h4>
              <div 
                className="figure-img-container"
                onClick={() => setSelectedFigure({ ...fig, url: imgUrl })}
              >
                <img 
                  src={imgUrl} 
                  alt={fig.title} 
                  className="figure-img"
                  onError={(e) => {
                    e.target.onerror = null;
                    e.target.parentNode.innerHTML = '<div style="padding: 40px; text-align: center; color: #94a3b8;">Figure image unavailable</div>';
                  }}
                />
              </div>
              <p style={{ fontSize: '0.85rem', color: '#cbd5e1', lineHeight: '1.4' }}>
                {fig.description}
              </p>
            </div>
          );
        })}
      </div>

      {/* Modal Preview */}
      {selectedFigure && (
        <div className="modal-overlay" onClick={() => setSelectedFigure(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="close-btn" onClick={() => setSelectedFigure(null)}>✕</button>
            <h3 style={{ color: '#ffffff' }}>{selectedFigure.title}</h3>
            <img src={selectedFigure.url} alt={selectedFigure.title} className="modal-img" />
            <p style={{ fontSize: '0.9rem', color: '#cbd5e1' }}>{selectedFigure.description}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default ResearchFigures;
