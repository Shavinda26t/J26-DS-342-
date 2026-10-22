import React, { useState, useEffect, useCallback } from 'react';
import { getComponent3Dashboard } from '../api/component3Api';
import FailureExplanation from '../components/component3/FailureExplanation';
import RootCausePanel from '../components/component3/RootCausePanel';
import LocalExplanation from '../components/component3/LocalExplanation';
import ResearchFigures from '../components/component3/ResearchFigures';
import PredictPanel from '../components/component3/PredictPanel';

const CausalXAI = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showStatic, setShowStatic] = useState(false);

  const handlePredictionChange = useCallback((active) => {
    if (active) {
      setShowStatic(false);
    }
  }, []);

  const fetchDashboardData = async () => {
    setLoading(true);
    setError(null);
    try {
      const dashRes = await getComponent3Dashboard();
      setData(dashRes);
    } catch (err) {
      console.error('Failed to load Component 3 dashboard:', err);
      setError(err.message || 'Failed to connect to Component 3 backend API.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  if (loading) {
    return (
      <div className="state-container" style={{ minHeight: '60vh' }}>
        <div className="spinner"></div>
        <h3 style={{ color: '#ffffff' }}>Loading Component 3 Causal XAI Dashboard...</h3>
        <p style={{ color: '#94a3b8' }}>Fetching integrated predictive, temporal, and matched failure-risk evidence.</p>
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="state-container" style={{ minHeight: '60vh' }}>
        <div style={{ fontSize: '2.5rem' }}>⚠️</div>
        <h3 style={{ color: '#fca5a5' }}>Component 3 Service Unavailable</h3>
        <p style={{ color: '#94a3b8', maxWidth: '500px' }}>
          {error}
        </p>
        <button className="btn-retry" onClick={fetchDashboardData}>
          Retry Connection
        </button>
      </div>
    );
  }

  const summary = data?.summary || {};
  const rootCauses = data?.root_cause_candidates || [];

  return (
    <div className="c3-dashboard">
      {/* Header */}
      <div className="c3-header">
        <div className="c3-title-row">
          <h1>
            Causal Explainable AI
            <span className="badge badge-purple" style={{ fontSize: '0.85rem' }}>HDD Failure Analysis</span>
          </h1>
          <div style={{ display: 'flex', gap: '10px' }}>
            <span className="badge badge-blue">Prediction Horizon: 30 Days</span>
            <span className="badge badge-purple">Root-Cause-Oriented Observational Evidence</span>
          </div>
        </div>
        <p className="c3-subtitle">
          Integrated predictive, temporal and matched failure-risk evidence for interpretable HDD failure analysis.
        </p>
      </div>

      {/* Scientific Warning Banner (always visible) */}
      <FailureExplanation summary={summary} showMetrics={false} />

      {/* Single HDD Prediction Dashboard (primary view) */}
      <PredictPanel onPredictionChange={handlePredictionChange} />

      {/* Fleet-level static research evidence (collapsed by default) */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '12px',
          padding: '14px 18px',
          borderRadius: '10px',
          border: '1px solid #334155',
          backgroundColor: 'rgba(15, 23, 42, 0.5)',
        }}
      >
        <div>
          <div style={{ fontSize: '0.95rem', fontWeight: 600, color: '#e2e8f0' }}>
            Fleet-Level Research Evidence
          </div>
          <p style={{ fontSize: '0.78rem', color: '#94a3b8', margin: '4px 0 0 0' }}>
            Population-level causal findings (PCMCI, matched ATT, evidence catalog, research
            figures, representative cases). Static and read-only — never recomputed by an upload.
          </p>
        </div>
        <button className="btn-retry" onClick={() => setShowStatic((v) => !v)}>
          {showStatic ? 'Hide fleet-level evidence' : 'Show fleet-level evidence'}
        </button>
      </div>

      {showStatic && (
        <>
          {/* Fleet Metric Summary Cards */}
          <FailureExplanation summary={summary} showWarning={false} />

          {/* Root Cause Candidate Hierarchy */}
          <RootCausePanel candidates={rootCauses} />

          {/* Research Visualizations */}
          <ResearchFigures />

          {/* Local Explanation Section (TP / FP / FN / TN) */}
          <LocalExplanation />
        </>
      )}
    </div>
  );
};

export default CausalXAI;
