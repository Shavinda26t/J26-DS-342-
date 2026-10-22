import React, { useState, useEffect, useCallback } from 'react';
import { getComponent3Dashboard, getComponent3Factors } from '../api/component3Api';
import FailureExplanation from '../components/component3/FailureExplanation';
import RootCausePanel from '../components/component3/RootCausePanel';
import CausalGraph from '../components/component3/CausalGraph';
import GlobalFeatureImportance from '../components/component3/GlobalFeatureImportance';
import LocalExplanation from '../components/component3/LocalExplanation';
import ResearchFigures from '../components/component3/ResearchFigures';
import PredictPanel from '../components/component3/PredictPanel';

const CausalXAI = () => {
  const [data, setData] = useState(null);
  const [factors, setFactors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [hasPrediction, setHasPrediction] = useState(false);
  const [showStatic, setShowStatic] = useState(false);

  const handlePredictionChange = useCallback((active) => {
    setHasPrediction(active);
    setShowStatic(false);
  }, []);

  const fetchDashboardData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [dashRes, factorsRes] = await Promise.all([
        getComponent3Dashboard(),
        getComponent3Factors().catch(() => [])
      ]);
      setData(dashRes);
      setFactors(Array.isArray(factorsRes) ? factorsRes : []);
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
  const temporalPathways = data?.temporal_pathways || [];
  const matchedEffects = data?.matched_failure_risk_evidence || [];

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

      {/* Summary Cards & Scientific Warning Banner */}
      <FailureExplanation summary={summary} />

      {/* Interactive CSV Upload Prediction */}
      <PredictPanel onPredictionChange={handlePredictionChange} />

      {/* Fleet-level static research evidence (hidden while a prediction is shown) */}
      {hasPrediction && (
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
              figures, representative cases) are hidden while an uploaded-HDD prediction is displayed.
            </p>
          </div>
          <button className="btn-retry" onClick={() => setShowStatic((v) => !v)}>
            {showStatic ? 'Hide fleet-level evidence' : 'Show fleet-level evidence'}
          </button>
        </div>
      )}

      {(!hasPrediction || showStatic) && (
        <>
          {/* Root Cause Candidate Hierarchy */}
          <RootCausePanel candidates={rootCauses} />

          {/* Temporal Pathways & ATT Forest / Matched Effects */}
          <CausalGraph pathways={temporalPathways} matchedEffects={matchedEffects} />

          {/* Integrated Evidence Catalog Table */}
          <GlobalFeatureImportance factors={factors} />

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
