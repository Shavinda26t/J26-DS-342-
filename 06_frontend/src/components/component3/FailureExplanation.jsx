import React from 'react';

const FailureExplanation = ({ summary }) => {
  if (!summary) return null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Persistent Scientific Warning Box */}
      <div className="warning-banner">
        <span style={{ fontSize: '1.4rem' }}>⚠️</span>
        <div>
          <strong style={{ display: 'block', marginBottom: '4px', color: '#fbbf24' }}>
            Causal Interpretation & Scientific Scope Notice
          </strong>
          <p style={{ fontSize: '0.9rem', lineHeight: '1.4' }}>
            Component 3 integrates predictive attribution, stable temporal relationships, and matched observational failure-risk evidence. 
            These results support root-cause-oriented reasoning but do not prove that a factor physically caused failure in an individual HDD.
          </p>
        </div>
      </div>

      {/* Metric Summary Cards */}
      <div className="metrics-grid">
        <div className="card metric-card">
          <span className="metric-label">Canonical Factors Evaluated</span>
          <span className="metric-value">{summary.canonical_factor_count ?? '—'}</span>
          <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Fleet SMART & Workload Factors</span>
        </div>

        <div className="card metric-card">
          <span className="metric-label">Stable PCMCI Factors</span>
          <span className="metric-value metric-highlight">{summary.factors_with_stable_pcmci_evidence ?? '—'}</span>
          <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Temporal Conditional Dependencies</span>
        </div>

        <div className="card metric-card">
          <span className="metric-label">Matched Failure-Risk Factors</span>
          <span className="metric-value">{summary.factors_with_matched_failure_risk_evidence ?? '—'}</span>
          <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Observational ATT Risk Contrasts</span>
        </div>

        <div className="card metric-card" style={{ borderColor: 'rgba(239, 68, 68, 0.4)' }}>
          <span className="metric-label">Primary Root-Cause Candidate</span>
          <span className="metric-value" style={{ color: '#fca5a5', fontSize: '1.25rem' }}>
            {summary.primary_root_cause_candidate || 'Not available in API payload'}
          </span>
          <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>High-Priority Fleet Evidence</span>
        </div>
      </div>
    </div>
  );
};

export default FailureExplanation;
