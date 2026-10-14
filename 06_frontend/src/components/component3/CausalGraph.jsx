import React from 'react';
import { formatEnumLabel, formatPP } from '../../utils/formatters';

const CausalGraph = ({ pathways = [], matchedEffects = [] }) => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
      {/* Temporal Pathway Section */}
      <div>
        <div className="section-header">
          <h2 className="section-title">PCMCI Stable Temporal Pathways</h2>
          <span className="badge badge-blue">Temporal Dependency Analysis</span>
        </div>

        <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {pathways.length === 0 ? (
            <p style={{ color: '#94a3b8' }}>No stable temporal pathways detected in exported results.</p>
          ) : (
            pathways.map((path, idx) => (
              <div key={idx} style={{ 
                backgroundColor: 'rgba(15, 23, 42, 0.6)', 
                padding: '16px', 
                borderRadius: '8px', 
                border: '1px solid #334155',
                display: 'flex',
                flexDirection: 'column',
                gap: '10px'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '1.05rem', fontWeight: '600', color: '#38bdf8' }}>
                    <span>{path.source_factor_name}</span>
                    <span>➔</span>
                    <span>{path.target_factor_name}</span>
                  </div>
                  <span className="badge badge-purple" style={{ fontSize: '0.75rem' }}>
                    Strongest Lag: {path.strongest_lag_days ?? '—'} Day(s)
                  </span>
                </div>

                <div style={{ display: 'flex', gap: '24px', fontSize: '0.85rem', color: '#cbd5e1' }}>
                  <span><strong>Temporal Magnitude:</strong> {path.temporal_magnitude ? path.temporal_magnitude.toFixed(4) : 'N/A'}</span>
                  <span><strong>Temporal Direction:</strong> {formatEnumLabel(path.temporal_direction)}</span>
                  <span><strong>Pair Rank:</strong> #{path.temporal_pair_rank}</span>
                </div>

                <p style={{ fontSize: '0.85rem', color: '#94a3b8', fontStyle: 'italic', marginTop: '4px' }}>
                  "{path.interpretation || 'Stable PCMCI temporal dependency.'}"
                </p>
              </div>
            ))
          )}

          <div style={{ fontSize: '0.8rem', color: '#94a3b8', borderTop: '1px solid #334155', paddingTop: '10px' }}>
            📌 <strong>Interpretation Note:</strong> This pathway represents a stable fleet-level temporal conditional dependency and does not by itself prove a physical causal mechanism.
          </div>
        </div>
      </div>

      {/* Matched ATT Observational Failure-Risk Section */}
      <div>
        <div className="section-header">
          <h2 className="section-title">Matched Observational 30-Day Failure-Risk Evidence (ATT)</h2>
          <span className="badge badge-purple">Propensity Matched Contrasts</span>
        </div>

        <div className="root-cause-grid">
          {matchedEffects.map((item, idx) => (
            <div key={item.factor_id || idx} className="card" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <h4 style={{ color: '#ffffff', fontSize: '1.05rem' }}>{item.factor_name}</h4>
                <span className="badge badge-blue" style={{ fontSize: '0.7rem' }}>
                  {formatEnumLabel(item.analysis_role)}
                </span>
              </div>

              <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#38bdf8' }}>
                {formatPP(item.risk_difference_pp)}
              </div>

              <div className="evidence-metrics" style={{ gridTemplateColumns: '1fr 1fr' }}>
                <div className="evidence-metric-item">
                  <span className="stat-label">Matched Pairs</span>
                  <span className="stat-val">{item.matched_pairs}</span>
                </div>
                <div className="evidence-metric-item">
                  <span className="stat-label">Bootstrap 95% CI</span>
                  <span className="stat-val" style={{ fontSize: '0.8rem' }}>
                    {item.cluster_bootstrap_95ci_pp 
                      ? `[${item.cluster_bootstrap_95ci_pp[0].toFixed(2)}, ${item.cluster_bootstrap_95ci_pp[1].toFixed(2)}] pp`
                      : 'N/A'}
                  </span>
                </div>
                <div className="evidence-metric-item">
                  <span className="stat-label">Treated Failures</span>
                  <span className="stat-val">{item.treated_failures}</span>
                </div>
                <div className="evidence-metric-item">
                  <span className="stat-label">Control Failures</span>
                  <span className="stat-val">{item.control_failures}</span>
                </div>
              </div>

              <div style={{ fontSize: '0.8rem', color: '#cbd5e1', lineHeight: '1.35' }}>
                <p><strong>Evidence Class:</strong> {formatEnumLabel(item.evidence_class)}</p>
                <p style={{ color: '#94a3b8', marginTop: '4px' }}>{item.interpretation}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default CausalGraph;
