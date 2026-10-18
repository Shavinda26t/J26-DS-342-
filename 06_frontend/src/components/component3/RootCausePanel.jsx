import React from 'react';
import { formatEnumLabel, formatPP } from '../../utils/formatters';

const RootCausePanel = ({ candidates = [] }) => {
  if (!candidates || candidates.length === 0) return null;

  const getTagClass = (classification) => {
    switch (classification) {
      case 'HIGH_PRIORITY_ROOT_CAUSE_CANDIDATE':
        return 'tag-high';
      case 'SUPPORTING_DEGRADATION_FACTOR':
        return 'tag-supporting';
      case 'EXPLORATORY_ROOT_CAUSE_CANDIDATE':
        return 'tag-exploratory';
      default:
        return 'tag-default';
    }
  };

  const getBorderClass = (classification) => {
    switch (classification) {
      case 'HIGH_PRIORITY_ROOT_CAUSE_CANDIDATE':
        return 'card-high';
      case 'SUPPORTING_DEGRADATION_FACTOR':
        return 'card-supporting';
      case 'EXPLORATORY_ROOT_CAUSE_CANDIDATE':
        return 'card-exploratory';
      default:
        return '';
    }
  };

  return (
    <div>
      <div className="section-header">
        <h2 className="section-title">Root-Cause Candidate Hierarchy</h2>
        <span className="badge badge-purple">Integrated Observational Evidence</span>
      </div>

      <div className="root-cause-grid">
        {candidates.map((item, idx) => {
          const ciLower = item.matched_failure_risk_ci ? item.matched_failure_risk_ci[0] : null;
          const ciUpper = item.matched_failure_risk_ci ? item.matched_failure_risk_ci[1] : null;

          return (
            <div key={item.factor_id || idx} className={`card evidence-card ${getBorderClass(item.classification)}`}>
              <div className="evidence-card-header">
                <div>
                  <h3 className="factor-name">{item.factor_name}</h3>
                  <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>ID: {item.factor_id}</span>
                </div>
                <span className={`classification-tag ${getTagClass(item.classification)}`}>
                  {formatEnumLabel(item.classification)}
                </span>
              </div>

              <div className="evidence-metrics">
                <div className="evidence-metric-item">
                  <span className="stat-label">Global SHAP Rank</span>
                  <span className="stat-val">#{item.global_shap_rank}</span>
                </div>
                <div className="evidence-metric-item">
                  <span className="stat-label">Temporal Degree</span>
                  <span className="stat-val">{item.stable_temporal_degree} link(s)</span>
                </div>
                <div className="evidence-metric-item">
                  <span className="stat-label">Matched 30d Risk Difference</span>
                  <span className="stat-val" style={{ color: '#38bdf8' }}>
                    {formatPP(item.matched_failure_risk_difference_pp)}
                  </span>
                </div>
                <div className="evidence-metric-item">
                  <span className="stat-label">Bootstrap 95% CI</span>
                  <span className="stat-val" style={{ fontSize: '0.85rem' }}>
                    {ciLower !== null && ciUpper !== null 
                      ? `[${ciLower.toFixed(2)}, ${ciUpper.toFixed(2)}] pp` 
                      : 'N/A'}
                  </span>
                </div>
              </div>

              <div style={{ fontSize: '0.85rem', color: '#cbd5e1', lineHeight: '1.4' }}>
                {item.classification === 'HIGH_PRIORITY_ROOT_CAUSE_CANDIDATE' && (
                  <p>Primary driver combining high predictive attribution with strong matched failure-risk evidence.</p>
                )}
                {item.classification === 'SUPPORTING_DEGRADATION_FACTOR' && (
                  <p>Upstream degradation indicator providing precursor temporal signal to uncorrectable sector errors.</p>
                )}
                {item.classification === 'EXPLORATORY_ROOT_CAUSE_CANDIDATE' && (
                  <p>Exploratory candidate with positive risk contrast requiring additional sample accumulation.</p>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default RootCausePanel;
