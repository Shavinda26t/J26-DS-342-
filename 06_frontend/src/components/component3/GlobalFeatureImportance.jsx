import React, { useState } from 'react';
import { formatEnumLabel, formatPP, formatPercentage } from '../../utils/formatters';

const GlobalFeatureImportance = ({ factors = [] }) => {
  const [filterText, setFilterText] = useState('');

  if (!factors || factors.length === 0) {
    return (
      <div className="card">
        <p style={{ color: '#94a3b8' }}>No factor evidence catalog data available.</p>
      </div>
    );
  }

  const filteredFactors = factors.filter((f) => {
    const q = filterText.toLowerCase();
    const fName = (f.factor_name || '').toLowerCase();
    const fId = (f.factor_id || '').toLowerCase();
    const fClass = (f.integrated_evidence?.classification || '').toLowerCase();
    return fName.includes(q) || fId.includes(q) || fClass.includes(q);
  });

  const getBadgeStyle = (classification) => {
    switch (classification) {
      case 'HIGH_PRIORITY_ROOT_CAUSE_CANDIDATE':
        return { bg: 'rgba(239, 68, 68, 0.2)', color: '#fca5a5', border: '1px solid rgba(239, 68, 68, 0.4)' };
      case 'SUPPORTING_DEGRADATION_FACTOR':
        return { bg: 'rgba(245, 158, 11, 0.2)', color: '#fcd34d', border: '1px solid rgba(245, 158, 11, 0.4)' };
      case 'EXPLORATORY_ROOT_CAUSE_CANDIDATE':
        return { bg: 'rgba(59, 130, 246, 0.2)', color: '#93c5fd', border: '1px solid rgba(59, 130, 246, 0.4)' };
      case 'PREDICTIVE_TEMPORAL_CONTEXT':
        return { bg: 'rgba(16, 185, 129, 0.2)', color: '#6ee7b7', border: '1px solid rgba(16, 185, 129, 0.4)' };
      default:
        return { bg: 'rgba(148, 163, 184, 0.15)', color: '#cbd5e1', border: '1px solid rgba(148, 163, 184, 0.3)' };
    }
  };

  return (
    <div>
      <div className="section-header">
        <div>
          <h2 className="section-title">Canonical Factor Integrated Evidence Catalog</h2>
          <p style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
            Comprehensive population-level breakdown across predictive, temporal, and matched observational evidence streams.
          </p>
        </div>
        <input
          type="text"
          placeholder="Filter factors..."
          value={filterText}
          onChange={(e) => setFilterText(e.target.value)}
          style={{
            padding: '8px 14px',
            borderRadius: '6px',
            border: '1px solid #334155',
            backgroundColor: '#0f172a',
            color: '#ffffff',
            fontSize: '0.85rem',
            width: '220px'
          }}
        />
      </div>

      <div className="table-container">
        <table className="c3-table">
          <thead>
            <tr>
              <th>Factor Name</th>
              <th>SHAP Rank</th>
              <th>Predictive Share</th>
              <th>PCMCI Degree</th>
              <th>Matched ATT Risk Diff</th>
              <th>Integrated Evidence Classification</th>
            </tr>
          </thead>
          <tbody>
            {filteredFactors.map((item, idx) => {
              const shapRank = item.predictive_evidence?.consensus_rank ?? item.global_shap_rank ?? 'N/A';
              const shapShare = item.predictive_evidence?.normalized_shap_share_pct;
              const pcmciDegree = item.temporal_evidence?.stable_cross_factor_degree ?? item.stable_temporal_degree ?? 0;
              const attDiff = item.matched_failure_risk_evidence?.risk_difference_pp ?? item.matched_failure_risk_difference_pp;
              const classification = item.integrated_evidence?.classification ?? item.classification;

              const badgeStyle = getBadgeStyle(classification);

              return (
                <tr key={item.factor_id || idx}>
                  <td>
                    <div style={{ fontWeight: '600', color: '#ffffff' }}>{item.factor_name}</div>
                    <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>{item.factor_id}</div>
                  </td>
                  <td>#{shapRank}</td>
                  <td>{shapShare !== undefined ? formatPercentage(shapShare) : 'N/A'}</td>
                  <td>{pcmciDegree} link(s)</td>
                  <td>
                    {attDiff !== undefined && attDiff !== null ? (
                      <span style={{ color: attDiff > 0 ? '#38bdf8' : '#e2e8f0', fontWeight: '600' }}>
                        {formatPP(attDiff)}
                      </span>
                    ) : (
                      <span style={{ color: '#94a3b8' }}>Not Evaluated</span>
                    )}
                  </td>
                  <td>
                    <span
                      style={{
                        padding: '4px 10px',
                        borderRadius: '6px',
                        fontSize: '0.75rem',
                        fontWeight: '600',
                        backgroundColor: badgeStyle.bg,
                        color: badgeStyle.color,
                        border: badgeStyle.border,
                        display: 'inline-block'
                      }}
                    >
                      {formatEnumLabel(classification)}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default GlobalFeatureImportance;
