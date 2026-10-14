import React, { useState, useEffect } from 'react';
import { getComponent3Case } from '../../api/component3Api';
import { formatEnumLabel, formatNullable, formatPercentage } from '../../utils/formatters';

const LocalExplanation = () => {
  const [activeTab, setActiveTab] = useState('TP');
  const [caseData, setCaseData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let isMounted = true;
    const fetchCase = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await getComponent3Case(activeTab);
        if (isMounted) setCaseData(data);
      } catch (err) {
        if (isMounted) setError(err.message || 'Failed to load case payload');
      } finally {
        if (isMounted) setLoading(false);
      }
    };

    fetchCase();
    return () => { isMounted = false; };
  }, [activeTab]);

  const hdd = caseData?.hdd || {};
  const pred = caseData?.prediction || {};
  const localExp = caseData?.local_explanation || {};
  const topFactors = localExp.top_factors || [];

  return (
    <div>
      <div className="section-header">
        <div>
          <h2 className="section-title">Representative Local Explanation Cases</h2>
          <p style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
            Instance-level SHAP predictive explanations cross-referenced with fleet-level evidence classes.
          </p>
        </div>
      </div>

      {/* Case Switcher Tabs */}
      <div className="case-tabs">
        {['TP', 'FP', 'FN', 'TN'].map((type) => (
          <button
            key={type}
            className={`tab-btn ${activeTab === type ? 'active' : ''}`}
            onClick={() => setActiveTab(type)}
          >
            {type} Case
          </button>
        ))}
      </div>

      {loading && (
        <div className="card state-container">
          <div className="spinner"></div>
          <p style={{ color: '#94a3b8' }}>Loading {activeTab} representative case data...</p>
        </div>
      )}

      {error && (
        <div className="card state-container" style={{ borderColor: '#ef4444' }}>
          <p style={{ color: '#fca5a5' }}>Error: {error}</p>
        </div>
      )}

      {!loading && !error && caseData && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Metadata & Prediction Box */}
          <div className="card" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
            <div>
              <span className="stat-label">Case Classification</span>
              <div style={{ fontSize: '1.2rem', fontWeight: '700', color: '#38bdf8' }}>
                {caseData.case_type} Representative Case
              </div>
            </div>
            <div>
              <span className="stat-label">Serial Number</span>
              <div style={{ fontSize: '0.95rem', color: '#f1f5f9' }}>
                {formatNullable(hdd.serial_number)}
              </div>
            </div>
            <div>
              <span className="stat-label">Observation Date</span>
              <div style={{ fontSize: '0.95rem', color: '#f1f5f9' }}>
                {formatNullable(hdd.observation_date)}
              </div>
            </div>
            <div>
              <span className="stat-label">30-Day Failure Probability</span>
              <div style={{ fontSize: '0.95rem', color: '#f1f5f9' }}>
                {formatNullable(pred.failure_probability)}
              </div>
            </div>
          </div>

          {/* Narrative Explanation */}
          {localExp.text && (
            <div className="card" style={{ backgroundColor: 'rgba(15, 23, 42, 0.7)', borderLeft: '4px solid #38bdf8' }}>
              <h4 style={{ color: '#ffffff', marginBottom: '8px' }}>Scientific Explanation Narrative</h4>
              <p style={{ fontSize: '0.9rem', color: '#cbd5e1', lineHeight: '1.5' }}>
                {localExp.text}
              </p>
            </div>
          )}

          {/* Top Local SHAP Factors Table */}
          <div className="card">
            <h4 style={{ color: '#ffffff', marginBottom: '14px' }}>Top Local Predictive Factors (SHAP)</h4>
            <div className="table-container">
              <table className="c3-table">
                <thead>
                  <tr>
                    <th>Factor Name</th>
                    <th>Local Direction</th>
                    <th>SHAP Share</th>
                    <th>Dominant SMART Metric</th>
                    <th>Fleet Root-Cause Alignment</th>
                  </tr>
                </thead>
                <tbody>
                  {topFactors.map((f, idx) => (
                    <tr key={f.factor_id || idx}>
                      <td>
                        <div style={{ fontWeight: '600', color: '#ffffff' }}>{f.factor_name}</div>
                        <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>{f.factor_id}</div>
                      </td>
                      <td>
                        <span style={{
                          padding: '3px 8px',
                          borderRadius: '4px',
                          fontSize: '0.75rem',
                          fontWeight: '600',
                          backgroundColor: f.local_direction === 'TOWARD_FAILURE_OUTPUT' 
                            ? 'rgba(239, 68, 68, 0.2)' 
                            : 'rgba(16, 185, 129, 0.2)',
                          color: f.local_direction === 'TOWARD_FAILURE_OUTPUT' ? '#fca5a5' : '#6ee7b7'
                        }}>
                          {formatEnumLabel(f.local_direction)}
                        </span>
                      </td>
                      <td>{formatPercentage(f.local_importance_share_pct)}</td>
                      <td style={{ fontSize: '0.8rem', color: '#cbd5e1' }}>
                        {formatNullable(f.dominant_local_feature)}
                      </td>
                      <td style={{ fontSize: '0.8rem' }}>
                        <span className="badge badge-purple" style={{ fontSize: '0.7rem' }}>
                          {formatEnumLabel(f.integrated_evidence_class || f.local_fleet_evidence_alignment)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default LocalExplanation;
