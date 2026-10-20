import React, { useState, useEffect } from 'react';
import { predictComponent3Csv } from '../../api/component3Api';
import UploadedHddVisualAnalysis from './UploadedHddVisualAnalysis';

const SMART_LABELS = {
  smart_1_raw: 'Read Error Rate',
  smart_4_raw: 'Start/Stop Activity',
  smart_5_raw: 'Reallocated Sectors',
  smart_7_raw: 'Seek Error Rate',
  smart_9_raw: 'Power-On Hours',
  smart_12_raw: 'Power Cycle Count',
  smart_187_raw: 'Reported Uncorrectable Errors',
  smart_188_raw: 'Command Timeout',
  smart_192_raw: 'Power-Off Retract Activity',
  smart_193_raw: 'Load/Unload Cycle Count',
  smart_194_raw: 'Temperature',
  smart_197_raw: 'Current Pending Sectors',
  smart_199_raw: 'UDMA CRC Errors',
  smart_240_raw: 'Head Flying Hours',
  smart_241_raw: 'Total LBAs Written',
  smart_242_raw: 'Total LBAs Read',
};

const parseFeature = (feature) => {
  let base = feature;
  let stat = '';
  let window = '';
  const statMatch = feature.match(/_(diff_1d|mean|std|max)_(\d+d)$/);
  if (statMatch) {
    stat = statMatch[1] === 'diff_1d' ? '1d change' : statMatch[1];
    window = statMatch[2];
    base = feature.slice(0, feature.length - statMatch[0].length);
  }
  const label = SMART_LABELS[base] || base;
  const parts = [label];
  if (window) parts.push(`${stat} over ${window}`);
  return parts.join(' · ');
};

const HORIZONS = ['7d', '14d', '30d'];

const RiskCard = ({ horizon, risk }) => {
  const flagged = risk.flagged;
  const pct = risk.risk_percent;
  const thresholdPct = (risk.decision_threshold * 100).toFixed(1);
  return (
    <div className="evidence-metric-item" style={{ textAlign: 'center' }}>
      <span className="stat-label">{horizon} Failure Risk</span>
      <span
        className="stat-val"
        style={{ fontSize: '1.4rem', color: flagged ? '#f87171' : '#38bdf8' }}
      >
        {pct < 0.01 ? '<0.01%' : `${pct.toFixed(2)}%`}
      </span>
      <span style={{ fontSize: '0.7rem', color: '#94a3b8' }}>
        {flagged ? (
          <span className="badge badge-purple" style={{ fontSize: '0.65rem' }}>Above threshold</span>
        ) : (
          <span style={{ color: '#64748b' }}>below threshold {thresholdPct}%</span>
        )}
      </span>
    </div>
  );
};

const EvidenceAlignment = ({ entry }) => {
  const fleet = entry.fleet_evidence || {};
  const facts = [];
  if (fleet.matched_risk_difference_pp != null) {
    facts.push(`${fleet.matched_risk_difference_pp >= 0 ? '+' : ''}${fleet.matched_risk_difference_pp.toFixed(1)} pp matched ATT`);
  }
  if (fleet.matched_evidence_class) {
    facts.push(fleet.matched_evidence_class.replace(/_/g, ' ').toLowerCase());
  }
  const pathway = (fleet.temporal_pathways || [])[0];
  if (pathway && pathway.counterpart_factor_name) {
    const arrow = pathway.direction === 'outgoing' ? '\u2192' : '\u2190';
    facts.push(`PCMCI ${arrow} ${pathway.counterpart_factor_name}`);
  }
  if (fleet.integrated_classification) {
    facts.push(fleet.integrated_classification.replace(/_/g, ' ').toLowerCase());
  }
  if (facts.length === 0) {
    facts.push('model-only, no strong fleet-level alignment');
  }
  return (
    <span style={{ fontSize: '0.68rem', color: '#94a3b8' }}>
      {facts.join(' · ')}
    </span>
  );
};

export const UploadedHddAnalysis = ({ pred }) => {
  const ranking = pred.local_factor_ranking || [];
  const narrative = pred.local_root_cause_explanation;
  const primaryRisk = pred.risks['30d'] || {};
  const predictedClass = primaryRisk.flagged ? 'HIGH RISK (flagged)' : 'LOW RISK (not flagged)';
  const sufficiency = pred.sufficient_history || {};
  const historyOk = sufficiency['30d'];

  return (
    <div
      style={{
        border: '1px solid #1e40af',
        backgroundColor: 'rgba(30, 58, 138, 0.18)',
        borderRadius: '8px',
        padding: '14px',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
        <span style={{ fontSize: '0.9rem', fontWeight: 700, color: '#93c5fd' }}>
          Uploaded HDD Analysis
        </span>
        <span className="badge badge-blue" style={{ fontSize: '0.65rem' }}>
          Dynamic · This Drive Only
        </span>
      </div>

      <div className="evidence-metrics" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))' }}>
        <div className="evidence-metric-item">
          <span className="stat-label">Serial Number</span>
          <span className="stat-val" style={{ fontSize: '0.95rem' }}>{pred.serial_number}</span>
        </div>
        <div className="evidence-metric-item">
          <span className="stat-label">Observation Date</span>
          <span className="stat-val" style={{ fontSize: '0.95rem' }}>{pred.as_of_date}</span>
        </div>
        <div className="evidence-metric-item">
          <span className="stat-label">History Status</span>
          <span className="stat-val" style={{ fontSize: '0.95rem' }}>
            {pred.history_rows} rows
          </span>
          <span style={{ fontSize: '0.68rem', color: historyOk ? '#4ade80' : '#fbbf24' }}>
            {historyOk ? 'sufficient (30d)' : 'partial window'}
          </span>
        </div>
        <div className="evidence-metric-item">
          <span className="stat-label">30d Risk Score</span>
          <span className="stat-val" style={{ fontSize: '0.95rem', color: primaryRisk.flagged ? '#f87171' : '#38bdf8' }}>
            {primaryRisk.risk_percent != null ? `${primaryRisk.risk_percent.toFixed(2)}%` : 'n/a'}
          </span>
        </div>
        <div className="evidence-metric-item">
          <span className="stat-label">Threshold</span>
          <span className="stat-val" style={{ fontSize: '0.95rem' }}>
            {primaryRisk.decision_threshold != null ? primaryRisk.decision_threshold.toFixed(4) : 'n/a'}
          </span>
        </div>
        <div className="evidence-metric-item">
          <span className="stat-label">Predicted Class</span>
          <span className="stat-val" style={{ fontSize: '0.8rem', color: primaryRisk.flagged ? '#f87171' : '#4ade80' }}>
            {predictedClass}
          </span>
        </div>
      </div>

      {narrative && (
        <div style={{ fontSize: '0.82rem', color: '#cbd5e1', lineHeight: 1.55 }}>
          <div style={{ fontWeight: 600, color: '#e2e8f0', marginBottom: '4px' }}>
            Local Root-Cause Explanation
          </div>
          <p style={{ margin: '0 0 6px 0' }}>{narrative.headline}</p>
          <p style={{ margin: '0 0 6px 0' }}>{narrative.dominant_driver}</p>
          <p style={{ margin: '0 0 6px 0', color: '#93c5fd' }}>{narrative.fleet_alignment}</p>
          <p style={{ margin: 0, fontSize: '0.74rem', color: '#64748b' }}>{narrative.scope}</p>
        </div>
      )}

      <div>
        <div style={{ fontWeight: 600, color: '#e2e8f0', fontSize: '0.82rem', marginBottom: '8px' }}>
          Local Root-Cause Factor Ranking (30d)
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          {ranking.slice(0, 8).map((entry) => {
            const increases = entry.local_direction === 'increases_risk';
            return (
              <div
                key={entry.factor_id}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '3px',
                  padding: '6px 8px',
                  backgroundColor: 'rgba(15, 23, 42, 0.5)',
                  borderRadius: '6px',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: '10px', alignItems: 'baseline' }}>
                  <span style={{ fontSize: '0.8rem', color: '#e2e8f0' }}>
                    <span style={{ color: '#64748b', marginRight: '6px' }}>#{entry.local_rank}</span>
                    {entry.factor_name}
                  </span>
                  <span style={{ fontSize: '0.78rem', fontWeight: 600, color: increases ? '#f87171' : '#4ade80', whiteSpace: 'nowrap' }}>
                    {increases ? '▲' : '▼'} {entry.local_shap_share_pct.toFixed(1)}%
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: '10px' }}>
                  <span style={{ fontSize: '0.68rem', color: '#94a3b8' }}>
                    {entry.dominant_smart_label} · {entry.dominant_metric}
                  </span>
                  <EvidenceAlignment entry={entry} />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

const PredictPanel = ({ onPredictionChange }) => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (onPredictionChange) {
      onPredictionChange(Boolean(result && result.predictions?.length));
    }
  }, [result, onPredictionChange]);

  const onFileChange = (e) => {
    setFile(e.target.files?.[0] || null);
    setResult(null);
    setError(null);
  };

  const onPredict = async () => {
    if (!file) {
      setError('Choose a CSV file first.');
      return;
    }
    setUploading(true);
    setError(null);
    setResult(null);
    try {
      const data = await predictComponent3Csv(file);
      setResult(data);
    } catch (err) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === 'string' ? detail : err.message || 'Prediction failed.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div>
      <div className="section-header">
        <h2 className="section-title">Interactive Failure Prediction</h2>
        <span className="badge badge-blue">CSV Upload · Live Model · Dynamic</span>
      </div>

      <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
        <p style={{ fontSize: '0.8rem', color: '#94a3b8', margin: 0 }}>
          Upload a drive history CSV to run a fully dynamic per-drive analysis: risk scoring, local SHAP,
          local root-cause factor ranking, local explanation narrative, and the "Uploaded HDD Visual
          Analysis" charts (local factor bars, ATT overlay, PCMCI overlay, dynamic pathway, SMART history).
          Fleet-level research evidence (PCMCI, matched ATT, evidence catalog, figures) is static and is
          reused read-only — it is never recomputed by an upload.
        </p>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px', flexWrap: 'wrap' }}>
          <input type="file" accept=".csv" onChange={onFileChange} style={{ color: '#cbd5e1', fontSize: '0.85rem' }} />
          <button
            className="btn-retry"
            onClick={onPredict}
            disabled={uploading}
            style={{ opacity: uploading ? 0.6 : 1 }}
          >
            {uploading ? 'Scoring…' : 'Predict Failure Risk'}
          </button>
          {file && (
            <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
              {file.name} ({(file.size / 1024).toFixed(1)} KB)
            </span>
          )}
        </div>

        <p style={{ fontSize: '0.78rem', color: '#94a3b8', margin: 0 }}>
          CSV columns: <code style={{ color: '#cbd5e1' }}>serial_number</code>,{' '}
          <code style={{ color: '#cbd5e1' }}>date</code>, plus any of the 16 raw SMART columns
          (<code style={{ color: '#cbd5e1' }}>smart_5_raw</code>,{' '}
          <code style={{ color: '#cbd5e1' }}>smart_187_raw</code>, …). One row per drive per day;
          30+ days of history recommended. Up to 25 drives per upload. Risk is scored for the most
          recent day of each drive.
        </p>

        {error && (
          <div style={{ backgroundColor: 'rgba(127, 29, 29, 0.35)', border: '1px solid #991b1b', borderRadius: '8px', padding: '10px 14px', color: '#fca5a5', fontSize: '0.85rem' }}>
            {error}
          </div>
        )}

        {result?.predictions?.map((pred) => (
          <div
            key={pred.serial_number}
            style={{ backgroundColor: 'rgba(15, 23, 42, 0.6)', border: '1px solid #334155', borderRadius: '8px', padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
              <div style={{ fontSize: '1rem', fontWeight: 600, color: '#38bdf8' }}>{pred.serial_number}</div>
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                <span className="badge badge-purple" style={{ fontSize: '0.7rem' }}>As of {pred.as_of_date}</span>
                <span className="badge badge-blue" style={{ fontSize: '0.7rem' }}>{pred.history_rows} days history</span>
              </div>
            </div>

            <div className="evidence-metrics" style={{ gridTemplateColumns: '1fr 1fr 1fr' }}>
              {HORIZONS.map((h) => (
                <RiskCard key={h} horizon={h} risk={pred.risks[h]} />
              ))}
            </div>

            <UploadedHddAnalysis pred={pred} />

            <UploadedHddVisualAnalysis pred={pred} />

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '12px' }}>
              {pred.top_factors_by_horizon.map((entry) => (
                <div key={entry.horizon} style={{ fontSize: '0.78rem' }}>
                  <div style={{ color: '#e2e8f0', fontWeight: 600, marginBottom: '6px' }}>
                    Top drivers · {entry.horizon}
                  </div>
                  {entry.top_factors.slice(0, 3).map((factor) => (
                    <div key={factor.feature} style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', padding: '2px 0', color: '#cbd5e1' }}>
                      <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {parseFeature(factor.feature)}
                      </span>
                      <span style={{ color: factor.shap_value >= 0 ? '#f87171' : '#4ade80', fontWeight: 600, whiteSpace: 'nowrap' }}>
                        {factor.shap_value >= 0 ? '▲' : '▼'} {Math.abs(factor.shap_value).toFixed(2)}
                      </span>
                    </div>
                  ))}
                </div>
              ))}
            </div>

            <div style={{ fontSize: '0.8rem', color: '#94a3b8', borderTop: '1px solid #334155', paddingTop: '10px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
              {pred.causal_context.map((note, idx) => (
                <p key={idx} style={{ margin: 0 }}>{note}</p>
              ))}
            </div>

            {pred.warnings.length > 0 && (
              <div style={{ fontSize: '0.78rem', color: '#fbbf24' }}>
                {pred.warnings.map((w, idx) => (
                  <p key={idx} style={{ margin: 0 }}>⚠ {w}</p>
                ))}
              </div>
            )}
          </div>
        ))}

        {result?.model_info && (
          <div style={{ fontSize: '0.75rem', color: '#64748b', borderTop: '1px solid #334155', paddingTop: '8px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <span>
              {result.model_info.model} · {result.model_info.features} features · thresholds from validation best-F1.{' '}
              {result.model_info.note}
            </span>
            {result.model_info.dynamic_scope && (
              <span>
                <strong style={{ color: '#94a3b8' }}>Dynamic:</strong> {result.model_info.dynamic_scope}{' '}
                <strong style={{ color: '#94a3b8' }}>Static:</strong> {result.model_info.static_scope}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default PredictPanel;
