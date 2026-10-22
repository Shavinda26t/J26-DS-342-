import React from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { formatEnumLabel } from '../../utils/formatters';

const CLASS_COLORS = {
  HIGH_PRIORITY_ROOT_CAUSE_CANDIDATE: '#ef4444',
  SUPPORTING_DEGRADATION_FACTOR: '#fbbf24',
  EXPLORATORY_ROOT_CAUSE_CANDIDATE: '#60a5fa',
  PREDICTIVE_TEMPORAL_CONTEXT: '#34d399',
  PREDICTIVE_SIGNAL_REQUIRES_SEMANTIC_REVIEW: '#a78bfa',
  PREDICTIVE_ONLY: '#94a3b8',
};

const classificationColor = (classification) =>
  CLASS_COLORS[classification] || '#94a3b8';

const tooltipStyle = {
  backgroundColor: '#0f172a',
  border: '1px solid #334155',
  borderRadius: '6px',
  color: '#e2e8f0',
  fontSize: '0.78rem',
};

const SectionHeading = ({ title, badge, note }) => (
  <div style={{ marginBottom: '4px' }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
      <h3 style={{ fontSize: '0.95rem', color: '#e2e8f0', margin: 0 }}>{title}</h3>
      {badge && (
        <span className="badge badge-blue" style={{ fontSize: '0.62rem' }}>
          {badge}
        </span>
      )}
    </div>
    {note && (
      <p style={{ fontSize: '0.74rem', color: '#94a3b8', margin: '4px 0 0 0' }}>{note}</p>
    )}
  </div>
);

const LocalFactorBars = ({ bars }) => (
  <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
    <SectionHeading
      title="1 · Local Factor Evidence Overview"
      badge="Local SHAP"
      note={
        'Local predictive contribution of each canonical factor for this uploaded HDD, '
        + 'with existing fleet PCMCI / matched-ATT / integrated-evidence markers. '
        + 'Local SHAP does not prove causality.'
      }
    />
    <ResponsiveContainer width="100%" height={Math.max(200, bars.length * 30)}>
      <BarChart data={bars} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
        <XAxis type="number" stroke="#64748b" fontSize={11} unit="%" />
        <YAxis
          type="category"
          dataKey="factor_name"
          width={190}
          stroke="#94a3b8"
          fontSize={11}
        />
        <Tooltip
          contentStyle={tooltipStyle}
          formatter={(value, name, props) => {
            const d = props.payload;
            return [
              `${value.toFixed(2)}% (${d.local_direction.replace('_', ' ')})`,
              'Local SHAP share',
            ];
          }}
        />
        <Bar dataKey="local_shap_share_pct" radius={[0, 4, 4, 0]}>
          {bars.map((entry) => (
            <Cell
              key={entry.factor_id}
              fill={entry.local_direction === 'increases_risk' ? '#ef4444' : '#34d399'}
              opacity={entry.locally_important ? 1 : 0.55}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
      {bars.map((entry) => (
        <div
          key={entry.factor_id}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '0.72rem',
            color: '#94a3b8',
            flexWrap: 'wrap',
          }}
        >
          <span style={{ color: '#cbd5e1', minWidth: '180px' }}>{entry.factor_name}</span>
          <span
            className="badge badge-purple"
            style={{ fontSize: '0.6rem', backgroundColor: `${classificationColor(entry.integrated_evidence_class)}22`, color: classificationColor(entry.integrated_evidence_class), border: `1px solid ${classificationColor(entry.integrated_evidence_class)}55` }}
          >
            {formatEnumLabel(entry.integrated_evidence_class)}
          </span>
          {entry.pcmci_support && (
            <span className="badge badge-blue" style={{ fontSize: '0.6rem' }}>PCMCI support</span>
          )}
          {entry.matched_att_support && (
            <span className="badge badge-purple" style={{ fontSize: '0.6rem' }}>Matched ATT support</span>
          )}
        </div>
      ))}
    </div>
  </div>
);

const DynamicPathway = ({ pathway }) => (
  <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
    <SectionHeading
      title="2 · Dynamic Root-Cause Pathway (This HDD)"
      badge={pathway.active ? 'Active' : 'Weak / Inactive'}
      note={pathway.summary}
    />
    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
      {pathway.steps.map((step, idx) => (
        <React.Fragment key={step.factor_id}>
          <div
            style={{
              flex: '1 1 200px',
              minWidth: '200px',
              padding: '10px',
              borderRadius: '8px',
              backgroundColor:
                step.local_contribution_pct >= 5 && (step.temporal_support || step.matched_risk_support)
                  ? 'rgba(239, 68, 68, 0.14)'
                  : 'rgba(15, 23, 42, 0.6)',
              border: '1px solid #334155',
            }}
          >
            <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#e2e8f0' }}>{step.factor_name}</div>
            <div style={{ fontSize: '0.72rem', color: '#94a3b8', marginTop: '4px' }}>
              Local contribution: <strong style={{ color: '#38bdf8' }}>{step.local_contribution_pct.toFixed(1)}%</strong>
            </div>
            <div style={{ display: 'flex', gap: '6px', marginTop: '6px', flexWrap: 'wrap' }}>
              <span className="badge" style={{ fontSize: '0.58rem', backgroundColor: step.temporal_support ? 'rgba(56, 189, 248, 0.18)' : 'rgba(148, 163, 184, 0.12)', color: step.temporal_support ? '#7dd3fc' : '#94a3b8', border: '1px solid #334155' }}>
                Temporal {step.temporal_support ? '✓' : '—'}
              </span>
              <span className="badge" style={{ fontSize: '0.58rem', backgroundColor: step.matched_risk_support ? 'rgba(167, 139, 250, 0.18)' : 'rgba(148, 163, 184, 0.12)', color: step.matched_risk_support ? '#c4b5fd' : '#94a3b8', border: '1px solid #334155' }}>
                Matched risk {step.matched_risk_support ? '✓' : '—'}
              </span>
            </div>
            <div style={{ fontSize: '0.68rem', color: classificationColor(step.evidence_classification), marginTop: '6px' }}>
              {formatEnumLabel(step.evidence_classification)}
            </div>
          </div>
          {idx < pathway.steps.length - 1 && (
            <span style={{ color: '#38bdf8', fontSize: '1.2rem' }}>&#8594;</span>
          )}
        </React.Fragment>
      ))}
    </div>
    {pathway.active && (
      <div style={{ fontSize: '0.72rem', color: '#94a3b8', borderTop: '1px solid #334155', paddingTop: '8px' }}>
        Root-cause-oriented evidence: local contribution is this drive's SHAP share; temporal and matched risk
        support are fleet-level observational estimates (not individual causal effects for this HDD).
      </div>
    )}
  </div>
);

const UploadedHddVisualAnalysis = ({ pred }) => {
  const visual = pred.visual_analysis;

  if (!visual) {
    return null;
  }

  return (
    <div
      style={{
        border: '1px solid #1e40af',
        backgroundColor: 'rgba(15, 23, 42, 0.5)',
        borderRadius: '10px',
        padding: '16px',
        display: 'flex',
        flexDirection: 'column',
        gap: '18px',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
        <span style={{ fontSize: '1rem', fontWeight: 700, color: '#93c5fd' }}>
          Single HDD Prediction Dashboard · Explanation
        </span>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          <span className="badge badge-blue" style={{ fontSize: '0.65rem' }}>
            {pred.serial_number}
          </span>
          <span className="badge badge-purple" style={{ fontSize: '0.65rem' }}>
            Dynamic · This Drive Only
          </span>
        </div>
      </div>

      <LocalFactorBars bars={visual.local_factor_bars} />
      <DynamicPathway pathway={visual.dynamic_root_cause_pathway} />
    </div>
  );
};

export default UploadedHddVisualAnalysis;
