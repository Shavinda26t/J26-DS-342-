import React from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceLine,
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

const AttOverlay = ({ overlay }) => {
  const factors = overlay.factors || [];
  const maxAbs = Math.max(1, ...factors.map((f) => Math.abs(f.risk_difference_pp || 0)));
  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
      <SectionHeading
        title={`2 · ${overlay.title}`}
        badge="Static ATT + Local Overlay"
        note={overlay.disclaimer}
      />
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {factors.map((f) => {
          const width = (Math.abs(f.risk_difference_pp || 0) / maxAbs) * 50;
          const positive = (f.risk_difference_pp || 0) >= 0;
          const dimmed = !f.relevant_to_local_hdd;
          return (
            <div key={f.factor_id} style={{ opacity: dimmed ? 0.35 : 1 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#cbd5e1' }}>
                <span>
                  {f.factor_name}
                  {!dimmed && (
                    <span className="badge badge-purple" style={{ fontSize: '0.58rem', marginLeft: '8px' }}>
                      Local match
                    </span>
                  )}
                </span>
                <span style={{ fontWeight: 600 }}>
                  {f.risk_difference_pp >= 0 ? '+' : ''}{f.risk_difference_pp.toFixed(2)} pp
                  {f.ci_pp && (
                    <span style={{ color: '#64748b', fontWeight: 400 }}>
                      {' '}[{f.ci_pp[0].toFixed(1)}, {f.ci_pp[1].toFixed(1)}]
                    </span>
                  )}
                </span>
              </div>
              <div style={{ position: 'relative', height: '10px', backgroundColor: '#0f172a', borderRadius: '4px' }}>
                <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: '1px', backgroundColor: '#475569' }} />
                <div
                  style={{
                    position: 'absolute',
                    left: positive ? '50%' : `${50 - width}%`,
                    width: `${width}%`,
                    top: 0,
                    bottom: 0,
                    backgroundColor: positive ? (dimmed ? '#64748b' : '#ef4444') : '#34d399',
                    borderRadius: '4px',
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

const PcmciOverlay = ({ overlay }) => {
  const nodes = overlay.nodes || [];
  const edges = overlay.edges || [];
  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
      <SectionHeading
        title={`3 · ${overlay.title}`}
        badge="Static PCMCI + Local Overlay"
        note={overlay.disclaimer}
      />
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {edges.map((e, idx) => {
          const active = e.relevant_to_local_hdd;
          const touches = e.touches_local_hdd;
          return (
            <div
              key={idx}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                padding: '8px 10px',
                borderRadius: '6px',
                backgroundColor: active ? 'rgba(239, 68, 68, 0.14)' : 'rgba(15, 23, 42, 0.5)',
                border: active ? '1px solid rgba(239, 68, 68, 0.4)' : '1px solid #1e293b',
                opacity: touches ? 1 : 0.35,
                flexWrap: 'wrap',
              }}
            >
              <span style={{ fontSize: '0.82rem', color: active ? '#fca5a5' : '#cbd5e1', fontWeight: active ? 600 : 400 }}>
                {e.source_factor_name} <span style={{ color: '#38bdf8' }}>&#8594;</span> {e.target_factor_name}
              </span>
              <span style={{ fontSize: '0.68rem', color: '#94a3b8' }}>
                {e.strongest_detection_rate != null && `Detected ${Math.round(e.strongest_detection_rate * 5)}/5 runs · `}
                {e.strongest_lag_days != null && `lag ${e.strongest_lag_days}d · `}
                {active ? 'Both endpoints locally important' : touches ? 'One endpoint locally important' : 'Not local to this HDD'}
              </span>
            </div>
          );
        })}
      </div>
      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', fontSize: '0.72rem', color: '#94a3b8' }}>
        {nodes.map((n) => (
          <span
            key={n.factor_id}
            className="badge"
            style={{
              fontSize: '0.62rem',
              backgroundColor: n.locally_important ? 'rgba(239, 68, 68, 0.18)' : 'rgba(148, 163, 184, 0.12)',
              color: n.locally_important ? '#fca5a5' : '#94a3b8',
              border: `1px solid ${n.locally_important ? 'rgba(239, 68, 68, 0.4)' : 'rgba(148, 163, 184, 0.25)'}`,
            }}
          >
            {n.factor_name}
            {n.locally_important && ` · ${n.local_shap_share_pct.toFixed(1)}%`}
          </span>
        ))}
      </div>
    </div>
  );
};

const DynamicPathway = ({ pathway }) => (
  <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
    <SectionHeading
      title="4 · Dynamic Root-Cause Pathway (This HDD)"
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

const EvidenceClassSummary = ({ summary }) => {
  const total = Math.max(1, summary.total_local_factors);
  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
      <SectionHeading
        title="5 · Local Integrated Evidence Class Summary"
        badge={`${summary.total_local_factors} local factors`}
        note="Evidence-class distribution of the factors locally active in this uploaded HDD (not the full fleet catalog)."
      />
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {summary.classes.map((c) => (
          <div key={c.classification} style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ width: '230px', fontSize: '0.75rem', color: '#cbd5e1' }}>
              {formatEnumLabel(c.classification)}
            </span>
            <div style={{ flex: 1, height: '14px', backgroundColor: '#0f172a', borderRadius: '4px', overflow: 'hidden' }}>
              <div
                style={{
                  width: `${(c.count / total) * 100}%`,
                  height: '100%',
                  backgroundColor: classificationColor(c.classification),
                }}
              />
            </div>
            <span style={{ fontSize: '0.78rem', color: '#e2e8f0', fontWeight: 600, width: '24px', textAlign: 'right' }}>
              {c.count}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

const SmartHistoryTraces = ({ traces }) => (
  <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
    <SectionHeading
      title="6 · 30-Day SMART History (This HDD)"
      badge="From Uploaded CSV"
      note="Raw SMART trajectories for the top local factors, built directly from the uploaded drive history."
    />
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '14px' }}>
      {traces.map((t) => {
        const trendColor = t.trend === 'increasing' ? '#ef4444' : t.trend === 'decreasing' ? '#34d399' : '#94a3b8';
        return (
          <div key={t.factor_id} style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.76rem', color: '#cbd5e1' }}>
              <span>{t.factor_name}</span>
              <span style={{ color: trendColor, fontWeight: 600 }}>
                {t.trend === 'increasing' ? '▲' : t.trend === 'decreasing' ? '▼' : '—'} {t.trend}
              </span>
            </div>
            <ResponsiveContainer width="100%" height={120}>
              <LineChart data={t.points} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="date" stroke="#64748b" fontSize={9} tick={false} />
                <YAxis stroke="#64748b" fontSize={9} width={44} />
                <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: '#94a3b8' }} />
                <Line type="monotone" dataKey="value" stroke={trendColor} strokeWidth={2} dot={false} connectNulls />
              </LineChart>
            </ResponsiveContainer>
            <span style={{ fontSize: '0.68rem', color: '#64748b' }}>{t.smart_label} ({t.smart_attribute})</span>
          </div>
        );
      })}
    </div>
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
          Uploaded HDD Visual Analysis
        </span>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          <span className="badge badge-blue" style={{ fontSize: '0.65rem' }}>
            {pred.serial_number}
          </span>
          <span className="badge badge-purple" style={{ fontSize: '0.65rem' }}>
            Dynamic · Replaces on next upload
          </span>
        </div>
      </div>

      <LocalFactorBars bars={visual.local_factor_bars} />
      <AttOverlay overlay={visual.att_overlay} />
      <PcmciOverlay overlay={visual.pcmci_overlay} />
      <DynamicPathway pathway={visual.dynamic_root_cause_pathway} />
      <EvidenceClassSummary summary={visual.local_evidence_class_summary} />
      <SmartHistoryTraces traces={visual.smart_history_traces} />
    </div>
  );
};

export default UploadedHddVisualAnalysis;
