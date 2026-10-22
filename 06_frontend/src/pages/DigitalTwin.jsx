import React, { useEffect, useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts';
import {
  getComponent2ConfigComparison,
  getComponent2Detections,
  getComponent2LeadTimeDistribution,
  getComponent2Overview,
  getComponent2RulMetrics,
  getComponent2Sweep,
  getComponent2Twin,
  postComponent2BatchInference,
} from '../api/component2Api';
import LiveTwin from './LiveTwin';

const COLORS = {
  accent: '#38bdf8',
  good: '#34d399',
  warn: '#fbbf24',
  bad: '#f87171',
  muted: '#64748b',
  card: '#1e293b',
  border: '#334155',
  text: '#f8fafc',
  subtext: '#94a3b8',
};

const tooltipStyle = {
  background: '#0f172a',
  border: `1px solid ${COLORS.border}`,
  borderRadius: 8,
  color: COLORS.text,
  fontSize: 13,
};

const axisStyle = { tick: { fill: COLORS.subtext, fontSize: 12 } };

const cardStyle = {
  background: COLORS.card,
  border: `1px solid ${COLORS.border}`,
  borderRadius: 12,
  padding: 18,
};

const fmt = (value, digits = 1) =>
  value === null || value === undefined ? '—' : Number(value).toFixed(digits);

function KpiCard({ label, value, sub, color = COLORS.accent }) {
  return (
    <div style={{ ...cardStyle, borderTop: `3px solid ${color}` }}>
      <div style={{ color: COLORS.subtext, fontSize: 13, textTransform: 'uppercase', letterSpacing: 0.6 }}>
        {label}
      </div>
      <div style={{ color, fontSize: 34, fontWeight: 700, margin: '8px 0 4px' }}>{value}</div>
      <div style={{ color: COLORS.subtext, fontSize: 13 }}>{sub}</div>
    </div>
  );
}

function SectionCard({ title, subtitle, children, style }) {
  return (
    <div style={{ ...cardStyle, ...style }}>
      <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 2 }}>{title}</div>
      {subtitle && <div style={{ color: COLORS.subtext, fontSize: 13, marginBottom: 14 }}>{subtitle}</div>}
      {children}
    </div>
  );
}

function StatusPill({ detected }) {
  const style = {
    display: 'inline-block',
    padding: '2px 10px',
    borderRadius: 999,
    fontSize: 12,
    fontWeight: 600,
    color: detected ? '#052e1b' : '#334155',
    background: detected ? COLORS.good : '#cbd5e1',
  };
  return <span style={style}>{detected ? 'Detected' : 'Not detected'}</span>;
}

export default function DigitalTwin() {
  const [overview, setOverview] = useState(null);
  const [detections, setDetections] = useState([]);
  const [distribution, setDistribution] = useState([]);
  const [sweep, setSweep] = useState([]);
  const [comparison, setComparison] = useState([]);
  const [rul, setRul] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const [serialInput, setSerialInput] = useState('');
  const [twinDetail, setTwinDetail] = useState(null);
  const [twinError, setTwinError] = useState(null);
  const [twinLoading, setTwinLoading] = useState(false);

  const [statusFilter, setStatusFilter] = useState('all');
  const [activeTab, setActiveTab] = useState('validation');
  
  // Batch inference state
  const [batchFile, setBatchFile] = useState(null);
  const [batchPreview, setBatchPreview] = useState([]);
  const [batchErrors, setBatchErrors] = useState([]);
  const [batchResults, setBatchResults] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchErrorMsg, setBatchErrorMsg] = useState(null);

  const load = () => {
    setLoading(true);
    setError(null);
    Promise.all([
      getComponent2Overview(),
      getComponent2Detections(),
      getComponent2LeadTimeDistribution(),
      getComponent2Sweep(),
      getComponent2ConfigComparison(),
      getComponent2RulMetrics(),
    ])
      .then(([ov, de, di, sw, co, ru]) => {
        setOverview(ov.data);
        setDetections(de.data);
        setDistribution(di.data);
        setSweep(sw.data);
        setComparison(co.data);
        setRul(ru.data);
      })
      .catch((err) => {
        setError(
          `Could not reach the backend API (${err.message}). ` +
            'Is it running on http://localhost:8000 ?'
        );
      })
      .finally(() => setLoading(false));
  };

  useEffect(load, []);

  const filteredDetections = useMemo(() => {
    if (statusFilter === 'detected') return detections.filter((d) => d.detected_within_60_days);
    if (statusFilter === 'not_detected') return detections.filter((d) => !d.detected_within_60_days);
    return detections;
  }, [detections, statusFilter]);

  const inspectTwin = (event) => {
    event.preventDefault();
    const serial = serialInput.trim();
    if (!serial) return;
    setTwinLoading(true);
    setTwinError(null);
    setTwinDetail(null);
    getComponent2Twin(serial)
      .then((res) => setTwinDetail(res.data))
      .catch((err) =>
        setTwinError(
          err.response?.status === 404
            ? `No held-out results for serial "${serial}".`
            : `Lookup failed: ${err.message}`
        )
      )
      .finally(() => setTwinLoading(false));
  };

  const downloadTemplate = () => {
    const template = `serial_number,date,smart_1_normalized,smart_5_normalized,smart_187_normalized,smart_197_normalized,smart_198_normalized,smart_1_normalized_change_7d,smart_1_normalized_change_14d,smart_1_normalized_rolling_std_7,smart_187_normalized_change_7d,smart_187_normalized_change_14d,smart_187_normalized_rolling_std_7,smart_1_available,smart_5_available,smart_187_available,smart_197_available,smart_198_available
HDD001,2023-01-01,100,100,97,100,100,0,0,2.6367,0,0,0,1,1,1,1,1
HDD001,2023-01-02,100,100,97,100,100,0,0,2.6367,0,0,0,1,1,1,1,1
HDD001,2023-01-30,100,100,97,100,100,0,0,2.6367,0,0,0,1,1,1,1,1`;
    const blob = new Blob([template], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'component2_batch_template.csv';
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const handleBatchFile = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setBatchFile(file);
    setBatchResults(null);
    setBatchErrors([]);
    setBatchErrorMsg(null);
    
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      const lines = text.split('\n').filter(l => l.trim());
      if (lines.length <= 1) {
        setBatchPreview([]);
        setBatchErrors(['CSV appears to be empty']);
        return;
      }
      const headers = lines[0].split(',').map(h => h.trim());
      const rows = lines.slice(1, 6).map(line => {
        const vals = line.split(',').map(v => v.trim());
        const obj = {};
        headers.forEach((h, i) => { obj[h] = vals[i] || ''; });
        return obj;
      });
      setBatchPreview(rows);
      
      const required = ['serial_number', 'date'];
      const missing = required.filter(r => !headers.includes(r));
      if (missing.length > 0) {
        setBatchErrors(prev => [...prev, `Missing required columns: ${missing.join(', ')}`]);
      }
    };
    reader.readAsText(file);
  };

  const runBatchInference = async () => {
    if (!batchFile) return;
    setBatchLoading(true);
    setBatchErrorMsg(null);
    setBatchResults(null);
    try {
      const res = await postComponent2BatchInference(batchFile);
      setBatchResults(res.data);
    } catch (err) {
      setBatchErrorMsg(err.response?.data?.detail || err.message);
    } finally {
      setBatchLoading(false);
    }
  };

  const s = overview?.summary ?? {};
  const fa = overview?.false_alarms ?? {};
  const cfg = overview?.config ?? {};
  const deployedRow = comparison.find((c) => c.role === 'currently_deployed') ?? null;
  const deployedSweep =
    sweep.find((p) => p.matches_production) ??
    sweep.find((p) => p.config === cfg.config) ??
    null;

  const renderSimulator = () => (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <SectionCard
        title="Twin Simulator (Batch Inference)"
        subtitle="Upload CSV with historical observations (≥30 per HDD) to predict RUL using trained BiLSTM. Inference-only, not part of held-out validation."
      >
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 16 }}>
          <button
            type="button"
            onClick={downloadTemplate}
            style={{
              background: 'transparent',
              color: COLORS.accent,
              border: `1px solid ${COLORS.accent}`,
              borderRadius: 8,
              padding: '8px 16px',
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            Download Template CSV
          </button>
          <label
            style={{
              background: COLORS.accent,
              color: '#0f172a',
              border: 'none',
              borderRadius: 8,
              padding: '8px 16px',
              fontWeight: 700,
              cursor: 'pointer',
            }}
          >
            Choose CSV
            <input
              type="file"
              accept=".csv"
              onChange={handleBatchFile}
              style={{ display: 'none' }}
            />
          </label>
          {batchFile && (
            <button
              type="button"
              onClick={runBatchInference}
              disabled={batchLoading}
              style={{
                background: batchLoading ? '#94a3b8' : COLORS.good,
                color: '#0f172a',
                border: 'none',
                borderRadius: 8,
                padding: '8px 16px',
                fontWeight: 700,
                cursor: batchLoading ? 'not-allowed' : 'pointer',
              }}
            >
              {batchLoading ? 'Running...' : 'Run Inference'}
            </button>
          )}
        </div>
        
        {batchFile && (
          <div style={{ marginBottom: 12 }}>
            <div style={{ color: COLORS.text, fontSize: 14, marginBottom: 6 }}>
              Selected: {batchFile.name} ({(batchFile.size/1024).toFixed(1)} KB)
            </div>
          </div>
        )}
        
        {batchErrors.length > 0 && (
          <div style={{ background: 'rgba(248,113,113,0.1)', border: `1px solid ${COLORS.bad}`, borderRadius: 8, padding: 12, marginBottom: 12 }}>
            {batchErrors.map((err, i) => (
              <div key={i} style={{ color: COLORS.bad, fontSize: 13 }}>{err}</div>
            ))}
          </div>
        )}
        
        {batchPreview.length > 0 && (
          <div style={{ marginBottom: 16 }}>
            <div style={{ color: COLORS.subtext, fontSize: 13, marginBottom: 8 }}>Preview (first 5 rows)</div>
            <div style={{ overflowX: 'auto' }}>
              <table style={tableStyle}>
                <thead>
                  <tr>
                    {Object.keys(batchPreview[0]).slice(0, 8).map(k => (
                      <th key={k} style={thStyle}>{k}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {batchPreview.map((row, i) => (
                    <tr key={i}>
                      {Object.values(row).slice(0, 8).map((v, j) => (
                        <td key={j} style={tdStyle}>{v}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
        
        {batchErrorMsg && (
          <div style={{ color: COLORS.bad, fontSize: 14, marginBottom: 12 }}>{batchErrorMsg}</div>
        )}
        
        {batchResults && (
          <div>
            <div style={{ color: COLORS.subtext, fontSize: 13, marginBottom: 10 }}>
              Results: {batchResults.count} groups
            </div>
            <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
              <table style={tableStyle}>
                <thead>
                  <tr>
                    <th style={{ ...thStyle, position: 'sticky', top: 0, background: COLORS.card }}>Serial</th>
                    <th style={{ ...thStyle, position: 'sticky', top: 0, background: COLORS.card }}>Status</th>
                    <th style={{ ...thStyle, position: 'sticky', top: 0, background: COLORS.card }}>Predicted RUL (days)</th>
                    <th style={{ ...thStyle, position: 'sticky', top: 0, background: COLORS.card }}>Obs Used</th>
                    <th style={{ ...thStyle, position: 'sticky', top: 0, background: COLORS.card }}>Error</th>
                  </tr>
                </thead>
                <tbody>
                  {batchResults.predictions.map((p, i) => (
                    <tr key={i}>
                      <td style={{ ...tdStyle, fontFamily: 'monospace' }}>{p.serial_number}</td>
                      <td style={tdStyle}>{p.status}</td>
                      <td style={tdStyle}>{p.predicted_rul_days != null ? p.predicted_rul_days.toFixed(2) : '—'}</td>
                      <td style={tdStyle}>{p.observations_used ?? p.available ?? '—'}</td>
                      <td style={{ ...tdStyle, color: COLORS.bad, fontSize: 12 }}>{p.error ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </SectionCard>
    </div>
  );

  const renderValidation = () => (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Header */}
      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
        <div>
          <h1 style={{ fontSize: 26, marginBottom: 4 }}>Adaptive Digital Twin</h1>
          <div style={{ color: COLORS.subtext, fontSize: 14 }}>
            Held-out validation · {s.test_hdds} failed HDDs + {fa.healthy_hdds} healthy controls ·
            detector: {cfg.config || 'unknown'}
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <span style={pillStyle}>Severity ≥ {cfg.min_severity}</span>
          <span style={pillStyle}>Persistence = {cfg.persistence} observations</span>
          <span style={pillStyle}>Thresholds: {cfg.threshold_table}</span>
          {cfg.matches_production && (
            <span style={{ ...pillStyle, background: COLORS.good, color: '#052e1b', border: 'none' }}>
              Production config
            </span>
          )}
        </div>
      </div>

      {/* KPIs */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: 16 }}>
        <KpiCard
          label="Detection rate (60d)"
          value={`${fmt(s.detection_rate_percent, 2)}%`}
          sub={`${s.detected} / ${s.test_hdds} failed HDDs`}
          color={COLORS.good}
        />
        <KpiCard
          label="False-alarm rate"
          value={`${fmt(fa.healthy_false_alarm_rate_percent, 1)}%`}
          sub={`${fa.healthy_false_alarms} / ${fa.healthy_hdds} healthy controls`}
          color={COLORS.warn}
        />
        <KpiCard
          label="Median lead time"
          value={`${fmt(s.median_lead_time_days, 0)} d`}
          sub={`mean ${fmt(s.mean_lead_time_days, 1)} d · max ${fmt(s.maximum_lead_time_days, 0)} d`}
          color={COLORS.accent}
        />
        <KpiCard
          label="Median persistence span"
          value={`${fmt(s.median_persistence_span_days, 1)} d`}
          sub={`${s.persistence_observations} observations · max ${fmt(s.maximum_persistence_span_days, 1)} d`}
          color="#a78bfa"
        />
      </div>

      {/* Charts row 1 */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: 16 }}>
        <SectionCard
          title="Lead-time distribution"
          subtitle="Detection horizon before failure, among detected HDDs"
        >
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={distribution} margin={{ top: 5, right: 10, bottom: 5, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} />
              <XAxis dataKey="lead_time_band" {...axisStyle} />
              <YAxis {...axisStyle} allowDecimals={false} />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'rgba(148,163,184,0.08)' }} />
              <Bar dataKey="hdd_count" name="HDDs" radius={[4, 4, 0, 0]}>
                {distribution.map((entry, index) => (
                  <Cell key={entry.lead_time_band} fill={index % 2 ? '#0ea5e9' : COLORS.accent} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </SectionCard>

        <SectionCard
          title="Detection vs false-alarm trade-off"
          subtitle="Threshold sweep on the tuning set (train+validation); green = deployed"
        >
          <ResponsiveContainer width="100%" height={260}>
            <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} />
              <XAxis
                type="number"
                dataKey="healthy_false_alarm_rate_percent"
                name="False-alarm rate"
                unit="%"
                {...axisStyle}
              />
              <YAxis
                type="number"
                dataKey="detection_rate_percent"
                name="Detection rate"
                unit="%"
                {...axisStyle}
              />
              <ZAxis type="number" dataKey="median_lead_time_days" range={[40, 260]} name="Median lead (d)" />
              <Tooltip
                contentStyle={tooltipStyle}
                formatter={(value, name) => [`${fmt(value, 2)}${name.includes('rate') ? '%' : ''}`, name]}
              />
              <Legend wrapperStyle={{ fontSize: 12, color: COLORS.subtext }} />
              <Scatter
                name="Other configs"
                data={sweep.filter((p) => !p.matches_production)}
                fill={COLORS.muted}
                fillOpacity={0.85}
              />
              {deployedSweep && (
                <Scatter
                  name="Deployed config"
                  data={[deployedSweep]}
                  shape="star"
                  fill={COLORS.good}
                  stroke={COLORS.good}
                />
              )}
            </ScatterChart>
          </ResponsiveContainer>
        </SectionCard>
      </div>

      {/* Config comparison + RUL metrics */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: 16 }}>
        <SectionCard
          title="Tuning-selected vs deployed (held-out test)"
          subtitle="Both rows evaluated once on the untouched test split"
        >
          <div style={{ overflowX: 'auto' }}>
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Role</th>
                  <th style={thStyle}>Config</th>
                  <th style={thStyle}>Sev.</th>
                  <th style={thStyle}>Persist.</th>
                  <th style={thStyle}>Detection</th>
                  <th style={thStyle}>False alarms</th>
                  <th style={thStyle}>Median lead</th>
                </tr>
              </thead>
              <tbody>
                {comparison.map((row) => {
                  const isDeployed = row.role === 'currently_deployed';
                  return (
                    <tr key={row.role} style={{ background: isDeployed ? 'rgba(52,211,153,0.08)' : 'transparent' }}>
                      <td style={{ ...tdStyle, color: isDeployed ? COLORS.good : COLORS.warn, fontWeight: 600 }}>
                        {isDeployed ? 'Deployed' : 'Tuning-selected'}
                      </td>
                      <td style={tdStyle}>{row.config}</td>
                      <td style={tdStyle}>{row.min_severity}</td>
                      <td style={tdStyle}>{row.persistence}</td>
                      <td style={tdStyle}>{fmt(row.detection_rate_percent, 2)}%</td>
                      <td style={tdStyle}>{fmt(row.healthy_false_alarm_rate_percent, 1)}%</td>
                      <td style={tdStyle}>{fmt(row.median_lead_time_days, 0)} d</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </SectionCard>

        <SectionCard
          title="RUL accuracy by adaptive state"
          subtitle="Held-out predictions grouped by twin state (MAE / RMSE, days)"
        >
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={(rul?.by_state ?? []).map((r) => ({ ...r, state: r.adaptive_state }))} margin={{ top: 5, right: 10, bottom: 5, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} />
              <XAxis dataKey="state" {...axisStyle} interval={0} tickFormatter={(v) => v.split(' / ')[0]} />
              <YAxis {...axisStyle} />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'rgba(148,163,184,0.08)' }} />
              <Legend wrapperStyle={{ fontSize: 12, color: COLORS.subtext }} />
              <Bar dataKey="mae_days" name="MAE (d)" fill={COLORS.accent} radius={[4, 4, 0, 0]} />
              <Bar dataKey="rmse_days" name="RMSE (d)" fill="#a78bfa" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </SectionCard>
      </div>

      {/* RUL trajectory + twin lookup */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: 16 }}>
        <SectionCard title="RUL error by checkpoint" subtitle="MAE / RMSE (days) vs observations before failure">
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={rul?.trajectory ?? []} margin={{ top: 5, right: 10, bottom: 5, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} />
              <XAxis dataKey="checkpoint_observations" unit=" obs" {...axisStyle} />
              <YAxis {...axisStyle} />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'rgba(148,163,184,0.08)' }} />
              <Legend wrapperStyle={{ fontSize: 12, color: COLORS.subtext }} />
              <Bar dataKey="mae" name="MAE (d)" fill={COLORS.accent} radius={[4, 4, 0, 0]} />
              <Bar dataKey="rmse" name="RMSE (d)" fill="#a78bfa" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </SectionCard>

        <SectionCard title="Inspect an HDD" subtitle="Look up a serial number's twin state and lead time">
          <form onSubmit={inspectTwin} style={{ display: 'flex', gap: 8, marginBottom: 14 }}>
            <input
              value={serialInput}
              onChange={(e) => setSerialInput(e.target.value)}
              placeholder="e.g. PL1331LAHES1JH"
              style={{
                flex: 1,
                background: '#0f172a',
                border: `1px solid ${COLORS.border}`,
                borderRadius: 8,
                color: COLORS.text,
                padding: '9px 12px',
                fontSize: 14,
              }}
            />
            <button
              type="submit"
              disabled={twinLoading}
              style={{
                background: COLORS.accent,
                color: '#0f172a',
                border: 'none',
                borderRadius: 8,
                padding: '0 18px',
                fontWeight: 700,
                cursor: 'pointer',
              }}
            >
              {twinLoading ? '…' : 'Lookup'}
            </button>
          </form>
          {twinError && <div style={{ color: COLORS.bad, fontSize: 14 }}>{twinError}</div>}
          {twinDetail && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 10 }}>
              <Detail label="Adaptive state" value={twinDetail.adaptive_state} />
              <Detail label="Failure date" value={twinDetail.failure_date ?? '—'} />
              <Detail
                label="Actual RUL"
                value={twinDetail.actual_rul_days === null ? '—' : `${fmt(twinDetail.actual_rul_days, 0)} d`}
              />
              <Detail
                label="Predicted RUL"
                value={twinDetail.predicted_rul_days === null ? '—' : `${fmt(twinDetail.predicted_rul_days, 1)} d`}
              />
              <Detail
                label="Absolute error"
                value={twinDetail.absolute_error_days === null ? '—' : `${fmt(twinDetail.absolute_error_days, 1)} d`}
              />
              <Detail
                label="Degradation persistence"
                value={`${fmt(twinDetail.current_degradation_persistence, 0)} (now) / ${fmt(twinDetail.longest_degradation_persistence, 0)} (max)`}
              />
              <Detail label="Detected" value={twinDetail.detected ? 'Yes' : 'No'} good={twinDetail.detected} />
              <Detail
                label="Lead time"
                value={twinDetail.lead_time_days === null ? '—' : `${fmt(twinDetail.lead_time_days, 0)} d`}
              />
            </div>
          )}
        </SectionCard>
      </div>

      {/* Detections table */}
      <SectionCard
        title="Per-HDD detection results"
        subtitle={`${filteredDetections.length} of ${detections.length} failed HDDs shown`}
      >
        <div style={{ display: 'flex', gap: 8, marginBottom: 14 }}>
          {['all', 'detected', 'not_detected'].map((key) => (
            <button
              key={key}
              type="button"
              onClick={() => setStatusFilter(key)}
              style={{
                background: statusFilter === key ? COLORS.accent : 'transparent',
                color: statusFilter === key ? '#0f172a' : COLORS.subtext,
                border: `1px solid ${statusFilter === key ? COLORS.accent : COLORS.border}`,
                borderRadius: 8,
                padding: '6px 14px',
                fontSize: 13,
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              {key === 'all' ? 'All' : key === 'detected' ? 'Detected' : 'Not detected'}
            </button>
          ))}
        </div>
        <div style={{ overflowX: 'auto', maxHeight: 420, overflowY: 'auto' }}>
          <table style={tableStyle}>
            <thead>
              <tr>
                <th style={{ ...thStyle, position: 'sticky', top: 0, background: COLORS.card }}>Serial</th>
                <th style={{ ...thStyle, position: 'sticky', top: 0, background: COLORS.card }}>Failure date</th>
                <th style={{ ...thStyle, position: 'sticky', top: 0, background: COLORS.card }}>First detection</th>
                <th style={{ ...thStyle, position: 'sticky', top: 0, background: COLORS.card }}>Lead time</th>
                <th style={{ ...thStyle, position: 'sticky', top: 0, background: COLORS.card }}>Span (d)</th>
                <th style={{ ...thStyle, position: 'sticky', top: 0, background: COLORS.card }}>Longest run</th>
                <th style={{ ...thStyle, position: 'sticky', top: 0, background: COLORS.card }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {filteredDetections.map((row) => (
                <tr key={row.serial_number}>
                  <td style={{ ...tdStyle, fontFamily: 'monospace' }}>{row.serial_number}</td>
                  <td style={tdStyle}>{row.failure_date}</td>
                  <td style={tdStyle}>{row.first_detection_date ?? '—'}</td>
                  <td style={tdStyle}>{row.lead_time_days === null ? '—' : `${fmt(row.lead_time_days, 0)} d`}</td>
                  <td style={tdStyle}>{row.persistence_span_days === null ? '—' : fmt(row.persistence_span_days, 1)}</td>
                  <td style={tdStyle}>{row.longest_persistence}</td>
                  <td style={tdStyle}>
                    <StatusPill detected={row.detected_within_60_days} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, padding: 24 }}>
      <header style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <h1 style={{ color: COLORS.text, margin: 0, fontSize: 26 }}>Component 2 — Adaptive Digital Twin</h1>
        <p style={{ color: COLORS.subtext, margin: 0 }}>
          {activeTab === 'validation' 
            ? 'Held-out validation of the adaptive digital twin and RUL predictor (single source of truth).' 
            : 'Batch inference using trained BiLSTM on user-provided test data.'}
        </p>
      </header>
      
      <div style={{ display: 'flex', gap: 8, borderBottom: `1px solid ${COLORS.border}`, paddingBottom: 0 }}>
        <button
          type="button"
          onClick={() => setActiveTab('validation')}
          style={{
            background: activeTab === 'validation' ? 'rgba(56,189,248,0.12)' : 'transparent',
            color: activeTab === 'validation' ? COLORS.accent : COLORS.subtext,
            border: 'none',
            borderBottom: activeTab === 'validation' ? `2px solid ${COLORS.accent}` : 'none',
            padding: '10px 16px',
            fontWeight: 600,
            cursor: 'pointer',
            marginBottom: -1,
          }}
        >
          Validation
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('simulator')}
          style={{
            background: activeTab === 'simulator' ? 'rgba(52,211,153,0.12)' : 'transparent',
            color: activeTab === 'simulator' ? COLORS.good : COLORS.subtext,
            border: 'none',
            borderBottom: activeTab === 'simulator' ? `2px solid ${COLORS.good}` : 'none',
            padding: '10px 16px',
            fontWeight: 600,
            cursor: 'pointer',
            marginBottom: -1,
          }}
        >
          Twin Simulator
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('live')}
          style={{
            background: activeTab === 'live' ? 'rgba(167,139,250,0.12)' : 'transparent',
            color: activeTab === 'live' ? '#a78bfa' : COLORS.subtext,
            border: 'none',
            borderBottom: activeTab === 'live' ? '2px solid #a78bfa' : 'none',
            padding: '10px 16px',
            fontWeight: 600,
            cursor: 'pointer',
            marginBottom: -1,
          }}
        >
          Live Twin
        </button>
      </div>

      {activeTab === 'validation' && renderValidation()}
      {activeTab === 'simulator' && renderSimulator()}
      {activeTab === 'live' && <LiveTwin />}
    </div>
  );
}

const pillStyle = {
  display: 'inline-block',
  padding: '6px 12px',
  borderRadius: 999,
  background: 'rgba(56,189,248,0.12)',
  border: `1px solid rgba(56,189,248,0.4)`,
  color: COLORS.accent,
  fontSize: 13,
  fontWeight: 600,
};

const tableStyle = {
  width: '100%',
  borderCollapse: 'collapse',
  fontSize: 14,
};

const thStyle = {
  textAlign: 'left',
  padding: '10px 12px',
  color: COLORS.subtext,
  fontSize: 12,
  textTransform: 'uppercase',
  letterSpacing: 0.5,
  borderBottom: `1px solid ${COLORS.border}`,
};

const tdStyle = {
  padding: '9px 12px',
  color: COLORS.text,
  borderBottom: `1px solid rgba(51,65,85,0.6)`,
};

function Detail({ label, value, good }) {
  return (
    <div style={{ background: '#0f172a', borderRadius: 8, padding: '10px 12px' }}>
      <div style={{ color: COLORS.subtext, fontSize: 12, marginBottom: 4 }}>{label}</div>
      <div style={{ color: good ? COLORS.good : COLORS.text, fontSize: 15, fontWeight: 600 }}>{value}</div>
    </div>
  );
}
