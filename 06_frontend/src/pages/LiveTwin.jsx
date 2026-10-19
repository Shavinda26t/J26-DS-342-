import React, { useState } from 'react';
import {
  deleteComponent2Twin,
  getComponent2TwinCalibration,
  postComponent2TwinLearn,
  postComponent2TwinSync,
} from '../api/component2Api';

const COLORS = {
  accent: '#38bdf8',
  good: '#34d399',
  warn: '#fbbf24',
  bad: '#f87171',
  muted: '#64748b',
  purple: '#a78bfa',
  card: '#1e293b',
  border: '#334155',
  text: '#f8fafc',
  subtext: '#94a3b8',
};

const cardStyle = {
  background: COLORS.card,
  border: `1px solid ${COLORS.border}`,
  borderRadius: 12,
  padding: 18,
};

const inputStyle = {
  background: '#0f172a',
  border: `1px solid ${COLORS.border}`,
  borderRadius: 8,
  color: COLORS.text,
  padding: '8px 10px',
  fontSize: 14,
  width: '100%',
  boxSizing: 'border-box',
};

const thStyle = {
  textAlign: 'left',
  padding: '9px 12px',
  color: COLORS.subtext,
  fontSize: 12,
  textTransform: 'uppercase',
  letterSpacing: 0.5,
  borderBottom: `1px solid ${COLORS.border}`,
};

const tdStyle = {
  padding: '8px 12px',
  color: COLORS.text,
  borderBottom: '1px solid rgba(51,65,85,0.6)',
};

const fmt = (v, d = 1) =>
  v === null || v === undefined ? '—' : Number(v).toFixed(d);

const STATE_COLORS = [COLORS.good, COLORS.accent, COLORS.warn, COLORS.bad];

const SMART_FIELDS = [
  { key: 'smart_1_normalized', label: 'SMART 1' },
  { key: 'smart_5_normalized', label: 'SMART 5' },
  { key: 'smart_187_normalized', label: 'SMART 187' },
  { key: 'smart_197_normalized', label: 'SMART 197' },
  { key: 'smart_198_normalized', label: 'SMART 198' },
];

function SectionCard({ title, subtitle, children, accent = COLORS.accent }) {
  return (
    <div style={{ ...cardStyle, borderTop: `3px solid ${accent}` }}>
      <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 2 }}>{title}</div>
      {subtitle && (
        <div style={{ color: COLORS.subtext, fontSize: 13, marginBottom: 14 }}>{subtitle}</div>
      )}
      {children}
    </div>
  );
}

function Metric({ label, value, color = COLORS.text, sub }) {
  return (
    <div style={{ background: '#0f172a', borderRadius: 8, padding: '12px 14px' }}>
      <div style={{ color: COLORS.subtext, fontSize: 12, textTransform: 'uppercase', letterSpacing: 0.5 }}>
        {label}
      </div>
      <div style={{ color, fontSize: 26, fontWeight: 700, margin: '6px 0 2px' }}>{value}</div>
      {sub && <div style={{ color: COLORS.subtext, fontSize: 12 }}>{sub}</div>}
    </div>
  );
}

export default function LiveTwin() {
  const [serial, setSerial] = useState('LIVE-001');
  const [obs, setObs] = useState({
    smart_1_normalized: 100,
    smart_5_normalized: 100,
    smart_187_normalized: 97,
    smart_197_normalized: 100,
    smart_198_normalized: 100,
  });
  const [pushCount, setPushCount] = useState(0);

  const [liveState, setLiveState] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [actualRul, setActualRul] = useState('');
  const [calibration, setCalibration] = useState(null);

  const baseDate = new Date('2023-01-01T00:00:00Z');
  const dateFor = (i) => new Date(baseDate.getTime() + i * 86400000).toISOString().slice(0, 10);

  const setField = (key, value) => {
    const num = value === '' ? '' : Number(value);
    setObs((o) => ({ ...o, [key]: num }));
  };

  const pushOne = async (observation, index) => {
    const res = await postComponent2TwinSync(serial, {
      serial_number: serial,
      date: dateFor(index),
      ...observation,
    });
    return res.data;
  };

  const pushObservation = async () => {
    setLoading(true);
    setError(null);
    try {
      const state = await pushOne(obs, pushCount);
      setLiveState(state);
      setPushCount((c) => c + 1);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const simulateSeries = async () => {
    setLoading(true);
    setError(null);
    try {
      let state = null;
      const n = 32;
      for (let i = 0; i < n; i += 1) {
        const declining = {
          ...obs,
          smart_1_normalized: Number(obs.smart_1_normalized) - i * 0.8,
          smart_187_normalized: Number(obs.smart_187_normalized) - i * 0.5,
        };
        // eslint-disable-next-line no-await-in-loop
        state = await pushOne(declining, pushCount + i);
      }
      setLiveState(state);
      setPushCount((c) => c + n);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const feedOutcome = async () => {
    if (actualRul === '') return;
    setLoading(true);
    setError(null);
    try {
      const res = await postComponent2TwinLearn(serial, Number(actualRul));
      setLiveState(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadCalibration = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getComponent2TwinCalibration();
      setCalibration(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const resetTwin = async () => {
    setLoading(true);
    setError(null);
    try {
      await deleteComponent2Twin(serial);
      setLiveState(null);
      setPushCount(0);
      setActualRul('');
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const forecast = liveState?.forecast;
  const interval = liveState?.interval;
  const adaptation = liveState?.online_adaptation;
  const thresholds = liveState?.adaptive_thresholds;
  const stateLevel = liveState?.adaptive_state ?? 0;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Controls */}
      <SectionCard
        title="Live Digital Twin Synchronization"
        subtitle="Stream SMART observations into a live twin. Each push runs adaptive thresholds, RUL + online learning, conformal confidence, data-quality and a multi-horizon forecast. State persists on the backend across calls."
        accent={COLORS.good}
      >
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 12, marginBottom: 14 }}>
          <div>
            <label style={{ color: COLORS.subtext, fontSize: 12, display: 'block', marginBottom: 4 }}>
              Serial number
            </label>
            <input value={serial} onChange={(e) => setSerial(e.target.value)} style={inputStyle} />
          </div>
          {SMART_FIELDS.map((f) => (
            <div key={f.key}>
              <label style={{ color: COLORS.subtext, fontSize: 12, display: 'block', marginBottom: 4 }}>
                {f.label}
              </label>
              <input
                type="number"
                value={obs[f.key]}
                onChange={(e) => setField(f.key, e.target.value)}
                style={inputStyle}
              />
            </div>
          ))}
        </div>

        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          <button type="button" onClick={pushObservation} disabled={loading} style={btnStyle(COLORS.accent, loading)}>
            {loading ? 'Working…' : 'Push Observation'}
          </button>
          <button type="button" onClick={simulateSeries} disabled={loading} style={btnStyle(COLORS.good, loading)}>
            Simulate 32-obs series
          </button>
          <button type="button" onClick={resetTwin} disabled={loading} style={btnStyle(COLORS.bad, loading)}>
            Reset twin
          </button>
          <button type="button" onClick={loadCalibration} disabled={loading} style={btnStyle(COLORS.purple, loading)}>
            Load calibration
          </button>
        </div>

        {error && (
          <div style={{ background: 'rgba(248,113,113,0.1)', border: `1px solid ${COLORS.bad}`, borderRadius: 8, padding: 10, marginTop: 12, color: COLORS.bad, fontSize: 13 }}>
            {error}
          </div>
        )}
      </SectionCard>

      {!liveState && !calibration && (
        <SectionCard title="No live state yet" subtitle="Push an observation or simulate a series to see the twin come alive." accent={COLORS.muted}>
          <div style={{ color: COLORS.subtext, fontSize: 14 }}>
            A drive needs 30 observations before the RUL model runs; below that the twin still tracks
            degradation state and severity. Use <b>Simulate 32-obs series</b> to reach RUL immediately.
          </div>
        </SectionCard>
      )}

      {liveState && (
        <>
          {/* Headline metrics */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 14 }}>
            <Metric
              label="Adaptive state"
              value={fmt(stateLevel, 0)}
              color={STATE_COLORS[stateLevel] || COLORS.text}
              sub={liveState.adaptive_state_name}
            />
            <Metric
              label="Adaptive severity"
              value={fmt(liveState.adaptive_severity, 0)}
              color={liveState.adaptive_severity >= 2 ? COLORS.warn : COLORS.good}
              sub={`accel z ${fmt(liveState.acceleration_z, 2)} · streak ${fmt(liveState.streak, 0)}`}
            />
            <Metric
              label="Predicted RUL"
              value={`${fmt(liveState.predicted_rul_days, 1)} d`}
              color={COLORS.accent}
              sub={
                interval
                  ? `90% CI [${fmt(interval.low, 1)}, ${fmt(interval.high, 1)}]`
                  : liveState.rul_error || 'insufficient history'
              }
            />
            <Metric
              label="Confidence"
              value={fmt(liveState.confidence, 2)}
              color={(liveState.confidence ?? 0) >= 0.6 ? COLORS.good : COLORS.warn}
              sub={`data quality ${fmt(liveState.data_quality, 2)}`}
            />
            <Metric
              label="Failure ETA"
              value={forecast?.failure_eta_day != null ? `${forecast.failure_eta_day} d` : '—'}
              color={COLORS.purple}
              sub={`history ${liveState.history_length} obs`}
            />
          </div>

          {/* Adaptation + thresholds + interval detail */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
            <SectionCard title="Online adaptation (Feature 1)" subtitle="RLS residual corrector on the frozen BiLSTM" accent={COLORS.good}>
              {adaptation ? (
                <div style={{ display: 'grid', gap: 8 }}>
                  <Row k="Raw BiLSTM RUL" v={`${fmt(adaptation.raw, 2)} d`} />
                  <Row k="Adapted RUL" v={`${fmt(adaptation.adapted, 2)} d`} color={COLORS.good} />
                  <Row k="Correction" v={`${fmt(adaptation.correction_days, 2)} d`} />
                  <Row k="Source" v={adaptation.source} />
                  <Row k="Personal updates" v={fmt(adaptation.personal_updates, 0)} />
                  <Row k="Global updates (seeded)" v={fmt(adaptation.global_updates, 0)} />
                  <Row k="Personal slope b1" v={fmt(adaptation.personal?.slope, 3)} />
                </div>
              ) : (
                <div style={{ color: COLORS.subtext, fontSize: 14 }}>No RUL yet (need 30 observations).</div>
              )}
            </SectionCard>

            <SectionCard title="Adaptive thresholds (Feature 3)" subtitle="Per-drive noise-scaled bounds (never looser than global)" accent={COLORS.warn}>
              {thresholds ? (
                <div style={{ display: 'grid', gap: 8 }}>
                  <Row k="SMART1 strictness" v={`×${fmt(thresholds.smart_1_strictness, 2)}`} />
                  <Row k="SMART187 strictness" v={`×${fmt(thresholds.smart_187_strictness, 2)}`} />
                  <Row k="SMART1 change noise" v={fmt(thresholds.smart_1_change_noise, 3)} />
                  <Row k="Global persistence" v={fmt(liveState.detector?.global_persistence, 0)} />
                  <Row k="Adaptive persistence" v={fmt(liveState.detector?.adaptive_persistence, 0)} color={COLORS.warn} />
                  <Row k="Qualifying rate" v={fmt(thresholds.observed_qualifying_rate, 2)} />
                  <Row k="Calibrated" v={thresholds.is_calibrated ? 'Yes' : 'Warming up'} />
                </div>
              ) : (
                <div style={{ color: COLORS.subtext, fontSize: 14 }}>—</div>
              )}
            </SectionCard>

            <SectionCard title="Confidence & data quality (Feature 2)" subtitle="Quality-gated conformal interval" accent={COLORS.accent}>
              {interval ? (
                <div style={{ display: 'grid', gap: 8 }}>
                  <Row k="Interval low" v={`${fmt(interval.low, 1)} d`} />
                  <Row k="Interval high" v={`${fmt(interval.high, 1)} d`} />
                  <Row k="Base half-width" v={`${fmt(interval.base_half_width, 1)} d`} />
                  <Row k="Target coverage" v={`${fmt((interval.coverage_alpha ? 1 - interval.coverage_alpha : 0) * 100, 0)}%`} />
                  <Row k="Calibration samples" v={fmt(interval.calibration_samples, 0)} />
                  <Row k="Health features avail" v={`${fmt(liveState.health_features_available, 0)} / 5`} />
                  <Row k="Temporal valid frac" v={fmt(liveState.temporal_valid_fraction, 2)} />
                </div>
              ) : (
                <div style={{ color: COLORS.subtext, fontSize: 14 }}>No interval until RUL is available.</div>
              )}
            </SectionCard>
          </div>

          {/* Forecast trajectory */}
          {forecast && (
            <SectionCard
              title="Future health forecast (Feature 4)"
              subtitle={`Multi-horizon projection · current state ${forecast.current_state_name} · streak ${forecast.current_streak} · persistence ${forecast.persistence_used}`}
              accent={COLORS.purple}
            >
              {forecast.first_escalation ? (
                <div style={{ background: 'rgba(167,139,250,0.1)', border: `1px solid ${COLORS.purple}`, borderRadius: 8, padding: 10, marginBottom: 12, color: COLORS.purple, fontSize: 13 }}>
                  Escalates to <b>{forecast.first_escalation.to_state_name}</b> by day {forecast.first_escalation.day} if degradation persists.
                </div>
              ) : (
                <div style={{ background: 'rgba(52,211,153,0.1)', border: `1px solid ${COLORS.good}`, borderRadius: 8, padding: 10, marginBottom: 12, color: COLORS.good, fontSize: 13 }}>
                  No state escalation projected within the forecast horizon.
                </div>
              )}
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
                  <thead>
                    <tr>
                      <th style={thStyle}>Horizon (day)</th>
                      <th style={thStyle}>Projected RUL</th>
                      <th style={thStyle}>RUL band</th>
                      <th style={thStyle}>Projected streak</th>
                      <th style={thStyle}>Projected state</th>
                      <th style={thStyle}>Escalates</th>
                    </tr>
                  </thead>
                  <tbody>
                    {forecast.trajectory.map((t) => (
                      <tr key={t.day}>
                        <td style={tdStyle}>+{t.day}</td>
                        <td style={tdStyle}>{t.projected_rul_days != null ? `${fmt(t.projected_rul_days, 1)} d` : '—'}</td>
                        <td style={{ ...tdStyle, color: COLORS.subtext, fontSize: 13 }}>
                          {t.rul_low != null ? `[${fmt(t.rul_low, 1)}, ${fmt(t.rul_high, 1)}]` : '—'}
                        </td>
                        <td style={tdStyle}>{t.projected_streak}</td>
                        <td style={{ ...tdStyle, color: STATE_COLORS[t.projected_state] || COLORS.text }}>
                          {t.projected_state_name}
                        </td>
                        <td style={{ ...tdStyle, color: t.escalates_from_current ? COLORS.warn : COLORS.muted }}>
                          {t.escalates_from_current ? '▲ yes' : '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div style={{ color: COLORS.subtext, fontSize: 12, marginTop: 10 }}>
                Assumptions: {forecast.assumptions.join(' · ')}
              </div>
            </SectionCard>
          )}

          {/* Online learning feedback */}
          <SectionCard
            title="Teach the twin (genuine online learning)"
            subtitle="Feed a realised RUL back. Updates the RLS adapter parameters and conformal calibration — the twin's future predictions genuinely change."
            accent={COLORS.good}
          >
            <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end', flexWrap: 'wrap' }}>
              <div style={{ minWidth: 180 }}>
                <label style={{ color: COLORS.subtext, fontSize: 12, display: 'block', marginBottom: 4 }}>
                  Actual RUL observed (days)
                </label>
                <input
                  type="number"
                  value={actualRul}
                  onChange={(e) => setActualRul(e.target.value)}
                  placeholder="e.g. 18"
                  style={inputStyle}
                />
              </div>
              <button type="button" onClick={feedOutcome} disabled={loading || actualRul === ''} style={btnStyle(COLORS.good, loading || actualRul === '')}>
                Feed outcome
              </button>
            </div>
            {adaptation?.personal && adaptation.personal.n_updates > 0 && (
              <div style={{ marginTop: 12, display: 'grid', gap: 6 }}>
                <Row k="Personal learner updates" v={fmt(adaptation.personal.n_updates, 0)} color={COLORS.good} />
                <Row k="Learned slope b1" v={fmt(adaptation.personal.slope, 3)} />
                <Row k="Learned intercept b0" v={fmt(adaptation.personal.intercept, 3)} />
                <Row k="Mean abs residual" v={fmt(adaptation.personal.mean_abs_residual, 2)} />
              </div>
            )}
          </SectionCard>
        </>
      )}

      {/* Calibration diagnostics */}
      {calibration && (
        <SectionCard title="Calibration diagnostics" subtitle="Conformal + online-adapter state, seeded from Component 2 held-out residuals" accent={COLORS.purple}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 16 }}>
            <div style={{ display: 'grid', gap: 6 }}>
              <div style={{ color: COLORS.subtext, fontSize: 12, textTransform: 'uppercase' }}>Conformal</div>
              <Row k="Calibration samples" v={fmt(calibration.conformal?.calibration_samples, 0)} />
              <Row k="Median abs residual" v={fmt(calibration.conformal?.median_abs_residual, 2)} />
              <Row k="Conformity quantile" v={fmt(calibration.conformal?.conformity_quantile, 2)} />
              <Row k="Target coverage" v={`${fmt((calibration.conformal?.target_coverage ?? 0) * 100, 0)}%`} />
            </div>
            <div style={{ display: 'grid', gap: 6 }}>
              <div style={{ color: COLORS.subtext, fontSize: 12, textTransform: 'uppercase' }}>Online adapter (global)</div>
              <Row k="Updates" v={fmt(calibration.online_adapter?.global?.n_updates, 0)} />
              <Row k="Slope b1" v={fmt(calibration.online_adapter?.global?.slope, 3)} />
              <Row k="Intercept b0" v={fmt(calibration.online_adapter?.global?.intercept, 3)} />
              <Row k="Serials tracked" v={fmt(calibration.online_adapter?.serials_tracked, 0)} />
            </div>
          </div>
        </SectionCard>
      )}
    </div>
  );
}

function Row({ k, v, color = COLORS.text }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, fontSize: 14 }}>
      <span style={{ color: COLORS.subtext }}>{k}</span>
      <span style={{ color, fontWeight: 600 }}>{v}</span>
    </div>
  );
}

function btnStyle(color, disabled) {
  return {
    background: disabled ? '#94a3b8' : color,
    color: '#0f172a',
    border: 'none',
    borderRadius: 8,
    padding: '9px 16px',
    fontWeight: 700,
    cursor: disabled ? 'not-allowed' : 'pointer',
  };
}
