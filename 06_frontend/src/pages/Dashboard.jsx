import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { apiClient } from '../api/client';
import {
  getComponent2LeadTimeDistribution,
  getComponent2Overview,
  getComponent2TwinRiskGate,
  getComponent2TwinSerials,
} from '../api/component2Api';
import { getComponent3Dashboard, getComponent3Summary } from '../api/component3Api';
import { getComponent4RLDecision } from '../api/component4Api';
import {
  IconActivity,
  IconAlertCircle,
  IconArrowRight,
  IconCpu,
  IconGauge,
  IconNetwork,
  IconRefreshCw,
  IconShield,
  IconShieldCheck,
} from '../components/common/Icons';
import { formatEnumLabel } from '../utils/formatters';
import '../styles/dashboard.css';

const DEMO_DECISION_REQUEST = {
  disk_id: 'DEMO-HDD-001',
  failure_probability: 0.75,
  prediction_horizon_days: 7,
  rul_days: 10,
  storage: {
    free_capacity_gb: 1200,
    data_to_migrate_gb: 500,
    healthy_redundant_copies: 2,
    target_healthy: true,
    migration_progress_fraction: 0,
    io_utilization: 0.35,
    admin_approved: false,
  },
};

const SOURCE_LABELS = {
  health: 'Backend health',
  overview: 'Digital-twin overview',
  leadTime: 'Lead-time distribution',
  serials: 'Live twin serials',
  riskGate: 'Risk gate',
  c3Summary: 'Causal XAI summary',
  c3Dashboard: 'Causal XAI evidence',
  c4Decision: 'Safe decision demo',
};

const n = (value, digits = 1) => {
  if (value === null || value === undefined || value === '') return '—';
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return '—';
  return parsed.toFixed(digits);
};

const whole = (value) => {
  if (value === null || value === undefined || value === '') return '—';
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return '—';
  return Math.round(parsed).toLocaleString();
};

const pValue = (value) => {
  if (value === null || value === undefined) return '—';
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return '—';
  if (parsed === 0) return '< 1e-300';
  if (parsed < 0.001) {
    const exponent = Math.floor(Math.log10(parsed));
    const mantissa = parsed / 10 ** exponent;
    return `${mantissa.toFixed(1)} × 10^${exponent}`;
  }
  return parsed.toFixed(3);
};

const caseTone = { TP: 'tp', FP: 'fp', FN: 'fn', TN: 'tn' };

const CASE_MEANING = {
  TP: 'Failure correctly flagged',
  FP: 'Failure predicted, none occurred',
  FN: 'Failure missed by the model',
  TN: 'Healthy drive correctly cleared',
};

const Dashboard = () => {
  const [data, setData] = useState({});
  const [errors, setErrors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshedAt, setRefreshedAt] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);

    const sources = [
      ['health', () => apiClient.get('/api/v1/health')],
      ['overview', () => getComponent2Overview()],
      ['leadTime', () => getComponent2LeadTimeDistribution()],
      ['serials', () => getComponent2TwinSerials()],
      ['riskGate', () => getComponent2TwinRiskGate()],
      ['c3Summary', () => getComponent3Summary()],
      ['c3Dashboard', () => getComponent3Dashboard()],
      [
        'c4Decision',
        () =>
          getComponent4RLDecision({
            ...DEMO_DECISION_REQUEST,
            observed_at: new Date().toISOString(),
          }),
      ],
    ];

    const results = await Promise.allSettled(sources.map(([, request]) => request()));

    const next = {};
    const failed = [];

    results.forEach((result, index) => {
      const [key] = sources[index];
      if (result.status === 'fulfilled') {
        const payload = result.value;
        next[key] = payload && payload.data !== undefined ? payload.data : payload;
      } else {
        failed.push(SOURCE_LABELS[key] || key);
      }
    });

    setData(next);
    setErrors(failed);
    setRefreshedAt(new Date());
    setLoading(false);
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const view = useMemo(() => {
    const overview = data.overview || {};
    const fleet = overview.summary || {};
    const deployed = overview.config || {};
    const bands = Array.isArray(data.leadTime) ? data.leadTime : [];
    const c3 = data.c3Summary || {};
    const evidence = data.c3Dashboard || {};
    const candidates = evidence.root_cause_candidates || [];
    const matched = evidence.matched_failure_risk_evidence || [];
    const pathways = evidence.temporal_pathways || [];
    const cases = evidence.representative_cases || {};
    const serials = data.serials || {};
    const gate = data.riskGate?.risk_gate || null;
    const decision = data.c4Decision || null;

    const primary = candidates[0] || null;
    const primaryMatched =
      (primary &&
        matched.find((row) => row.factor_id === primary.factor_id)) ||
      matched[0] ||
      null;
    const pathway = pathways[0] || null;

    const bandMax = bands.reduce(
      (max, band) => Math.max(max, Number(band.hdd_count) || 0),
      1,
    );

    const caseOrder = ['TP', 'FP', 'FN', 'TN'];
    const caseRows = caseOrder
      .filter((key) => cases[key])
      .map((key) => {
        const entry = cases[key];
        const top = entry.top_factors?.[0] || null;
        return {
          type: key,
          factor: top?.factor_name || 'No local factor recorded',
          share: top?.local_importance_share_pct,
          direction: top?.local_direction,
          evidenceClass: top?.integrated_evidence_class,
        };
      });

    return {
      backendOnline: data.health?.status === 'online',
      fleet,
      deployed,
      bands,
      bandMax,
      c3,
      candidates,
      matched,
      pathways,
      primary,
      primaryMatched,
      pathway,
      cases: caseRows,
      serials,
      gate,
      decision,
      interpretationWarning: evidence.interpretation_warning || '',
    };
  }, [data]);

  if (loading) {
    return (
      <div className="dash">
        <div className="dash-skeleton" style={{ height: 150 }} />
        <div className="dash-kpis">
          {[0, 1, 2, 3].map((key) => (
            <div className="dash-skeleton" key={key} style={{ height: 112 }} />
          ))}
        </div>
        <div className="dash-state">
          <div className="ui-spinner ui-spinner--lg" />
          <h3 style={{ color: 'var(--ui-text)' }}>Loading system overview…</h3>
          <p>Aggregating held-out validation metrics across all four research modules.</p>
        </div>
      </div>
    );
  }

  const {
    backendOnline,
    fleet,
    deployed,
    bands,
    bandMax,
    c3,
    primary,
    primaryMatched,
    pathway,
    cases,
    serials,
    gate,
    decision,
    interpretationWarning,
  } = view;

  const allFailed = errors.length > 0 && Object.keys(data).length === 0;

  if (allFailed) {
    return (
      <div className="dash-state">
        <IconAlertCircle size={38} />
        <h3 style={{ color: 'var(--ui-danger-text)' }}>System overview unavailable</h3>
        <p>
          None of the backend services responded. Confirm the FastAPI server is running on
          port 8000 and that Component 2/3 artifacts are present.
        </p>
        <button type="button" className="ui-btn ui-btn--primary" onClick={load}>
          Retry connection
        </button>
      </div>
    );
  }

  const detectionRate = fleet.detection_rate_percent;
  const statusTone = backendOnline ? 'ok' : 'down';

  return (
    <div className="dash">
      {/* Hero */}
      <section className="dash-hero">
        <div className="dash-hero__main">
          <span className="dash-hero__eyebrow">
            <span className={`dash-dot dash-dot--${statusTone}`} />
            {backendOnline ? 'Backend online' : 'Backend unreachable'}
          </span>
          <h2 className="dash-hero__title">HDD Health Research Framework</h2>
          <p className="dash-hero__sub">
            Unified overview of fleet failure prediction, adaptive digital-twin lifecycle
            modelling, causal root-cause evidence, and safety-checked mitigation policy.
            All figures come from held-out validation artifacts — none are recomputed here.
          </p>
          <div className="dash-chips">
            <span className="dash-chip">
              <span className="dash-dot dash-dot--ok" />
              Data mode: held-out validation
            </span>
            <span className="dash-chip">
              <span className="dash-dot dash-dot--ok" />
              Prediction horizon: 30 days
            </span>
            <span className="dash-chip">
              <span className="dash-dot dash-dot--warn" />
              Actions: dry-run only
            </span>
            <span className="dash-chip">
              <span className="dash-dot dash-dot--ok" />
              Auth: local demo session
            </span>
          </div>
        </div>
        <div className="dash-hero__actions">
          <button type="button" className="ui-btn ui-btn--outline ui-btn--sm" onClick={load}>
            <IconRefreshCw size={15} />
            Refresh
          </button>
          <span className="dash-hero__meta">
            Last refreshed
            <br />
            {refreshedAt ? refreshedAt.toLocaleTimeString() : '—'}
          </span>
        </div>
      </section>

      {errors.length > 0 && (
        <div className="dash-callout dash-callout--warn">
          <IconAlertCircle size={18} />
          <div>
            <span className="dash-callout__title">
              {errors.length} panel{errors.length > 1 ? 's' : ''} could not load
            </span>
            {errors.join(' · ')}. Remaining panels are shown with live data.
          </div>
        </div>
      )}

      {/* Modules */}
      <div className="dash-head">
        <span className="dash-head__title">Research modules</span>
        <span className="dash-head__hint">Each module exposes its own dashboard</span>
      </div>

      <section className="dash-modules">
        <article className="dash-module dash-module--stub">
          <div className="dash-module__top">
            <span className="dash-module__icon dash-module__icon--amber">
              <IconActivity size={19} />
            </span>
            <div className="dash-module__heading">
              <div className="dash-module__title">
                Fleet Intelligence <span className="dash-module__tag">C1</span>
              </div>
              <div className="dash-module__desc">
                Behavioural drift and fleet-level monitoring.
              </div>
            </div>
            <span className="dash-dot dash-dot--warn" title="Service scaffolded" />
          </div>
          <div className="dash-module__metrics">
            <div>
              <div className="dash-module__metric-label">Pipeline</div>
              <div className="dash-module__metric-value">Scaffolded</div>
            </div>
            <div>
              <div className="dash-module__metric-label">Drift feed</div>
              <div className="dash-module__metric-value">Not wired</div>
            </div>
          </div>
          <Link className="dash-module__link" to="/fleet">
            Open module <IconArrowRight size={14} />
          </Link>
        </article>

        <article className="dash-module">
          <div className="dash-module__top">
            <span className="dash-module__icon dash-module__icon--green">
              <IconRefreshCw size={19} />
            </span>
            <div className="dash-module__heading">
              <div className="dash-module__title">
                Digital Twin <span className="dash-module__tag">C2</span>
              </div>
              <div className="dash-module__desc">
                Adaptive degradation detector and RUL lifecycle modelling.
              </div>
            </div>
            <span className="dash-dot dash-dot--ok" title="Live held-out metrics" />
          </div>
          <div className="dash-module__metrics">
            <div>
              <div className="dash-module__metric-label">Detection rate</div>
              <div className="dash-module__metric-value">{n(detectionRate, 1)}%</div>
            </div>
            <div>
              <div className="dash-module__metric-label">Lead time</div>
              <div className="dash-module__metric-value">{n(fleet.mean_lead_time_days, 1)} d</div>
            </div>
          </div>
          <Link className="dash-module__link" to="/digital-twin">
            Open module <IconArrowRight size={14} />
          </Link>
        </article>

        <article className="dash-module">
          <div className="dash-module__top">
            <span className="dash-module__icon">
              <IconNetwork size={19} />
            </span>
            <div className="dash-module__heading">
              <div className="dash-module__title">
                Causal XAI <span className="dash-module__tag">C3</span>
              </div>
              <div className="dash-module__desc">
                Integrated predictive, temporal and matched failure-risk evidence.
              </div>
            </div>
            <span className="dash-dot dash-dot--ok" title="Live evidence matrix" />
          </div>
          <div className="dash-module__metrics">
            <div>
              <div className="dash-module__metric-label">Canonical factors</div>
              <div className="dash-module__metric-value">{whole(c3.canonical_factor_count)}</div>
            </div>
            <div>
              <div className="dash-module__metric-label">Root candidates</div>
              <div className="dash-module__metric-value">
                {whole(c3.high_priority_root_cause_candidates)} high
              </div>
            </div>
          </div>
          <Link className="dash-module__link" to="/causal-xai">
            Open module <IconArrowRight size={14} />
          </Link>
        </article>

        <article className="dash-module">
          <div className="dash-module__top">
            <span className="dash-module__icon dash-module__icon--indigo">
              <IconShield size={19} />
            </span>
            <div className="dash-module__heading">
              <div className="dash-module__title">
                Safe Optimization <span className="dash-module__tag">C4</span>
              </div>
              <div className="dash-module__desc">
                Risk-aware mitigation policy with an independent safety validator.
              </div>
            </div>
            <span className="dash-dot dash-dot--ok" title="DQN policy available" />
          </div>
          <div className="dash-module__metrics">
            <div>
              <div className="dash-module__metric-label">Policy</div>
              <div className="dash-module__metric-value">DQN v2</div>
            </div>
            <div>
              <div className="dash-module__metric-label">Mode</div>
              <div className="dash-module__metric-value">Dry-run</div>
            </div>
          </div>
          <Link className="dash-module__link" to="/safe-rl">
            Open module <IconArrowRight size={14} />
          </Link>
        </article>
      </section>

      {/* Primary evidence + detection performance */}
      <section className="dash-cols">
        <article className="dash-panel">
          <div className="dash-panel__head">
            <span className="dash-panel__title">Primary root-cause evidence</span>
            {primary && (
              <span className="badge badge-purple">{formatEnumLabel(primary.classification)}</span>
            )}
          </div>
          <div className="dash-panel__body">
            {primary ? (
              <>
                <div>
                  <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--ui-text)' }}>
                    {primary.factor_name}
                  </div>
                  <div style={{ fontSize: '0.78rem', color: 'var(--ui-text-muted)', marginTop: 4 }}>
                    Ranked {whole(primary.global_shap_rank)} of {whole(c3.canonical_factor_count)} by
                    global SHAP consensus
                  </div>
                </div>

                <div className="dash-rows">
                  <div className="dash-row">
                    <span className="dash-row__label">Global SHAP rank</span>
                    <span className="dash-row__value">#{whole(primary.global_shap_rank)}</span>
                  </div>
                  <div className="dash-row">
                    <span className="dash-row__label">Stable PCMCI degree</span>
                    <span className="dash-row__value">{whole(primary.stable_temporal_degree)}</span>
                  </div>
                  <div className="dash-row">
                    <span className="dash-row__label">Matched incident pairs</span>
                    <span className="dash-row__value">
                      {whole(primaryMatched?.matched_pairs)}
                    </span>
                  </div>
                  <div className="dash-row">
                    <span className="dash-row__label">Treated vs control failures</span>
                    <span className="dash-row__value">
                      {whole(primaryMatched?.treated_failures)} vs{' '}
                      {whole(primaryMatched?.control_failures)}
                    </span>
                  </div>
                  <div className="dash-row">
                    <span className="dash-row__label">Matched 30-day risk difference</span>
                    <span className="dash-row__value dash-row__value--bad">
                      {primaryMatched ? `+${n(primaryMatched.risk_difference_pp, 2)} pp` : '—'}
                    </span>
                  </div>
                  <div className="dash-row">
                    <span className="dash-row__label">Clustered-bootstrap 95% CI</span>
                    <span className="dash-row__value">
                      {primaryMatched
                        ? `${n(primaryMatched.cluster_bootstrap_95ci_pp?.[0], 2)} – ${n(
                            primaryMatched.cluster_bootstrap_95ci_pp?.[1],
                            2,
                          )} pp`
                        : '—'}
                    </span>
                  </div>
                  <div className="dash-row">
                    <span className="dash-row__label">McNemar exact p-value</span>
                    <span className="dash-row__value">
                      {pValue(primaryMatched?.mcnemar_exact_p_value)}
                    </span>
                  </div>
                  <div className="dash-row">
                    <span className="dash-row__label">Max |SMD| after matching</span>
                    <span className="dash-row__value">
                      {n(primaryMatched?.max_abs_smd_after_matching, 3)}
                    </span>
                  </div>
                </div>

                {pathway && (
                  <div className="dash-callout dash-callout--info">
                    <IconNetwork size={17} />
                    <div>
                      <span className="dash-callout__title">Stable temporal pathway</span>
                      {pathway.source_factor_name} → {pathway.target_factor_name} · strongest lag{' '}
                      {n(pathway.strongest_lag_days, 0)} day
                      {Number(pathway.strongest_lag_days) === 1 ? '' : 's'} · detection rate{' '}
                      {n(pathway.strongest_detection_rate, 2)}
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="dash-callout">No root-cause candidates were returned.</div>
            )}
          </div>
        </article>

        <article className="dash-panel">
          <div className="dash-panel__head">
            <span className="dash-panel__title">Detection performance</span>
            <span className="badge badge-blue">C2</span>
          </div>
          <div className="dash-panel__body">
            <div className="dash-bars">
              {bands.length ? (
                bands.map((band) => {
                  const width = Math.max(
                    (Number(band.hdd_count) || 0) / bandMax * 100,
                    Number(band.hdd_count) ? 4 : 0,
                  );
                  return (
                    <div className="dash-bar-row" key={band.lead_time_band}>
                      <span>{band.lead_time_band}</span>
                      <span className="dash-bar">
                        <span className="dash-bar__fill" style={{ width: `${width}%` }} />
                      </span>
                      <span className="dash-bar__count">{whole(band.hdd_count)}</span>
                    </div>
                  );
                })
              ) : (
                <div className="dash-callout">Lead-time distribution unavailable.</div>
              )}
            </div>

            <div className="dash-rows">
              <div className="dash-row">
                <span className="dash-row__label">Failed drives in test split</span>
                <span className="dash-row__value">{whole(fleet.test_hdds)}</span>
              </div>
              <div className="dash-row">
                <span className="dash-row__label">Detected within 60 days</span>
                <span className="dash-row__value dash-row__value--ok">
                  {whole(fleet.detected)} ({n(detectionRate, 1)}%)
                </span>
              </div>
              <div className="dash-row">
                <span className="dash-row__label">Not detected</span>
                <span className="dash-row__value">{whole(fleet.not_detected)}</span>
              </div>
              <div className="dash-row">
                <span className="dash-row__label">Median persistence span</span>
                <span className="dash-row__value">
                  {n(fleet.median_persistence_span_days, 1)} d
                </span>
              </div>
              <div className="dash-row">
                <span className="dash-row__label">Deployed persistence / severity</span>
                <span className="dash-row__value">
                  {whole(deployed.persistence)} / {whole(deployed.min_severity)}
                </span>
              </div>
              <div className="dash-row">
                <span className="dash-row__label">Matches production config</span>
                <span
                  className={`dash-row__value ${
                    deployed.matches_production ? 'dash-row__value--ok' : 'dash-row__value--bad'
                  }`}
                >
                  {deployed.matches_production ? 'Yes' : 'No'}
                </span>
              </div>
            </div>

            <Link className="dash-module__link" to="/digital-twin">
              Open Digital Twin <IconArrowRight size={14} />
            </Link>
          </div>
        </article>
      </section>

      {/* Twin status + safe decision */}
      <section className="dash-cols">
        <article className="dash-panel">
          <div className="dash-panel__head">
            <span className="dash-panel__title">Live digital twin</span>
            <span className="badge badge-blue">Risk-gated adaptation</span>
          </div>
          <div className="dash-panel__body">
            <div className="dash-rows">
              <div className="dash-row">
                <span className="dash-row__label">Active twins</span>
                <span className="dash-row__value">{whole(serials.count)}</span>
              </div>
              <div className="dash-row">
                <span className="dash-row__label">Serial numbers</span>
                <span className="dash-row__value">
                  {Array.isArray(serials.serials) && serials.serials.length
                    ? serials.serials.join(', ')
                    : '—'}
                </span>
              </div>
              <div className="dash-row">
                <span className="dash-row__label">Risk-gate mode</span>
                <span className="dash-row__value dash-row__value--accent">
                  {gate?.mode ? String(gate.mode).toUpperCase() : '—'}
                </span>
              </div>
              <div className="dash-row">
                <span className="dash-row__label">Adaptation deployed</span>
                <span className="dash-row__value">
                  {gate ? (gate.deployed ? 'Yes' : 'No (shadow)') : '—'}
                </span>
              </div>
              <div className="dash-row">
                <span className="dash-row__label">Gate decision counts</span>
                <span className="dash-row__value">
                  {gate?.counts
                    ? `ADAPT ${whole(gate.counts.ADAPT)} · WAIT ${whole(
                        gate.counts.WAIT,
                      )} · REJECT ${whole(gate.counts.REJECT)}`
                    : '—'}
                </span>
              </div>
              <div className="dash-row">
                <span className="dash-row__label">Gate threshold / alpha</span>
                <span className="dash-row__value">
                  {gate ? `${n(gate.threshold, 1)} / ${n(gate.alpha, 2)}` : '—'}
                </span>
              </div>
            </div>
            <p style={{ fontSize: '0.78rem', color: 'var(--ui-text-muted)' }}>
              Online adaptation runs in shadow mode: updates are evaluated but never promoted
              without a validated challenger.
            </p>
            <Link className="dash-module__link" to="/digital-twin">
              Open live twin <IconArrowRight size={14} />
            </Link>
          </div>
        </article>

        <article className="dash-panel">
          <div className="dash-panel__head">
            <span className="dash-panel__title">Safe mitigation decision</span>
            <span className="badge badge-purple">Dry-run</span>
          </div>
          <div className="dash-panel__body">
            {decision ? (
              <>
                <div className="dash-rows">
                  <div className="dash-row">
                    <span className="dash-row__label">Policy</span>
                    <span className="dash-row__value">{decision.policy_name || '—'}</span>
                  </div>
                  <div className="dash-row">
                    <span className="dash-row__label">Checkpoint</span>
                    <span className="dash-row__value">
                      {decision.model_checkpoint || 'Rule-based baseline'}
                    </span>
                  </div>
                  <div className="dash-row">
                    <span className="dash-row__label">DQN proposed action</span>
                    <span className="dash-row__value">{decision.proposed_action || '—'}</span>
                  </div>
                  <div className="dash-row">
                    <span className="dash-row__label">Safety-approved action</span>
                    <span className="dash-row__value dash-row__value--accent">
                      {decision.recommended_action || '—'}
                    </span>
                  </div>
                  <div className="dash-row">
                    <span className="dash-row__label">Decision status</span>
                    <span
                      className={`dash-row__value ${
                        decision.decision_status === 'ELIGIBLE_DRY_RUN'
                          ? 'dash-row__value--ok'
                          : 'dash-row__value--bad'
                      }`}
                    >
                      {decision.decision_status || '—'}
                    </span>
                  </div>
                  <div className="dash-row">
                    <span className="dash-row__label">Filtered by safety</span>
                    <span className="dash-row__value">
                      {decision.filtered_by_safety ? 'Yes' : 'No'}
                    </span>
                  </div>
                </div>
                <p style={{ fontSize: '0.8rem', color: 'var(--ui-text-muted)', lineHeight: 1.5 }}>
                  {decision.explanation}
                </p>
              </>
            ) : (
              <div className="dash-callout">No decision returned by the policy service.</div>
            )}
            <Link className="dash-module__link" to="/safe-rl">
              Open Safe Optimization <IconArrowRight size={14} />
            </Link>
          </div>
        </article>
      </section>

      {/* Representative cases */}
      {cases.length > 0 && (
        <>
          <div className="dash-head">
            <span className="dash-head__title">Representative local explanations</span>
            <span className="dash-head__hint">C3 · 30-day model, held-out test cases</span>
          </div>
          <section className="dash-cases">
            {cases.map((row) => (
              <article className="dash-case" key={row.type}>
                <span className="dash-case__type">
                  <span className={`dash-case__dot dash-case__dot--${caseTone[row.type] || 'tn'}`} />
                  {row.type} · {CASE_MEANING[row.type]}
                </span>
                <div className="dash-case__factor">{row.factor}</div>
                <div className="dash-case__meta">
                  {n(row.share, 1)}% of local attribution ·{' '}
                  {row.direction === 'TOWARD_FAILURE_OUTPUT' ? 'toward failure' : 'away from failure'}
                </div>
              </article>
            ))}
          </section>
        </>
      )}

      {/* Interpretation warning */}
      <div className="dash-callout dash-callout--warn">
        <IconAlertCircle size={18} />
        <div>
          <span className="dash-callout__title">Interpretation boundary</span>
          {interpretationWarning ||
            'Root-cause classifications integrate predictive, temporal and matched observational evidence. They do not constitute proof of physical causation.'}
        </div>
      </div>

      {/* Footer status strip */}
      <section className="dash-panel">
        <div className="dash-panel__head">
          <span className="dash-panel__title">Runtime status</span>
          <span className="dash-chip">
            <span className={`dash-dot dash-dot--${backendOnline ? 'ok' : 'down'}`} />
            {backendOnline ? 'API healthy' : 'API unreachable'}
          </span>
        </div>
        <div className="dash-panel__body dash-panel__body--tight">
          <div className="dash-toolbar">
            <span className="dash-chip">
              <IconShieldCheck size={13} />
              No real storage operations are executed
            </span>
            <span className="dash-chip">
              <IconCpu size={13} />
              Models served from repository artifacts
            </span>
            <span className="dash-chip">
              <IconGauge size={13} />
              Metrics are read-only
            </span>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Dashboard;
