
import React, { useState } from 'react';

import {
  getComponent4SafeDecision,
  getComponent4RLDecision,
} from '../api/component4Api';


const initialForm = {
  disk_id: 'DEMO_HDD_001',
  risk_percent: '75',
  rul_days: '10',
  free_capacity_gb: '1200',
  data_to_migrate_gb: '500',
  healthy_redundant_copies: '2',
  migration_progress_fraction: '0',
  io_utilization: '0.35',
  target_healthy: true,
  admin_approved: false,
};

const numericFields = [
  { key: 'risk_percent', label: '7-day failure risk (%)', min: 0, max: 100, step: 0.1 },
  { key: 'rul_days', label: 'Remaining useful life (days)', min: 0, step: 1 },
  { key: 'free_capacity_gb', label: 'Free destination capacity (GB)', min: 0, step: 1 },
  { key: 'data_to_migrate_gb', label: 'Data to migrate (GB)', min: 0, step: 1 },
  { key: 'healthy_redundant_copies', label: 'Healthy redundant copies', min: 0, step: 1 },
  { key: 'migration_progress_fraction', label: 'Migration progress (0–1)', min: 0, max: 1, step: 0.05 },
  { key: 'io_utilization', label: 'I/O utilization (0–1)', min: 0, max: 1, step: 0.05 },
];

const inputStyle = {
  width: '100%',
  padding: '10px 12px',
  borderRadius: 8,
  background: '#0f172a',
  color: '#f8fafc',
  border: '1px solid #475569',
  marginTop: 5,
};

const labelStyle = {
  display: 'block',
  color: '#cbd5e1',
  fontSize: 13,
  fontWeight: 500,
};

function SafeOptimization() {
  const [form, setForm] = useState(initialForm);
  const [policyMode, setPolicyMode] = useState('TRAINED_DQN');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');


const update = (key, value) => {
  setForm((previous) => ({
    ...previous,
    [key]: value,
  }));

  // A previous recommendation no longer matches
  // the HDD inputs after any field is changed.
  setResult(null);
  setError('');
};


  const submitDecision = async (event) => {
    event.preventDefault();
    setError('');
    setResult(null);

    const values = Object.fromEntries(
      numericFields.map(({ key }) => [key, Number(form[key])])
    );

    const invalidNumber = numericFields.some(({ key, min, max }) => {
      const value = values[key];

      return (
        form[key] === '' ||
        !Number.isFinite(value) ||
        (min !== undefined && value < min) ||
        (max !== undefined && value > max)
      );
    });

    if (
      !form.disk_id.trim() ||
      invalidNumber ||
      !Number.isInteger(values.healthy_redundant_copies)
    ) {
      setError('Please enter a disk ID and valid numeric values.');
      return;
    }

    const request = {
      disk_id: form.disk_id.trim(),
      observed_at: new Date().toISOString(),
      failure_probability: values.risk_percent / 100,
      prediction_horizon_days: 7,
      rul_days: values.rul_days,
      storage: {
        free_capacity_gb: values.free_capacity_gb,
        data_to_migrate_gb: values.data_to_migrate_gb,
        healthy_redundant_copies: values.healthy_redundant_copies,
        target_healthy: form.target_healthy,
        migration_progress_fraction: values.migration_progress_fraction,
        io_utilization: values.io_utilization,
        admin_approved: form.admin_approved,
      },
    };

    setLoading(true);

    try {const response =
        policyMode === 'TRAINED_DQN'
            ? await getComponent4RLDecision(request)
            : await getComponent4SafeDecision(request);

        setResult(response);
    } catch (err) {
      const detail = err.response?.data?.detail;
      const message = typeof detail === 'string'
        ? detail
        : err.message || 'Unknown request error';

      setError(
        `Could not retrieve the recommendation: ${message}. ` +
        'Check that the FastAPI backend is running.'
      );
    } finally {
      setLoading(false);
    }
  };

  const statusColor = result?.decision_status === 'ELIGIBLE_DRY_RUN'
    ? '#34d399'
    : result?.decision_status === 'PENDING_ADMIN_APPROVAL'
      ? '#fbbf24'
      : '#fca5a5';

  return (
    <div style={{ display: 'grid', gap: 22 }}>
      <header className="c3-header">
        <div className="c3-title-row">
          <h1>Safe HDD Mitigation</h1>
          <span className="badge badge-purple">Component 4</span>
        </div>

        <p className="c3-subtitle">
          Risk-aware recommendations with independent safety validation.
        </p>
      </header>

      
    <div style={{ marginBottom: 20 }}>

        <div
            style={{
                padding: '10px 14px',
                border: '1px solid #334155',
                borderRadius: 8,
                color: '#94a3b8',
                fontSize: 13,
            }}
            >
            {policyMode === 'TRAINED_DQN'
                ? (
                'Model: Trained DQN with independent safety validation. ' +
                'Risk and RUL are supplied as inputs. ' +
                'Recommendations are advisory; no physical HDD operations occur.'
                )
                : (
                'Model: Rule-based mitigation baseline with independent ' +
                'safety validation. Recommendations are advisory.'
                )
            }
        </div>
      </div>

      <form onSubmit={submitDecision} className="card">
        
        <div style={{ marginBottom: 20 }}>
        <label style={labelStyle}>
            Mitigation Decision Model

            <select
            value={policyMode}
            onChange={(event) => {
                setPolicyMode(event.target.value);
                setResult(null);
                setError('');
            }}
            style={{
                ...inputStyle,
                maxWidth: 420,
                cursor: 'pointer',
            }}
            >
            <option value="TRAINED_DQN">
                Trained DQN + Independent Safety Validator
            </option>

            <option value="RULE_BASED">
                Rule-Based Safety Baseline
            </option>
            </select>
        </label>
        </div>

        <h2 className="section-title" style={{ marginBottom: 8 }}>
          HDD Risk and Storage Context
        </h2>

        <p style={{ color: '#94a3b8', marginBottom: 18 }}>
          Edit the values to investigate different mitigation decisions.
        </p>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))',
          gap: 16,
        }}>
          <label style={labelStyle}>
            Disk ID
            <input
              type="text"
              style={inputStyle}
              value={form.disk_id}
              onChange={(e) => update('disk_id', e.target.value)}
              required
            />
          </label>

          {numericFields.map((field) => (
            <label key={field.key} style={labelStyle}>
              {field.label}
              <input
                type="number"
                style={inputStyle}
                value={form[field.key]}
                min={field.min}
                max={field.max}
                step={field.step}
                onChange={(e) => update(field.key, e.target.value)}
                required
              />
            </label>
          ))}
        </div>

        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 24,
          marginTop: 22,
          marginBottom: 22,
        }}>
          <label style={labelStyle}>
            <input
              type="checkbox"
              checked={form.target_healthy}
              onChange={(e) => update('target_healthy', e.target.checked)}
              style={{ marginRight: 8 }}
            />
            Migration destination healthy
          </label>

          <label style={labelStyle}>
            <input
              type="checkbox"
              checked={form.admin_approved}
              onChange={(e) => update('admin_approved', e.target.checked)}
              style={{ marginRight: 8 }}
            />
            Simulated administrator approval
          </label>
        </div>

        <button
          type="submit"
          disabled={loading}
          style={{
            background: '#38bdf8',
            color: '#0f172a',
            border: 'none',
            borderRadius: 8,
            padding: '12px 20px',
            fontWeight: 700,
            cursor: loading ? 'wait' : 'pointer',
            opacity: loading ? 0.7 : 1,
          }}
        >
          {loading ? 'Evaluating...' : 'Generate Safe Recommendation'}
        </button>

        {error && (
          <p role="alert" style={{
            color: '#fca5a5',
            marginTop: 16,
            overflowWrap: 'anywhere',
          }}>
            {error}
          </p>
        )}
      </form>

      {result && (
        <section className="card" aria-live="polite">
          <div className="section-header">
            <h2 className="section-title">Decision Result</h2>
            <span className="badge badge-blue">
              {result.execution_mode}
            </span>
          </div>

          <div className="metrics-grid">
            {[
            [
                'Decision source',
                result.model_checkpoint
                ? 'Trained DQN'
                : 'Rule-based baseline',
            ],

            ...(result.risk_level
                ? [['Risk classification', result.risk_level]]
                : []),

            ['Proposed action', result.proposed_action],
            ['Safe recommendation', result.recommended_action],
            ].map(([label, value]) => (

              <div className="card metric-card" key={label}>
                <span className="metric-label">{label}</span>
                <strong style={{
                  fontSize: 17,
                  overflowWrap: 'anywhere',
                }}>
                  {value}
                </strong>
              </div>
            ))}
          </div>

          <div style={{
            marginTop: 20,
            padding: 16,
            border: `1px solid ${statusColor}`,
            borderRadius: 10,
            background: '#0f172a',
          }}>
            <strong style={{ color: statusColor }}>
              Decision status: {result.decision_status}
            </strong>
            <p style={{ marginTop: 10, color: '#cbd5e1' }}>
              {result.explanation}
            </p>
          </div>

            {result.model_checkpoint && (
            <div
                style={{
                marginTop: 18,
                color: '#94a3b8',
                fontSize: 13,
                }}
            >
                <p>
                Trained checkpoint: {result.model_checkpoint}
                </p>

                <p>
                Safety filtering applied:{' '}
                {result.filtered_by_safety ? 'Yes' : 'No'}
                </p>
            </div>
            )}

          <h3 style={{ marginTop: 24, marginBottom: 12 }}>
            Independent Safety Assessment
          </h3>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
            gap: 12,
            color: '#cbd5e1',
          }}>
            <p>
              Technically eligible:{' '}
              <strong>
                {result.safety?.technically_eligible ? 'Yes' : 'No'}
              </strong>
            </p>

            <p>
              Approval required:{' '}
              <strong>
                {result.safety?.requires_admin_approval ? 'Yes' : 'No'}
              </strong>
            </p>

            <p>
              Approval pending:{' '}
              <strong>
                {result.safety?.approval_pending ? 'Yes' : 'No'}
              </strong>
            </p>

           
            <p>
            {result.model_checkpoint
                ? 'Validator default fallback: '
                : 'Safety fallback: '}
            <strong>{result.safety?.fallback_action}</strong>
            </p>

          </div>
        
          
            {result.model_checkpoint &&
            result.filtered_by_safety &&
            result.proposed_action === 'MIGRATE_DATA' && (
                <p
                style={{
                    marginTop: 16,
                    color: '#cbd5e1',
                    fontSize: 13,
                }}
                >
                DQN safety filter: the proposed migration was
                rejected by safety checks. The experimental DQN
                filter substituted{' '}
                <strong>{result.recommended_action}</strong>.
                The validator's general fallback is{' '}
                <strong>{result.safety?.fallback_action}</strong>.
                </p>
            )}


          {result.safety?.violations?.length > 0 && (
            <div style={{ marginTop: 18 }}>
              <strong style={{ color: '#fca5a5' }}>
                Safety violations
              </strong>
              <ul style={{ paddingLeft: 24, marginTop: 8 }}>
                {result.safety.violations.map((violation, index) => (
                  <li key={index}>{violation}</li>
                ))}
              </ul>
            </div>
          )}

          <p style={{
            color: '#94a3b8',
            marginTop: 20,
            fontSize: 13,
          }}>
            Policy: {result.policy_name}.
            This recommendation is advisory and does not
            perform migration, isolation or replacement.
          </p>
        </section>
      )}
    </div>
  );
}

export default SafeOptimization;
