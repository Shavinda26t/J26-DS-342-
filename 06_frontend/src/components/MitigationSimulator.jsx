import React, { useMemo, useState } from 'react';
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts';
import { simulateMitigation } from './component4Simulation';

const tile = {
  padding: 16, background: '#0f172a', border: '1px solid #334155',
  borderRadius: 10, minWidth: 0,
};
const small = { color: '#94a3b8', fontSize: 13 };
const main = { color: '#f8fafc', fontSize: 21, fontWeight: 700 };
const fmt = (value, digits = 2) => Number(value).toFixed(digits);

export default function MitigationSimulator({ form, decision }) {
  const [show, setShow] = useState(false);
  const simulation = useMemo(
    () => simulateMitigation(form, decision), [form, decision]
  );
  const first = simulation.timeline[0];
  const last = simulation.timeline[simulation.timeline.length - 1];
  const hasTransition = simulation.timeline.length > 1;

  return (
    <section className="card" style={{ display: 'grid', gap: 15, border: '1px solid #475569' }}>
      <div className="section-header">
        <h2 className="section-title">What-If Mitigation Simulator</h2>
        <span className="badge badge-purple">SIMULATED ONLY</span>
      </div>
      <p style={small}>
        Preview the hypothetical effect of the safety-checked recommendation on migration progress,
        free storage capacity and modeled failure consequences. The HDD failure-risk estimate is
        <strong> not reduced</strong> by migration.
      </p>
      <button
        type="button"
        onClick={() => setShow(current => !current)}
        style={{ justifySelf: 'start', padding: '12px 18px', borderRadius: 8,
          background: '#38bdf8', border: 'none', color: '#0f172a', fontWeight: 750, cursor: 'pointer' }}
      >
        {show ? 'Hide Simulation' : 'Simulate Recommended Action'}
      </button>
      {show && (
        <div style={{ display: 'grid', gap: 17 }} aria-live="polite">
          <div style={{ padding: 12, borderRadius: 8,
            border: `1px solid ${simulation.blocked ? '#fbbf24' : '#34d399'}`,
            background: '#0b1222', fontSize: 14 }}>
            <strong>{simulation.blocked ? 'Safety restriction — no intervention applied' : `What-if action: ${simulation.action}`}</strong>
            <p style={{ marginTop: 7, color: '#cbd5e1' }}>{simulation.reason}</p>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(205px,1fr))', gap: 12 }}>
            <div style={tile}>
              <div style={{ fontWeight: 700, marginBottom: 13 }}>BEFORE</div>
              <div style={small}>Migration progress</div><div style={main}>{fmt(first.progressPercent, 0)}%</div>
              <div style={{ ...small, marginTop: 9 }}>Destination free capacity</div>
              <div style={main}>{fmt(first.capacity, 1)} GB</div>
              <div style={{ ...small, marginTop: 9 }}>Modeled failure consequence</div>
              <div style={main}>{fmt(first.consequence, 2)} units</div>
            </div>
            <div style={{ ...tile, borderColor: hasTransition ? '#34d399' : '#64748b' }}>
              <div style={{ fontWeight: 700, marginBottom: 13 }}>AFTER HYPOTHETICAL ACTION</div>
              <div style={small}>Migration progress</div><div style={{ ...main, color: '#34d399' }}>{fmt(last.progressPercent, 0)}%</div>
              <div style={{ ...small, marginTop: 9 }}>Destination free capacity</div>
              <div style={main}>{fmt(last.capacity, 1)} GB</div>
              <div style={{ ...small, marginTop: 9 }}>Modeled failure consequence</div>
              <div style={main}>{fmt(last.consequence, 2)} units</div>
            </div>
          </div>
          <div style={{ ...tile, display: 'grid', gap: 9 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
              <strong>Simulated migration progress</strong>
              <span>{fmt(last.progressPercent, 0)}%</span>
            </div>
            <div style={{ height: 13, background: '#334155', borderRadius: 99, overflow: 'hidden' }}>
              <div style={{ width: `${last.progressPercent}%`, height: '100%', background: '#34d399' }} />
            </div>
            <span style={small}>HDD seven-day failure risk remains {fmt(simulation.risk * 100, 1)}% (input assumption).</span>
          </div>
          <div style={tile}>
            <h3 style={{ fontSize: 17, marginBottom: 8 }}>Modeled expected daily failure loss</h3>
            <p style={{ ...small, marginBottom: 14 }}>
              Action projection vs NO_OP at the same starting conditions. Lower is better.
              This chart is not a measured prediction and is not the total RL episode reward.
            </p>
            <div style={{ width: '100%', height: 270 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={simulation.timeline} margin={{ top: 4, right: 15, left: 1, bottom: 5 }}>
                  <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
                  <XAxis dataKey="label" tick={{ fill: '#cbd5e1', fontSize: 11 }} />
                  <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} width={47} />
                  <Tooltip contentStyle={{ background: '#0f172a', borderColor: '#64748b', color: '#f8fafc' }} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Line type="linear" dataKey="expectedLoss" name="Simulated intervention" stroke="#34d399" strokeWidth={3} dot />
                  <Line type="linear" dataKey="noOpLoss" name="NO_OP comparison" stroke="#fbbf24" strokeWidth={2} strokeDasharray="5 5" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div style={{ ...tile, overflowX: 'auto' }}>
            <h3 style={{ fontSize: 17, marginBottom: 12 }}>Simulated state transitions</h3>
            <table style={{ width: '100%', borderCollapse: 'collapse', color: '#e2e8f0', fontSize: 12 }}>
              <thead><tr>{['Stage', 'Progress', 'Moved in stage', 'Free capacity', 'Failure consequence'].map(v =>
                <th key={v} style={{ textAlign: 'left', padding: 8, borderBottom: '1px solid #64748b' }}>{v}</th>)}</tr></thead>
              <tbody>{simulation.timeline.map((point) => (
                <tr key={point.stage}>
                  <td style={{ padding: 8 }}>{point.label}</td>
                  <td style={{ padding: 8 }}>{fmt(point.progressPercent, 0)}%</td>
                  <td style={{ padding: 8 }}>{fmt(point.movedGb, 1)} GB</td>
                  <td style={{ padding: 8 }}>{fmt(point.capacity, 1)} GB</td>
                  <td style={{ padding: 8 }}>{fmt(point.consequence, 2)}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
          <p style={{ ...small, borderLeft: '3px solid #fbbf24', paddingLeft: 12 }}>
            {simulation.disclaimer} Uses constant seven-day risk, a 25%-per-step migration increment,
            a 90% assumed protection effectiveness and the V2 expected-loss equation.
          </p>
        </div>
      )}
    </section>
  );
}
