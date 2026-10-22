import React from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const tooltipStyle = {
  background: '#081426',
  border: '1px solid #304b70',
  borderRadius: 8,
  color: '#ffffff',
  fontSize: 12,
};
const tooltipTextStyle = { color: '#ffffff' };

const compactDate = value => value?.slice(5);
const formatNumber = value => Number(value).toLocaleString(undefined, { maximumFractionDigits: 6 });

function EmptyCharts({ loading }) {
  const cards = [
    ['Power-on time and drive temperature', 'Waiting for the live agent to supply raw operating-hours and temperature readings.'],
    ['Drive sector health', 'Waiting for reallocated, pending, and uncorrectable sector counts.'],
    ['Frozen score comparison', 'A score and locked threshold appear after 30 consecutive valid daily records.'],
  ];
  return <div className="fleet-charts fleet-charts--empty">
    {cards.map(([title, description]) => <article className="fleet-chart" key={title}><div><h3>{title}</h3><p>{description}</p></div><div className="fleet-chart__empty"><span aria-hidden="true">⌁</span><strong>{loading ? 'Loading live history' : 'Waiting for live device data'}</strong><small>No sample values are displayed.</small></div></article>)}
  </div>;
}

export default function ReplayMonitoring({ replay, mode = 'replay', loading = false }) {
  const session = replay?.session;
  if (!session) {
    if (mode === 'live') return <><p className="fleet-replay-monitor__note">Live charts will populate from persisted raw daily records when the collection agent is connected.</p><EmptyCharts loading={loading} /></>;
    return <div className="fleet-empty"><h3>No historical replay loaded</h3><p>Load a validated Backblaze window above to visualize its raw drive-health history.</p></div>;
  }

  const rows = session.rows.map(row => ({
    ...row,
    dateLabel: compactDate(row.date),
    smart_5_raw: Number(row.smart_5_raw),
    smart_9_raw: Number(row.smart_9_raw),
    smart_194_raw: Number(row.smart_194_raw),
    smart_197_raw: Number(row.smart_197_raw),
    smart_198_raw: Number(row.smart_198_raw),
  }));
  const result = replay.result;
  const scoreData = result?.status === 'scored' ? [
    { name: 'Anomaly score', value: result.anomaly_score, fill: result.candidate_flag ? '#f59e0b' : '#38bdf8' },
    { name: 'Locked threshold', value: result.frozen_threshold, fill: '#818cf8' },
  ] : [];

  const isLive = mode === 'live';
  return <div className="fleet-replay-monitor" aria-label={isLive ? 'Live device visualization' : 'Historical replay visualization'}>
    <header className="fleet-replay-monitor__header">
      <div><h3>{session.source_serial}</h3><p>{session.model}</p></div>
      <div><strong>{session.start_date}</strong><span> to </span><strong>{session.end_date}</strong></div>
      <span className={`fleet-status ${result?.status === 'scored' ? (result.candidate_flag ? 'fleet-status--candidate' : 'fleet-status--scored') : 'fleet-status--warming'}`}>
        {result?.status === 'scored' ? (result.candidate_flag ? 'Candidate' : 'Scored') : `Warming ${replay.progress}/30`}
      </span>
    </header>
    <p className="fleet-replay-monitor__note">{isLive ? 'Live shadow observations for the selected drive.' : 'Historical Backblaze compatibility replay.'} The charts translate the five observed raw drive-health readings into plain-language labels; no normalized values or failure labels are included.</p>
    {rows.length === 0 ? <EmptyCharts loading={loading} /> : <>
    <div className="fleet-charts">
      <article className="fleet-chart">
        <div><h3>Power-on time and drive temperature</h3><p>How long the drive has operated and its reported temperature across the selected 30 calendar days.</p></div>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={rows} margin={{ top: 12, right: 8, left: 0, bottom: 4 }}>
            <CartesianGrid stroke="#263e5a" strokeDasharray="3 3" />
            <XAxis dataKey="dateLabel" stroke="#8fa9cb" tick={{ fontSize: 11 }} minTickGap={24} />
            <YAxis yAxisId="hours" stroke="#38bdf8" tick={{ fontSize: 11 }} width={52} tickFormatter={formatNumber} />
            <YAxis yAxisId="temp" orientation="right" stroke="#f59e0b" tick={{ fontSize: 11 }} width={38} />
            <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipTextStyle} itemStyle={tooltipTextStyle} labelFormatter={label => `Date: ${label}`} formatter={(value, name) => [formatNumber(value), name]} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line yAxisId="hours" type="monotone" dataKey="smart_9_raw" name="Power-on hours" unit=" hours" stroke="#38bdf8" strokeWidth={2} dot={false} />
            <Line yAxisId="temp" type="monotone" dataKey="smart_194_raw" name="Drive temperature" unit=" °C" stroke="#f59e0b" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </article>
      <article className="fleet-chart">
        <div><h3>Drive sector health</h3><p>Reallocated sectors were remapped, pending sectors are waiting for recovery, and uncorrectable sectors could not be recovered.</p></div>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={rows} margin={{ top: 12, right: 8, left: 0, bottom: 4 }}>
            <CartesianGrid stroke="#263e5a" strokeDasharray="3 3" />
            <XAxis dataKey="dateLabel" stroke="#8fa9cb" tick={{ fontSize: 11 }} minTickGap={24} />
            <YAxis stroke="#a7b9d4" tick={{ fontSize: 11 }} width={52} domain={[0, 'auto']} tickFormatter={formatNumber} />
            <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipTextStyle} itemStyle={tooltipTextStyle} labelFormatter={label => `Date: ${label}`} formatter={(value, name) => [formatNumber(value), name]} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey="smart_5_raw" name="Reallocated sectors" stroke="#60a5fa" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="smart_197_raw" name="Pending sectors" stroke="#a78bfa" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="smart_198_raw" name="Uncorrectable sectors" stroke="#34d399" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </article>
      <article className="fleet-chart fleet-chart--score">
        <div><h3>Frozen score comparison</h3><p>{result?.status === 'scored' ? 'Day-30 sector anomaly score compared with this model’s locked threshold.' : 'Available after all 30 consecutive daily records have been accepted.'}</p></div>
        {scoreData.length ? <>
          <ResponsiveContainer width="100%" height={210}>
            <BarChart data={scoreData} layout="vertical" margin={{ top: 12, right: 20, left: 16, bottom: 4 }}>
              <CartesianGrid stroke="#263e5a" strokeDasharray="3 3" />
              <XAxis type="number" stroke="#8fa9cb" tick={{ fontSize: 11 }} tickFormatter={formatNumber} />
              <YAxis type="category" dataKey="name" stroke="#a7b9d4" tick={{ fontSize: 11 }} width={106} />
              <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipTextStyle} itemStyle={tooltipTextStyle} formatter={value => [formatNumber(value), 'Value']} />
              <Bar dataKey="value" radius={[0, 5, 5, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <p className="fleet-chart__result"><strong>{result.candidate_flag ? 'Research candidate flag: Yes' : 'Research candidate flag: No'}</strong> · Score {formatNumber(result.anomaly_score)} · Threshold {formatNumber(result.frozen_threshold)}</p>
        </> : <div className="fleet-chart__waiting"><progress max="30" value={replay.progress} /><span>{replay.progress}/30 valid daily records</span></div>}
      </article>
    </div>
    </>}
    <p className="fleet-replay-monitor__limit">This graph is descriptive. The score is not a failure probability and does not trigger alerts or healing.</p>
  </div>;
}
