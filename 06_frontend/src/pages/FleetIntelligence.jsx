import React, { useEffect, useRef, useState } from 'react';
import { getShadowHistory, getShadowStates } from '../api/component1Api';
import { IconActivity, IconLayoutGrid, IconShield, IconRefreshCw } from '../components/common/Icons';
import '../styles/fleet.css';
import BackblazeReplay from '../components/component1/BackblazeReplay';
import ReplayMonitoring from '../components/component1/ReplayMonitoring';

const steps = [
  ['Collect daily SMART data', 'The client agent supplies one dated raw SMART record per drive. It may collect more often, but this scorer processes one valid record per calendar day.'],
  ['Validate the record', 'Check device identity, one of the five supported HDD models, and the five required raw SMART values. Missing or invalid values cannot be used.'],
  ['Build a 30-day history', 'Require 30 consecutive valid daily records for the same drive and model. A missing day resets the history; show Warming until ready.'],
  ['Score with the frozen model', 'Produce an anomaly score, compare it with that model’s frozen threshold, and display a research candidate flag when exceeded.'],
];
const number = value => value == null ? 'Unavailable' : String(value);
const statusLabel = drive => drive.status === 'unsupported_model' ? 'Unsupported' : drive.status === 'warming' ? 'Warming' : drive.status === 'scored' ? (drive.candidate_flag === true ? 'Candidate' : 'Scored') : 'Unavailable';

export default function FleetIntelligence() {
  const integration = useRef(null);
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [retry, setRetry] = useState(0);
  const [search, setSearch] = useState('');
  const [model, setModel] = useState('');
  const [status, setStatus] = useState('');
  const [monitorSource, setMonitorSource] = useState('live');
  const [replayView, setReplayView] = useState({ session: null, progress: 0, result: null });
  const [selectedLiveSerial, setSelectedLiveSerial] = useState('');
  const [liveHistory, setLiveHistory] = useState(null);
  const [liveHistoryError, setLiveHistoryError] = useState('');
  const [liveHistoryLoading, setLiveHistoryLoading] = useState(false);
  useEffect(() => {
    let active = true;
    let timer;
    const refresh = async () => {
      try {
        const response = await getShadowStates();
        if (active) { setData(response.data); setError(''); }
      } catch { if (active) setError('Unable to load device records. Check the backend connection and frozen scorer artifacts.'); }
      finally { if (active) timer = setTimeout(refresh, 10000); }
    };
    refresh();
    return () => { active = false; clearTimeout(timer); };
  }, [retry]);
  useEffect(() => {
    if (replayView.session?.replay_id) setMonitorSource('replay');
  }, [replayView.session?.replay_id]);
  const drives = data?.drives ?? [];
  useEffect(() => {
    if (!drives.length) setSelectedLiveSerial('');
    else if (!drives.some(drive => drive.serial_number === selectedLiveSerial)) setSelectedLiveSerial(drives[0].serial_number);
  }, [drives, selectedLiveSerial]);
  const selectedLiveDrive = drives.find(drive => drive.serial_number === selectedLiveSerial);
  useEffect(() => {
    let active = true;
    if (monitorSource !== 'live' || !selectedLiveSerial) {
      setLiveHistory(null);
      setLiveHistoryError('');
      return () => { active = false; };
    }
    setLiveHistoryLoading(true);
    getShadowHistory(selectedLiveSerial)
      .then(({ data: history }) => { if (active) { setLiveHistory(history); setLiveHistoryError(''); } })
      .catch(() => { if (active) { setLiveHistory(null); setLiveHistoryError('Unable to load this drive’s raw history.'); } })
      .finally(() => { if (active) setLiveHistoryLoading(false); });
    return () => { active = false; };
  }, [monitorSource, selectedLiveSerial, selectedLiveDrive?.end_date, selectedLiveDrive?.history_days]);
  const models = [...new Set([...(data?.supported_models ?? []), ...drives.map(d => d.model)])].sort();
  const visible = drives.filter(d => (!model || d.model === model) && (!status || (status === 'candidate' ? d.status === 'scored' && d.candidate_flag === true : d.status === status)) && `${d.serial_number} ${d.model}`.toLowerCase().includes(search.trim().toLowerCase()));
  const counts = [
    ['Devices observed', data ? drives.length : null, IconActivity],
    ['Ready to score', data ? drives.filter(d => d.status === 'scored').length : null, IconRefreshCw],
    ['Candidate flags', data ? drives.filter(d => d.status === 'scored' && d.candidate_flag === true).length : null, IconShield],
    ['Supported HDD models', data ? data.supported_models.length : 5, IconLayoutGrid],
  ];
  return <section className="fleet" aria-labelledby="fleet-title">
    <header className="fleet-heading">
      <div><p className="fleet-eyebrow">C1 · Anomaly detection</p><h1 id="fleet-title">Fleet Intelligence</h1></div>
      <span className="fleet-shadow"><IconShield size={16} />Research shadow mode</span>
    </header>
    <p className="fleet-banner"><span aria-hidden="true">ⓘ</span> Research anomaly candidate only · Not a failure probability. Live agent validation pending.</p>
    {error && <div className="fleet-error" role="alert"><div>{error}{data && <strong> Showing previously loaded records; counts and results may be stale.</strong>}</div><button type="button" onClick={() => setRetry(n => n + 1)}>Retry</button></div>}
    <div className="fleet-stats" aria-label="Fleet summary">
      {counts.map(([label, value, Icon]) => <article className="fleet-stat" key={label}><Icon size={29} /><div><h2>{label}</h2><strong>{value ?? '—'}</strong></div></article>)}
    </div>
    <BackblazeReplay onReplayChange={setReplayView} />
    <div className="fleet-grid">
      <section className="fleet-panel fleet-monitor" aria-labelledby="monitor-title">
        <div className="fleet-toolbar"><div><h2 id="monitor-title">Device monitoring</h2><div className="fleet-monitor-tabs" role="tablist" aria-label="Monitoring data source"><button type="button" role="tab" aria-selected={monitorSource === 'live'} onClick={() => setMonitorSource('live')}>Live devices</button><button type="button" role="tab" aria-selected={monitorSource === 'replay'} disabled={!replayView.session} onClick={() => setMonitorSource('replay')}>Historical replay</button></div></div>{monitorSource === 'live' && <div className="fleet-filters">
          <input type="search" aria-label="Search drives" placeholder="Search drives" value={search} onChange={e => setSearch(e.target.value)} />
          <select aria-label="Filter by model" value={model} onChange={e => setModel(e.target.value)}><option value="">All models</option>{models.map(m => <option key={m}>{m}</option>)}</select>
          <select aria-label="Filter by status" value={status} onChange={e => setStatus(e.target.value)}><option value="">All statuses</option><option value="warming">Warming</option><option value="scored">Scored</option><option value="candidate">Candidate</option><option value="unsupported_model">Unsupported</option></select>
        </div>}</div>
        {monitorSource === 'replay' ? <ReplayMonitoring replay={replayView} /> : <>
        <p className="fleet-scroll-hint">Scroll the table to view all columns.</p>
        <div className="fleet-table-scroll" tabIndex={0} role="region" aria-label="Device records table">
          <table className="fleet-table"><caption className="fleet-sr-only">Daily records from the frozen HDD anomaly scorer</caption><thead><tr>{['Drive', 'Model', 'Last record', 'History', 'Score / Threshold', 'Status', 'Details'].map(label => <th key={label} scope="col">{label}</th>)}</tr></thead>
            <tbody>{visible.map(d => <tr key={d.serial_number}>
              <th scope="row">{d.serial_number}</th><td>{d.model}</td><td className="fleet-date">{d.end_date ?? 'Unavailable'}</td>
              <td><span>{d.status === 'unsupported_model' ? 'Unavailable' : `${d.history_days}/30 days`}</span>{d.status !== 'unsupported_model' && <progress aria-label={`${d.serial_number} daily history`} value={d.history_days} max="30" />}{d.reset_reason === 'gap' && <small>Gap reset</small>}</td>
              <td className="fleet-score">{d.status === 'scored' ? <><span title={number(d.anomaly_score)}>{number(d.anomaly_score)}</span><small>/ {number(d.frozen_threshold)}</small></> : <span className="fleet-muted">Unavailable</span>}</td>
              <td><span className={`fleet-status fleet-status--${statusLabel(d).toLowerCase()}`}>{statusLabel(d)}</span></td>
              <td><details className="fleet-details"><summary aria-label={`Scorer details for ${d.serial_number}`}>View</summary><dl><dt>Scorer version</dt><dd>{d.scorer_version ?? 'Unavailable'}</dd><dt>Research candidate flag</dt><dd>{d.status === 'scored' && d.candidate_flag != null ? (d.candidate_flag ? 'Yes' : 'No') : 'Unavailable'}</dd><dt>Interpretation</dt><dd>{d.status === 'unsupported_model' ? 'This model is not supported by the frozen HDD detector.' : d.status === 'warming' ? 'Collecting 30 consecutive valid daily SMART records.' : 'Research anomaly candidate only; not a failure probability.'}</dd></dl></details></td>
            </tr>)}</tbody></table>
        </div>
        {!data && !error && <div className="fleet-empty" role="status"><IconRefreshCw size={34} /><h3>Loading device records</h3><p>Reading the latest shadow observations.</p></div>}
        {!data && error && <div className="fleet-empty"><IconActivity size={34} /><h3>Device records unavailable</h3><p>Retry when the API is available. No device counts can be shown yet.</p></div>}
        {data && drives.length === 0 && <div className="fleet-empty"><span className="fleet-empty-icon"><IconActivity size={34} /></span><h3>No device records yet</h3><p>Supply validated daily SMART records to start building 30-day histories.</p><a href="#fleet-integration" onClick={() => { if (integration.current) integration.current.open = true; }}>View integration requirements <span aria-hidden="true">↗</span></a></div>}
        {data && drives.length > 0 && visible.length === 0 && <div className="fleet-empty"><IconActivity size={34} /><h3>No matching devices</h3><p>Try a different drive, model, or status.</p><button onClick={() => { setSearch(''); setModel(''); setStatus(''); }}>Clear filters</button></div>}
        {data && drives.length > 0 && <p className="fleet-table-note">{visible.length} of {drives.length} devices · Ready to score means a scored 30-day window is available.</p>}
        <section className="fleet-live-charts" aria-labelledby="live-charts-title">
          <div className="fleet-live-charts__heading"><div><h3 id="live-charts-title">Live device health visualization</h3><p>Raw daily readings from the selected live shadow device.</p></div>{drives.length > 0 && <label>Select drive<select aria-label="Select live device for visualization" value={selectedLiveSerial} onChange={event => setSelectedLiveSerial(event.target.value)}>{drives.map(drive => <option key={drive.serial_number} value={drive.serial_number}>{drive.serial_number} · {drive.model}</option>)}</select></label>}</div>
          {liveHistoryError && <p role="alert" className="fleet-error">{liveHistoryError}</p>}
          <ReplayMonitoring mode="live" loading={liveHistoryLoading} replay={liveHistory ? { session: { source_serial: liveHistory.serial_number, model: liveHistory.model, start_date: liveHistory.start_date ?? 'No valid daily record', end_date: liveHistory.end_date, rows: liveHistory.rows }, progress: liveHistory.history_days, result: liveHistory.result } : null} />
        </section>
        </>}
      </section>
      <aside className="fleet-aside"><section className="fleet-panel fleet-how"><h2>How scoring works</h2><ol>{steps.map(([title, description]) => <li key={title}><h3>{title}</h3><p>{description}</p></li>)}</ol></section>
        <details className="fleet-panel fleet-integration" id="fleet-integration" ref={integration}><summary>Integration requirements</summary><p>The live agent mapping still needs validation: stable drive identity, exact model, raw units, and calendar-day timezone.</p><p>Required raw values: SMART 5, 9, 194, 197, and 198. Normalized values are not substitutes.</p>{data && <ul>{data.supported_models.map(m => <li key={m}>{m}</li>)}</ul>}</details>
      </aside>
    </div>
    <p className="fleet-limitations"><span aria-hidden="true">ⓘ</span> Regime/drift outputs and automatic healing are unavailable. SSDs are not validated.</p>
  </section>;
}
