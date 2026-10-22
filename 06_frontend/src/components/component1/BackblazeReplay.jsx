import React, { useEffect, useRef, useState } from 'react';
import { apiClient } from '../../api/client';

const root = '/api/v1/component1';
const fields = ['serial_number', 'model', 'date', 'smart_5_raw', 'smart_9_raw', 'smart_194_raw', 'smart_197_raw', 'smart_198_raw'];
const message = error => typeof error.response?.data?.detail === 'string' ? error.response.data.detail : 'Request interrupted or API unavailable. Your accepted daily records are saved. Retry or resume when the API is available.';
const storageKey = 'hdd.backblaze.replay';

export default function BackblazeReplay({ onReplayChange }) {
  const [session, setSession] = useState(null);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [events, setEvents] = useState([]);
  const [archive, setArchive] = useState('D:\\Downloads\\data_Q4_2022.zip');
  const stopped = useRef(false);
  const mounted = useRef(true);
  const upload = useRef(null);
  const apply = data => { setSession(data); setProgress(data.completed_days ?? 0); setResult(data.result ?? null); };
  useEffect(() => {
    mounted.current = true;
    const token = localStorage.getItem(storageKey);
    if (token) {
      setBusy(true);
      apiClient.get(`${root}/replay/${token}`).then(({ data }) => { if (mounted.current) apply(data); })
        .catch(e => { if (mounted.current) setError(message(e)); })
        .finally(() => { if (mounted.current) setBusy(false); });
    }
    return () => { mounted.current = false; stopped.current = true; upload.current?.abort(); };
  }, []);
  useEffect(() => {
    onReplayChange?.({ session, progress, result });
  }, [onReplayChange, progress, result, session]);
  const prepare = async event => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.csv') || file.size > 65536) {
      setError('Select the extracted 30-row CSV, up to 64 KiB. The quarterly ZIP must be streamed locally using the command below.');
      return;
    }
    setBusy(true); setError('');
    upload.current = new AbortController();
    try {
      const { data } = await apiClient.post(`${root}/replay/prepare`, file, { headers: { 'Content-Type': 'text/csv' }, signal: upload.current.signal });
      if (!mounted.current) return;
      apply(data); setEvents([]); localStorage.setItem(storageKey, data.replay_id);
    } catch (e) { if (mounted.current) setError(message(e)); }
    finally { if (mounted.current) setBusy(false); upload.current = null; }
  };
  const run = async () => {
    setBusy(true); setError(''); stopped.current = false;
    try {
      const { data } = await apiClient.get(`${root}/replay/${session.replay_id}`);
      if (!mounted.current) return;
      apply(data);
      for (let i = data.completed_days; i < 30 && !stopped.current; i++) {
        const response = await apiClient.post(`${root}/shadow/records`, data.rows[i], { params: { replay_id: data.replay_id } });
        if (!mounted.current) return;
        setProgress(i + 1); setResult(response.data);
        setEvents(previous => [...previous, `Day ${i + 1}: ${data.rows[i].date} - ${response.data.status}`]);
      }
    } catch (e) { if (mounted.current) setError(message(e)); }
    finally { if (mounted.current) setBusy(false); }
  };
  const download = async () => {
    setError('');
    try {
      const response = await apiClient.get(`${root}/replay/${session.replay_id}/csv`, { responseType: 'blob' });
      const url = URL.createObjectURL(response.data);
      const link = document.createElement('a'); link.href = url; link.download = 'backblaze_window.csv'; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) { setError(message(e)); }
  };
  const quoted = "'" + archive.replace(/'/g, "''").replace(/[\r\n]/g, '') + "'";
  return <details className="fleet-panel fleet-replay"><summary>Replay Backblaze data</summary>
    <p>Historical compatibility test only. The frozen model and thresholds do not change. Research anomaly candidate only; not a failure probability.</p>
    <h3>1. Extract locally from a quarterly ZIP</h3>
    <p>The large archive stays on your computer. This app uploads only a validated 30-row CSV, not the ZIP. Run the command from the repository root, on the computer holding the archive.</p>
    <label>Full path to your quarterly ZIP<input aria-label="Quarterly ZIP path" value={archive} onChange={e => setArchive(e.target.value)} /></label>
    <pre><code>{`.\\.venv\\Scripts\\python 05_backend\\tools\\backblaze_replay.py extract ${quoted} --output backblaze_window.csv --overwrite`}</code></pre>
    <p>The tool streams daily CSVs, selects a supported drive with 30 consecutive valid days, and prints extraction errors here in your terminal: missing raw columns or values, duplicates, unsupported models, or no complete window. It never exports failure labels.</p>
    <a href="https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q4_2022.zip">Download official Backblaze 2022 Q4 ZIP</a>
    <h3>2. Load the extracted window</h3>
    <label className="fleet-replay-file">Select extracted CSV<input type="file" accept=".csv,text/csv" disabled={busy} onChange={prepare} /></label>
    {busy && upload.current && <button type="button" onClick={() => upload.current?.abort()}>Cancel upload</button>}
    {error && <p role="alert" className="fleet-error">{error}</p>}
    {session && <>
      <p><strong>Source drive:</strong> {session.source_serial} | <strong>Model:</strong> {session.model}<br /><strong>Date range:</strong> {session.start_date} to {session.end_date} (30 calendar days)</p>
      <p className="fleet-replay-id">Isolated identity: {session.replay_identity}. Replay history is stored separately and is excluded from the live device table and counts.</p>
      <button type="button" onClick={download}>Download generated CSV</button>
      <div className="fleet-replay-preview" role="region" tabIndex={0} aria-label="Extracted 30-row preview"><table className="fleet-table"><caption>Extracted 30-row preview (raw SMART only)</caption><thead><tr>{fields.map(f => <th key={f}>{f}</th>)}</tr></thead><tbody>{session.rows.map(row => <tr key={row.date}>{fields.map(f => <td key={f}>{row[f]}</td>)}</tr>)}</tbody></table></div>
      <h3>3. Run replay</h3><p>Records are sent in chronological order, one daily record per request. These are 30 historical calendar days, replayed faster than real time.</p>
      <div className="fleet-replay-actions"><button type="button" disabled={busy || progress === 30} onClick={run}>{progress === 30 ? 'Replay complete' : progress ? 'Resume replay' : 'Run replay'}</button>
        {busy && !upload.current && <button type="button" onClick={() => { stopped.current = true; }}>Pause after current record</button>}
      </div>
      <p role="status">{progress}/30 daily records accepted{progress > 0 && progress < 30 ? ' - Warming' : ''}{progress === 30 ? ' - Scored' : ''}</p><progress max="30" value={progress} aria-label="Replay daily history" />
      {progress === 30 && <p>This replay is complete. Select the CSV again to create a new isolated replay; the previous run will not be mixed with it.</p>}
      {events.length > 0 && <details><summary>Daily replay responses</summary><ol>{events.map((text, i) => <li key={i}>{text}</li>)}</ol></details>}
      {result?.status === 'scored' && <section className="fleet-replay-result" aria-label="Day 30 frozen response"><h3>Day 30 frozen response</h3><dl>{[['Anomaly score', result.anomaly_score], ['Locked threshold', result.frozen_threshold], ['Research candidate flag', result.candidate_flag ? 'Yes' : 'No'], ['Scoring date', result.end_date], ['Model', result.model], ['Model version', result.model_version], ['Scorer version', result.scorer_version]].map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{String(value ?? 'Unavailable')}</dd></div>)}</dl><p>Historical compatibility result; not a failure probability. No live alerts or automatic healing.</p></section>}
    </>}
  </details>;
}
