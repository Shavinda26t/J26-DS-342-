# Frozen HDD anomaly shadow integration

The Fleet Intelligence `/fleet` page reads `GET /api/v1/component1/shadow` every ten seconds. Submit a **research-contract** record to `POST /api/v1/component1/shadow/records`. This path is independent of digital-twin warnings, notifications, RL decisions, and healing. `/component1/analyze` explicitly reports drift as unavailable.

## Agent mapping status ? blocked on absent source

The current repository has no SMART collection-agent implementation or captured live payload. Searches of tracked files and source text for agent/collector/smartctl found only the Notebook 24 synthetic `agent_example_input.json`. Notebook 24 explicitly says the live agent repository was absent; Notebook 28 sets `agent_schema_verified_against_live_device=false`. The existing digital-twin sync API accepts `serial_number` plus an `observation` object containing normalized SMART fields, and does not establish a raw-value or device-model mapping. It is unsafe to infer that an agent uses this research contract.

| Accepted research field | Mapping used | Live-agent evidence needed |
|---|---|---|
| serial_number | Exact nonblank string; per-drive state key | Stable physical drive identifier, replacement/host collision behavior |
| model | Exact frozen model string | Agent model field and normalization policy |
| date | Explicit YYYY-MM-DD calendar date, unchanged | Timestamp source, timezone, end-of-day selection and sampling cadence |
| smart_5_raw | Observed raw number unchanged | Attribute 5 raw sector count encoding |
| smart_9_raw | Observed raw number unchanged | Attribute 9 power-on unit (expected hours; vendor encoding must be confirmed) |
| smart_194_raw | Observed raw number unchanged | Temperature representation (expected Celsius; packed bytes must not pass as a decoded temperature) |
| smart_197_raw | Observed raw number unchanged | Attribute 197 pending-sector count encoding |
| smart_198_raw | Observed raw number unchanged | Attribute 198 uncorrectable-sector count encoding |

No normalized fallback, missing-to-zero conversion, timestamp truncation, daily aggregation, or live agent adapter is implemented without that evidence. Required raw values must be finite nonnegative JSON numbers; null, absent, booleans, strings, negatives, and float32 overflow are rejected (422). A legitimate measured zero is retained. Unrelated fields, including failure labels, are ignored and never persisted or scored.

## Frozen inference and state policy

`app/ml/frozen_hdd.py` reproduces Notebook 24's float32 intake, float64 log1p and per-model scaling, final-day mask, masked adjacent differences, and reconstruction residuals. Primary output is sector error (SMART 5/197/198), compared with **strict >** to the existing per-model threshold. Temperature/joint outputs remain secondary. There is no training or calibration path.

The candidate lock hash is checked against the research handoff; checkpoint, preprocessing, training/scoring settings, thresholds, and telemetry contract hashes are checked against that lock before accepting records. Checkpoint model order and preprocessing metadata are checked too. Missing or modified artifacts fail closed with 503; there is no substitute model. The checkpoint is already available locally but ignored by Git: fresh checkouts must provision the original artifact at the lock's exact relative path. No datasets or generated score partitions are needed.

- Supported models: TOSHIBA MG07ACA14TA, ST12000NM0008, ST4000DM000, ST8000NM0055, HGST HMS5C4040BLE640.
- Days 1?29: warming; day 30 and each subsequent consecutive day: scored.
- Gap: discard rolling history and restart at one valid day. An invalid record is rejected without mutation; skipping that day therefore resets on the next accepted day. Corrections may be submitted before a later day is accepted.
- Exact latest-day duplicate: idempotent response with duplicate=true and the existing score/history. Conflicts return 409. Original raw numbers are compared before float32 rounding.
- Older days/backfills and model changes for an existing serial return 409.
- Unsupported models are displayed as unsupported and never scored.
- At most 30 accepted raw rows per drive are retained. State uses the twin's atomic JSON replacement pattern with fsync, plus a cross-process file lock and read-under-lock to prevent lost updates between local workers. Failed writes do not commit in-memory success. Corrupt state or a scorer-version mismatch fails closed.
- Default state: `app/runtime/hdd_shadow.json` (Git ignored); override with `HDD_SHADOW_STATE`. Use local shared filesystem storage for workers on one host. This is not a distributed multi-host database.

## UI and research limits

The page displays model, date, warmup count/status, primary anomaly score, frozen threshold, candidate flag, and scorer version. It states ?Research anomaly candidate; not a failure probability.? Failed refreshes mark previously loaded results as potentially stale. Regime/drift remain null and unavailable; SSD support is not validated. No production warning or automatic healing is dispatched.

Q4 blind evaluation remains complete and unchanged: 51 reserved first failures, 34 complete primary 14-day histories, 11 recalled, and 17 incomplete/unscoreable. Other methods performed well; the integration does not select a new winner. Preserve the handoff limitations: Q3-informed selection, few complete histories in some models, ST4000DM000 synthetic sensitivity misses, and four provisional model names still needing independent HDD type verification.

## Verification

From `05_backend`: `../.venv/Scripts/python -m pytest tests/test_component1_shadow.py -q`.
From repository root: `npm.cmd run build --prefix 06_frontend`.

Tests execute Notebook 24's original inference/scorer definitions independently on the same deterministic **synthetic** 35-day replay for all five models, comparing all residuals, scores, and flags (rtol=1e-5, atol=1e-6). They do not claim a replay of captured live-agent records or the notebook's original 50-drive dataset. Tests also cover persistence, invalid values, conflicts, unsupported models, gap reset, model changes, state/artifact version rejection, write failure, and concurrent store instances.

For browser verification, install the Playwright npm package in the local test environment, run the backend with a fresh temporary HDD_SHADOW_STATE, run Vite on port 5173, and run `node 05_backend/tests/shadow_browser_smoke.cjs` from the repository root (set NODE_PATH to the directory containing the locally installed playwright package). The script uses the existing demo login, posts synthetic daily records to the real API, and checks warming, day-30 values, and gap reset on the rendered page. It uses installed headless Edge; no deployment occurs.

The team must provide the tested collection-agent source/revision and a captured payload (including missing raw attributes), confirm device identity and model mapping, raw units/decoding, and the calendar-day timezone/end-of-day policy. Only then can a truthful agent adapter and representative live-agent replay be completed.

## Historical Backblaze replay

Fleet Intelligence includes a separate historical replay panel. The quarterly ZIP is streamed locally by `tools/backblaze_replay.py`; only its validated 30-row CSV is uploaded. Each upload creates a UUID replay identity and a separate durable rolling-history file, so historical tests never appear in live-agent device state or summary counts. See `tools/README.md` for the exact command and UI flow.

The replay path uses the existing `ShadowStore.ingest` method for each chronological daily record. It preserves warmup and scoring behavior and only adds session validation around the record. Replays cannot change the checkpoint, preprocessing, model list, hashes, thresholds, alert behavior, or healing behavior.

The local `data_Q4_2022.zip` compatibility check selected source serial `1030A001F97G`, model `TOSHIBA MG07ACA14TA`, for 2022-10-01 through 2022-10-30. Days 1–29 returned `warming`; day 30 returned sector anomaly score `0.00045455319923348725` against locked threshold `0.012183227576315403`, so the research candidate flag was `false`. This is one historical compatibility observation, not an accuracy result or permission to tune the detector.

## Results from this checkout

- Component 1 shadow and replay tests: **63 passed**; independent Notebook 24 parity across all five models and streamed Backblaze-style ZIP coverage included.
- Existing backend suite (before the final seven shadow checks were added, excluding the RL module while its dependency installed): **262 passed, 11 failed, 7 errors**. The failing conformal/online-feedback checks require the absent Component 2 checkpoint `02_component_2_digital_twin/results/rul_sequences/sequence_length_experiments/seq_30/bilstm/best_bilstm_model.pt`; calibration and RUL outputs consequently remain unavailable. No replacement was synthesized.
- Existing Component 4 RL API tests after installing their dependency: **9 passed**.
- Frontend production build: passed (existing >500 kB bundle-size advisory).
- Browser smoke: passed with installed headless Edge, real API and Vite, existing demo login, temporary shadow state; warming -> day 30 score/threshold/version -> gap reset, no page errors. The fixture is synthetic, not a captured agent record.
- `git diff --check`: passed. No research artifacts, datasets, thresholds, or daily-score partitions modified. Nothing deployed, no live alerts enabled.
