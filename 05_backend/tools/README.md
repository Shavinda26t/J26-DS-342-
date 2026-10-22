# Backblaze historical replay

The Fleet Intelligence replay panel uses a local extraction step because a quarterly Backblaze ZIP is hundreds of megabytes and the current FastAPI app has no durable upload-job infrastructure. The extractor streams CSV members from the ZIP and writes only a validated 30-row file. It never exports the `failure` column.

From the repository root:

```powershell
.\.venv\Scripts\python 05_backend\tools\backblaze_replay.py extract "D:\Downloads\data_Q4_2022.zip" --output backblaze_window.csv --overwrite
```

Then open Fleet Intelligence, expand **Replay Backblaze data**, choose `backblaze_window.csv`, inspect all 30 rows, and click **Run replay**. You can download the server-validated CSV before or after replay.

The dashboard creates a fresh UUID replay session for every selected CSV. Its identity is `REPLAY_<uuid>`, and its rolling state is stored separately from live shadow-agent history. An interrupted run can resume at the next unaccepted day. A completed replay cannot be appended to; select the CSV again to start another isolated run.

The CLI can also prepare and run a replay against the local backend:

```powershell
.\.venv\Scripts\python 05_backend\tools\backblaze_replay.py replay backblaze_window.csv --url http://127.0.0.1:8000
```

The backend accepts no quarterly ZIP upload. `/api/v1/component1/replay/prepare` is limited to a 64 KiB, exactly 30-row CSV with the exact frozen fields. This prevents a large synchronous request from occupying application memory. Historical scores are compatibility observations only: the checkpoint, preprocessing, hashes, thresholds, alerts, and healing behavior remain unchanged.

The official Backblaze dataset page documents one drive snapshot per row per day and provides the quarterly archives: <https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data>.
