# Adapted from the user-supplied backblaze_replay_tool.zip.
# Auto-selection now exports the first complete eligible window, avoiding a full-quarter second scan.
#!/usr/bin/env python3
"""Extract a frozen-scorer window from Backblaze daily CSVs, then replay it.

Uses Python's standard library. ZIPs are streamed; raw CSVs are not extracted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
import zipfile
from collections import deque
from datetime import date, timedelta
from pathlib import Path

FEATURES = (
    "smart_5_raw", "smart_9_raw", "smart_194_raw", "smart_197_raw", "smart_198_raw"
)
COLUMNS = ("serial_number", "model", "date", *FEATURES)
SUPPORTED = (
    "TOSHIBA MG07ACA14TA", "ST12000NM0008", "ST4000DM000",
    "ST8000NM0055", "HGST HMS5C4040BLE640",
)
DAY_FILE = re.compile(r"(20\d{2}-\d{2}-\d{2})\.csv$", re.IGNORECASE)


def valid_values(row: dict[str, str]) -> bool:
    try:
        return all(row[f].strip() != "" and math.isfinite(float(row[f]))
                   and 0 <= float(row[f]) <= 3.4028234663852886e38 for f in FEATURES)
    except (KeyError, ValueError, OverflowError):
        return False


def source_entries(inputs: list[Path]):
    entries = []
    for item in inputs:
        paths = sorted(item.glob("*.csv")) if item.is_dir() else [item]
        if not paths:
            raise ValueError(f"No CSV files in {item}")
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.suffix.lower() == ".zip":
                with zipfile.ZipFile(path) as z:
                    for info in z.infolist():
                        if info.is_dir() or "__MACOSX" in info.filename:
                            continue
                        if info.filename.lower().endswith(".csv"):
                            match = DAY_FILE.search(Path(info.filename).name)
                            entries.append((match.group(1) if match else None,
                                            path, info.filename))
            elif path.suffix.lower() == ".csv":
                match = DAY_FILE.search(path.name)
                entries.append((match.group(1) if match else None, path, None))
            else:
                raise ValueError(f"Expected a ZIP, CSV or CSV directory: {path}")
    named_days = [day for day, _, _ in entries if day is not None]
    if len(named_days) != len(set(named_days)):
        raise ValueError("Duplicate daily CSV date in archive/input")
    if not entries:
        raise ValueError("No CSV files were found in the input")
    # A combined CSV can hold many dates; it is sorted after filtering one serial.
    return sorted(entries, key=lambda x: (x[0] or "", str(x[1]), x[2] or ""))


def rows_of(entry, target_serial=None):
    _, path, member = entry
    if member is not None:
        z = zipfile.ZipFile(path)
        stream = io.TextIOWrapper(z.open(member), encoding="utf-8-sig", newline="")
    else:
        z = None
        stream = path.open("r", encoding="utf-8-sig", newline="")
    try:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header is None:
            return
        positions = {name: i for i, name in enumerate(header)}
        missing = set(COLUMNS) - positions.keys()
        if len(header) != len(set(header)):
            raise ValueError("Duplicate CSV column names")
        if missing:
            raise ValueError(f"{path} {member or ''}: missing {sorted(missing)}")
        idx_serial = positions["serial_number"]
        max_idx = max(positions[k] for k in COLUMNS)
        for number, values in enumerate(reader, start=2):
            if target_serial is not None:
                if len(values) <= idx_serial or values[idx_serial] != target_serial:
                    continue
            if len(values) <= max_idx:
                raise ValueError(f"{path} {member or ''} row {number}: truncated row")
            record = {key: values[positions[key]] for key in COLUMNS}
            if entry[0] is not None and record["date"] != entry[0]:
                raise ValueError(f"{path} {member or ''} row {number}: filename/date mismatch")
            yield record
    finally:
        stream.close()
        if z is not None:
            z.close()


def choose_serial(entries, allowed: set[str], days: int, end_date: str | None) -> str:
    if any(day is None for day, _, _ in entries):
        raise ValueError("Automatic selection needs daily YYYY-MM-DD.csv files. "
                         "For a combined CSV, pass --serial explicitly.")
    previous = {}
    streak = {}
    model_for = {}
    for entry in entries:
        day = date.fromisoformat(entry[0])
        ready = set()
        seen = set()
        for row in rows_of(entry):
            model, serial = row["model"], row["serial_number"]
            if not serial.strip():
                raise ValueError("Empty drive identity")
            if model not in allowed:
                continue
            if serial in seen:
                raise ValueError(f"Duplicate drive/date: {serial} {day}")
            seen.add(serial)
            if serial in model_for and model_for[serial] != model:
                raise ValueError(f"Model changed for {serial}")
            model_for[serial] = model
            if not valid_values(row):
                streak[serial] = 0
            elif previous.get(serial) == day - timedelta(days=1):
                streak[serial] = streak.get(serial, 0) + 1
            else:
                streak[serial] = 1
            previous[serial] = day
            if streak[serial] >= days:
                ready.add(serial)
        if ready and (end_date is None or str(day) == end_date):
            serial = min(ready)
            print(f"Selected supported {model_for[serial]} drive {serial} "
                  f"with {days} valid consecutive days ending {day}")
            return serial, str(day)
    raise ValueError(f"No supported drive has {days} consecutive valid daily records")


def collect(entries, serial: str, allowed: set[str]):
    selected = {}
    models = set()
    for entry in entries:
        for row in rows_of(entry, serial):
            try:
                date.fromisoformat(row["date"])
            except ValueError as exc:
                raise ValueError(f"Invalid ISO date for {serial}: {row['date']}") from exc
            if row["date"] in selected:
                raise ValueError(f"Duplicate drive/date: {serial} {row['date']}")
            selected[row["date"]] = row
            models.add(row["model"])
    if not selected:
        raise ValueError(f"Serial {serial} not found")
    if len(models) != 1 or next(iter(models)) not in allowed:
        raise ValueError(f"Serial {serial} has an unsupported or changing model: {models}")
    return [selected[k] for k in sorted(selected)]


def last_window(rows, days: int, end_date: str | None):
    history = deque(maxlen=days)
    previous = None
    found = None
    for row in rows:
        current = date.fromisoformat(row["date"])
        if not valid_values(row):
            history.clear()
        else:
            if previous != current - timedelta(days=1):
                history.clear()
            history.append(row)
            if len(history) == days and (end_date is None or row["date"] == end_date):
                found = list(history)
        previous = current
    if found is None:
        raise ValueError(f"No {days}-day consecutive valid window "
                         f"{'ending ' + end_date if end_date else 'for this drive'}")
    return found


def frozen_models(lock_path: Path | None) -> set[str]:
    if lock_path is None:
        component = Path(__file__).resolve().parents[2] / "01_component_1_behavioral_drift_genai"
        result = component / "results/single_model_q1q2_2021_v1"
        lock_path = result / "q4_blind_candidate_protocol_v1/q4_candidate_lock.json"
        handoff = json.loads((result / "q4_research_review_and_handoff_v1/research_integration_handoff.json").read_text())
        if hashlib.sha256(lock_path.read_bytes()).hexdigest() != handoff["candidate_lock_sha256"]:
            raise ValueError("Candidate lock hash mismatch")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    models = set(lock["future_test_drives_by_model"])
    if models != set(SUPPORTED):
        raise ValueError("Candidate lock model list does not match this frozen five-model tool")
    return models


def validate_window(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"Window must have exactly these columns: {COLUMNS}")
        rows = list(reader)
    return validate_rows(rows)


def validate_rows(rows):
    if len(rows) != 30:
        raise ValueError(f"Expected exactly 30 rows, found {len(rows)}")
    if len({r["serial_number"] for r in rows}) != 1 or len({r["model"] for r in rows}) != 1:
        raise ValueError("Window must contain one serial and one model")
    if rows[0]["model"] not in SUPPORTED:
        raise ValueError("Unsupported HDD model")
    for i, row in enumerate(rows):
        if not row["serial_number"].strip():
            raise ValueError("Empty serial_number")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["date"]):
            raise ValueError("Dates must be YYYY-MM-DD")
        date.fromisoformat(row["date"])
        if not valid_values(row):
            raise ValueError(f"Invalid raw SMART value on row {i + 2}")
        if i and date.fromisoformat(row["date"]) != date.fromisoformat(rows[i-1]["date"]) + timedelta(days=1):
            raise ValueError("Window dates must be consecutive and ordered")
    return rows


def extract(args):
    if args.days != 30:
        raise ValueError("The frozen model requires exactly 30 consecutive daily records")
    allowed = frozen_models(args.candidate_lock)
    entries = source_entries(args.source)
    if args.end_date:
        date.fromisoformat(args.end_date)
    serial = args.serial
    if not serial:
        serial, selected_end = choose_serial(entries, allowed, args.days, args.end_date)
        start = str(date.fromisoformat(selected_end) - timedelta(days=29))
        entries = [entry for entry in entries if start <= entry[0] <= selected_end]
    rows = collect(entries, serial, allowed)
    window = last_window(rows, args.days, args.end_date)
    path = args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {path}; pass --overwrite to replace it")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: row[k] for k in COLUMNS} for row in window)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"serial_number": serial, "model": window[0]["model"],
                "start_date": window[0]["date"], "end_date": window[-1]["date"],
                "days": len(window), "csv_sha256": digest,
                "source_paths": [str(p) for p in args.source],
                "failure_labels_exported": False,
                "description": "Raw SMART daily replay window; no training or threshold changes"}
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    validate_window(path)
    print(f"Wrote {path} ({window[0]['date']} to {window[-1]['date']}); SHA256 {digest}")
    print("Load the CSV in Fleet Intelligence > Replay Backblaze data, or use replay --url http://127.0.0.1:8000.")


def replay(args):
    """Prepare an isolated session, then use the real shadow intake one day at a time."""
    rows = validate_window(args.window)
    base = args.url.rstrip("/")
    from urllib.parse import urlparse
    if urlparse(base).hostname not in ("localhost", "127.0.0.1"):
        raise ValueError("Replay requires a local API URL")
    def request(path, body, content_type="application/json"):
        req = urllib.request.Request(base + path, data=body, headers={"Content-Type": content_type}, method="POST")
        with urllib.request.urlopen(req, timeout=args.timeout) as response:
            return json.load(response)
    prepared = request("/api/v1/component1/replay/prepare", args.window.read_bytes(), "text/csv")
    print("Isolated replay session:", prepared["replay_id"])
    for i, row in enumerate(prepared["rows"], 1):
        result = request("/api/v1/component1/shadow/records?replay_id=" + prepared["replay_id"], json.dumps(row).encode())
        print(f"Day {i:02d} {row['date']}: {result['status']}")
    print(json.dumps(result, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ex = commands.add_parser("extract", help="Generate a 30-day raw SMART CSV from ZIP/CSV input")
    ex.add_argument("source", type=Path, nargs="+", help="Backblaze quarterly ZIP, daily CSV, or CSV directory")
    ex.add_argument("--output", type=Path, required=True)
    ex.add_argument("--serial", help="Exact serial; otherwise choose a supported drive automatically")
    ex.add_argument("--end-date", help="Require this exact final YYYY-MM-DD date")
    ex.add_argument("--days", type=int, default=30)
    ex.add_argument("--candidate-lock", type=Path, help="Optional q4_candidate_lock.json model-list check")
    ex.add_argument("--overwrite", action="store_true")
    ex.set_defaults(func=extract)
    rp = commands.add_parser("replay", help="POST each daily row to a local Component 1 intake endpoint")
    rp.add_argument("window", type=Path)
    rp.add_argument("--url", default="http://127.0.0.1:8000", help="Local backend origin (not a route)")
    rp.add_argument("--timeout", type=float, default=30)
    rp.set_defaults(func=replay)
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
