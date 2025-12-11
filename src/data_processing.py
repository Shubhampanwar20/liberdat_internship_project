import json
import os
from collections import defaultdict
from datetime import timedelta
import pandas as pd
from src import config
from src.utils import parse_iso

def _to_naive(dt):
    if dt is None:
        return None
    try:
        if getattr(dt, "tzinfo", None) is not None:
            return dt.replace(tzinfo=None)
        if hasattr(dt, "tz") and dt.tz is not None:
            return pd.to_datetime(dt).tz_convert(None).to_pydatetime()
        return pd.to_datetime(dt).to_pydatetime()
    except Exception:
        t = pd.to_datetime(dt, utc=True)
        return t.tz_convert(None).to_pydatetime()

def split_interval_across_days(start_dt, end_dt, value, max_years=5, max_iters=20000):
    """
    Split value across calendar dates proportionally by overlap length.
    - If interval > max_years (default 5 years) treat as single-day allocation.
    - max_iters prevents accidental infinite loops.
    Returns dict(date -> float)
    """
    start_naive = _to_naive(start_dt)
    end_naive = _to_naive(end_dt) if end_dt is not None else None
    if start_naive is None:
        return {}
    if end_naive is None:
        end_naive = start_naive + timedelta(seconds=1)

    # Very large intervals -> suspicious. Avoid iterating for years.
    total_seconds = (end_naive - start_naive).total_seconds()
    if total_seconds < 0:
        # invalid interval: place on start date
        return {start_naive.date(): float(value)}
    max_seconds = max_years * 365 * 24 * 3600
    if total_seconds > max_seconds:
        # Warning to user (we don't print huge text here to keep CLI clean)
        print(f"WARNING: large interval detected {start_naive} -> {end_naive}, assigning to start date.")
        return {start_naive.date(): float(value)}

    if start_naive.date() == end_naive.date():
        return {start_naive.date(): float(value)}

    allocations = defaultdict(float)
    cur = start_naive
    iters = 0
    while cur.date() <= end_naive.date():
        iters += 1
        if iters > max_iters:
            # safety: break and assign remaining to current date
            allocations[cur.date()] += float(value)
            print("WARNING: hit iteration cap splitting interval; assigning remainder to current date")
            break
        next_midnight = (pd.Timestamp(cur.date()) + pd.Timedelta(days=1)).to_pydatetime()
        seg_end = min(next_midnight, end_naive)
        overlap = (seg_end - cur).total_seconds()
        frac = overlap / total_seconds if total_seconds > 0 else 1.0
        allocations[cur.date()] += float(value) * frac
        cur = seg_end
    return allocations

def process_timeseries(ts_records, metric_name='STEPS'):
    daily_acc = defaultdict(float)
    seen_any = False

    for rec in ts_records:
        try:
            count = rec.get('count') if rec.get('count') is not None else rec.get('value') or 0
        except Exception:
            count = 0
        metric = (rec.get('metric') or rec.get('type') or '')
        if str(metric).upper() != metric_name:
            continue

        start_raw = rec.get('start') or rec.get('startTime') or rec.get('from')
        end_raw = rec.get('end') or rec.get('endTime') or rec.get('to')
        start_dt = parse_iso(start_raw)
        end_dt = parse_iso(end_raw) if end_raw is not None else None

        if start_dt is None:
            continue
        if end_dt is None:
            end_dt = start_dt + timedelta(seconds=1)

        try:
            parts = split_interval_across_days(start_dt, end_dt, float(count))
            for d, v in parts.items():
                daily_acc[d] += v
            seen_any = True
        except Exception as e:
            # fallback: assign whole to start date
            daily_acc[start_dt.date()] += float(count)
            seen_any = True

    if not seen_any:
        return pd.DataFrame(columns=['date', 'Daily_Step_Count'])

    df = pd.DataFrame([{'date': pd.to_datetime(d), 'Daily_Step_Count': int(round(v))} for d, v in daily_acc.items()])
    df = df.sort_values('date').reset_index(drop=True)

    min_date = df['date'].min()
    max_date = df['date'].max()
    full_idx = pd.date_range(min_date, max_date, freq='D')
    full_df = pd.DataFrame({'date': full_idx})
    merged = full_df.merge(df, on='date', how='left')
    merged['Daily_Step_Count'] = merged['Daily_Step_Count'].fillna(0).astype(int)
    return merged

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(config.TS_FILE):
        print("ERROR: timeseries file not found at", config.TS_FILE)
        return

    try:
        ts = load_json(config.TS_FILE)
    except Exception as e:
        print("ERROR: failed to read timeseries JSON:", e)
        return

    # find list of records inside the JSON
    if isinstance(ts, dict):
        possible = ['records', 'data', 'items', 'timeseries', 'intervals']
        found = None
        for k in possible:
            if k in ts and isinstance(ts[k], list):
                found = ts[k]
                break
        if found is None:
            if any(isinstance(v, list) for v in ts.values()):
                for v in ts.values():
                    if isinstance(v, list):
                        found = v
                        break
        if found is None:
            if isinstance(ts, list):
                found = ts
            else:
                found = [ts]
        ts_records = found
    elif isinstance(ts, list):
        ts_records = ts
    else:
        print("ERROR: unexpected JSON structure for timeseries file")
        return

    daily = process_timeseries(ts_records, metric_name='STEPS')
    outpath = os.path.join(config.OUTPUT_DIR, 'daily_steps.csv')
    daily.to_csv(outpath, index=False)
    print(f"Saved daily steps to {outpath} (rows: {len(daily)})")

if __name__ == "__main__":
    main()
