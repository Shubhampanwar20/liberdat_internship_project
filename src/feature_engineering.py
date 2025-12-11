import json
import os
import pandas as pd
from src import config
from src.utils import parse_iso

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def find_records(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ("records","data","items","entries","events","measurements"):
            v = obj.get(k)
            if isinstance(v, list):
                return v
        for v in obj.values():
            if isinstance(v, list):
                return v
        return [obj]
    return []

def normalize_record_to_row(r):
    date_raw = r.get("date") or r.get("day") or r.get("timestamp") or r.get("startDate") or r.get("start")
    dt = parse_iso(date_raw)
    if dt is None:
        return None
    row = {"date": pd.to_datetime(dt.date())}
    for k in ("value","status","type","category","label"):
        if k in r:
            row[k] = r.get(k)
    # small additional fields
    for k in ("notes","note","description"):
        if k in r:
            row[k] = r.get(k)
    return row

def main():
    # determine categorical file path with multiple fallbacks
    cat_path = getattr(config, "CAT_FILE", None) or getattr(config, "CLIN_FILE", None) or os.path.join("data", "categorical-data.json")
    if not os.path.exists(cat_path):
        print("ERROR: categorical file not found at", cat_path)
        return

    os.makedirs(getattr(config, "OUTPUT_DIR", "outputs"), exist_ok=True)

    try:
        data = load_json(cat_path)
    except Exception as e:
        print("ERROR: cannot read categorical JSON:", e)
        return

    records = find_records(data)
    rows = []
    for r in records:
        row = normalize_record_to_row(r)
        if row is not None:
            rows.append(row)

    if not rows:
        print("ERROR: no valid categorical records found in", cat_path)
        return

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    outpath = os.path.join(getattr(config, "OUTPUT_DIR", "outputs"), "daily_clinical_features.csv")
    df.to_csv(outpath, index=False)
    print(f"Saved clinical features to {outpath} (rows: {len(df)})")

if __name__ == "__main__":
    main()
