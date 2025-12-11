import pandas as pd
from datetime import datetime

def parse_iso(x):
    """
    Safely parse ISO datetime strings.
    Accepts formats like:
    - '2024-05-01T12:00:00Z'
    - '2024-05-01T12:00:00.123Z'
    - '2024-05-01'
    Returns datetime or None.
    """
    if not x:
        return None
    try:
        return pd.to_datetime(x, utc=True).to_pydatetime()
    except Exception:
        try:
            return datetime.fromisoformat(x)
        except:
            return None

def ensure_daily_index(df, date_col='date'):
    """
    Ensure the dataframe has every date between min and max.
    Fill missing rows with NaN.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    full_range = pd.date_range(df[date_col].min(), df[date_col].max(), freq="D")
    full_df = pd.DataFrame({date_col: full_range})
    merged = full_df.merge(df, on=date_col, how='left')
    return merged
