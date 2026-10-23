"""Output-file helpers (URLs, JSON / CSV / TXT writers)."""
import json
from pathlib import Path

import pandas as pd

from config import OUTPUT_DIR
from common import clean_for_json


def to_output_url(path):
    """Absolute file path inside OUTPUT_DIR -> '/outputs/<relative>' (served by Flask). None when missing / outside OUTPUT_DIR."""
    if not path: return None
    try:
        relative = Path(path).resolve().relative_to(OUTPUT_DIR)
    except (OSError, ValueError):
        return None
    return f"/outputs/{relative.as_posix()}"


def output_path_from_url(url):
    """Inverse of to_output_url (used to delete the previous run's files)."""
    if not url: return None
    relative = url.split("?", 1)[0]
    if relative.startswith("/outputs/"): relative = relative[len("/outputs/"):]
    elif relative.startswith("outputs/"): relative = relative[len("outputs/"):]
    else: return None
    return OUTPUT_DIR / relative


def save_json(data, filename):
    out = OUTPUT_DIR / filename; out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f: json.dump(clean_for_json(data), f, indent=2, ensure_ascii=False)
    return str(out)


def save_csv(rows_or_df, filename):
    out = OUTPUT_DIR / filename; out.parent.mkdir(parents=True, exist_ok=True)
    df = rows_or_df if isinstance(rows_or_df, pd.DataFrame) else pd.DataFrame(rows_or_df)
    df.to_csv(out, index=False); return str(out)


def write_text_report(lines, filename):
    out = OUTPUT_DIR / filename; out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f: f.write("\n".join(lines))
    return str(out)
