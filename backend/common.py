"""Small helpers shared by the backend modules.

pick_device / ceil_to / json_safe are VERBATIM copies of the helpers in notebook 01 cell 6 (identical in 02/03/04).
Everything below the marker line is backend-only (JSON sanitising for the HTTP API).
"""
import math
import numpy as np
import torch


def pick_device(pref="auto"):
    """'auto' -> CUDA if available else CPU. Never hard-codes CUDA."""
    pref = (pref or "auto").lower()
    if pref == "cpu": return torch.device("cpu")
    if torch.cuda.is_available(): return torch.device("cuda")
    if pref in ("cuda", "gpu"): print("[warn] CUDA was requested but is not available -> using CPU.")
    return torch.device("cpu")


def json_safe(o):
    """Recursively convert numpy / tuple / Path objects into JSON-compatible python types."""
    if isinstance(o, dict): return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [json_safe(v) for v in o]
    if isinstance(o, np.generic): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (str, int, float, bool)) or o is None: return o
    return str(o)


def ceil_to(x, m=32): return int(math.ceil(x / m) * m)


# ============================================================================================================================
# backend-only helpers
# ============================================================================================================================
def clean_for_json(o):
    """json_safe + NaN / +-inf -> None.  Flask would otherwise emit the invalid JSON literal `NaN`, which browsers' JSON.parse rejects."""
    o = json_safe(o)
    if isinstance(o, dict): return {k: clean_for_json(v) for k, v in o.items()}
    if isinstance(o, list): return [clean_for_json(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o): return None
    return o
