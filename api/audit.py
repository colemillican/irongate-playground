import json
import os
from datetime import datetime
from typing import Any, Dict


def append_audit_event(event: Dict[str, Any], log_path: str = "logs/audit.jsonl") -> None:
    """
    Append-only audit log in JSON Lines format.
    Safe for regulated environments: who/what/when/retrieval/citations.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    event = dict(event)
    event["ts_utc"] = datetime.utcnow().isoformat() + "Z"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
