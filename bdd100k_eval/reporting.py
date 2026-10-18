"""Existing JSON, CSV and text-table output formats."""
import csv
import json
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional


def json_default(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    raise TypeError(
        f"Không thể chuyển {type(value).__name__} sang JSON"
    )


def write_json(path, obj):
    content = json.dumps(
        obj,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
        default=json_default,
    )
    Path(path).write_text(content, encoding="utf-8")


def format_table(headers: List[str], rows: List[List[str]], align_right: Optional[List[bool]] = None) -> str:
    if not rows:
        return "(Không có dữ liệu)"
    cols = len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for i in range(cols):
            val = str(row[i]) if i < len(row) else ""
            if len(val) > widths[i]:
                widths[i] = len(val)

    if align_right is None:
        align_right = [False] * cols

    def build_line(values, fill=" "):
        cells = []
        for i, val in enumerate(values):
            w = widths[i]
            cells.append(val.rjust(w, fill) if align_right[i] else val.ljust(w, fill))
        return "| " + " | ".join(cells) + " |"

    separator = "|-" + "-|-".join(["-" * w for w in widths]) + "-|"
    header_line = build_line(headers)
    data_lines = [build_line(row) for row in rows]
    return "\n".join([header_line, separator] + data_lines)


def export_csv(headers: List[str], rows: List[List[str]], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

